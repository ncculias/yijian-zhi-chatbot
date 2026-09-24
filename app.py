import logging
import random
import re
import subprocess
from typing import cast

import chainlit as cl
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore

from index import get_or_create_vector_store
from settings import settings

logger = logging.getLogger(__name__)

_SHARED_VECTOR_STORE: QdrantVectorStore | None = None

# 對話歷史保留上限。一問一答為兩則，20 則約等於最近 10 輪。
# 留太多會稀釋問題改寫的準確度，也會墊高每次提問的 token 成本。
MAX_HISTORY_MESSAGES = 20

# 改寫後檢索句的長度上限。改寫是機率性的，模型偶爾會不改寫而直接作答，
# 產出一整段文字；超過此長度視為改寫失敗，退回使用者原句以免污染檢索。
MAX_SEARCH_QUERY_LENGTH = 100

# 供問題改寫參考的歷史則數與每則字數上限。改寫只需辨認指涉對象，
# 取最近幾則即可；截斷過長的回答可壓低成本並避免重點被稀釋。
CONDENSE_HISTORY_MESSAGES = 6
CONDENSE_MESSAGE_MAX_CHARS = 400

# 系統指令要求模型在回答結尾以此開頭列出延伸問題，這裡據以抽出做成按鈕。
FOLLOWUP_MARKER = "你可以接著問"
MAX_FOLLOWUP_LABEL = 60

# 語料每一則的固定開頭，由 split.py 建檔時寫入：《夷堅志》篇名(南宋洪邁撰)：正文
# 用來把篇名從正文切出來，在送進 prompt 時單獨標示。
DOCUMENT_TITLE_PATTERN = re.compile(r"^《夷堅志》(.+?)\(南宋洪邁撰\)：(.*)$", re.DOTALL)


def format_history_for_condense(history: list[BaseMessage]) -> str:
    """把對話歷史攤平成純文字，供問題改寫使用。

    只取最近幾則並截斷過長的回答：改寫只需要辨認代名詞與序號指涉的對象，
    給太多內容反而稀釋重點，也會增加每次提問的成本。
    """
    lines = []
    for m in history[-CONDENSE_HISTORY_MESSAGES:]:
        role = "使用者" if isinstance(m, HumanMessage) else "系統"
        content = " ".join(str(m.content).split())
        if len(content) > CONDENSE_MESSAGE_MAX_CHARS:
            content = content[:CONDENSE_MESSAGE_MAX_CHARS] + "…"
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


def format_context(docs: list[Document]) -> str:
    """把檢索結果整理成逐則標明篇名的純文字。

    不這麼做的話，list[Document] 會以 Python 的 repr 形式進入 prompt：十則原文
    擠成一行，中間夾著 id 與 metadata，條目之間只靠「), Document(」分隔。篇名
    雖然就寫在 page_content 開頭，卻被埋在這串雜訊裡，模型因此容易把正文首句
    當成篇名，也容易把相鄰條目的情節縫在一起。
    """
    blocks = []
    for index, doc in enumerate(docs, start=1):
        matched = DOCUMENT_TITLE_PATTERN.match(doc.page_content.strip())
        if matched:
            title, body = matched.group(1).strip(), matched.group(2).strip()
        else:
            # 格式不符時不猜，整段當正文保留：寧可不標篇名，也不要標錯
            title, body = "篇名不詳", doc.page_content.strip()
            logger.warning("檢索結果不符語料格式，無法取出篇名: %r", doc.page_content[:40])
        blocks.append(f"[資料 {index}] 篇名：〈{title}〉\n原文：{body}")
    return "\n\n".join(blocks)


def split_followups(answer_text: str) -> tuple[str, list[str]]:
    """把回答拆成本文與延伸問題。

    延伸問題改以按鈕呈現，因此要從本文移除，否則同樣的內容會既是文字又是按鈕。
    模型的格式不保證穩定，解析不出東西時原樣回傳——寧可維持純文字，也不要吃掉內容。
    """
    body, marker, tail = answer_text.partition(FOLLOWUP_MARKER)
    if not marker:
        return answer_text, []
    questions = []
    for line in tail.splitlines():
        q = re.sub(r"^[：:・·•\-\*\d\.\s]+", "", line).strip()
        if 4 <= len(q) <= MAX_FOLLOWUP_LABEL:
            questions.append(q)
    if not questions:
        return answer_text, []
    return body.rstrip(), questions[:3]


def get_shared_vector_store() -> QdrantVectorStore | None:
    global _SHARED_VECTOR_STORE
    if _SHARED_VECTOR_STORE is None:
        _SHARED_VECTOR_STORE = get_or_create_vector_store()
    return _SHARED_VECTOR_STORE


# 首頁建議問題的題庫，依「問法」分四組。每次開啟新對話時各組抽一題，
# 讓使用者每次看到的組合不同，但四種問法一定都示範得到。
STARTER_POOL: list[list[tuple[str, str]]] = [
    # 主題檢索：示範可以請系統橫跨全書找同類故事
    [
        ("雷擊報應的故事", "夷堅志裡有哪些關於雷擊報應的故事？請列出三則並簡述。"),
        ("冥府審判的故事", "夷堅志裡有哪些關於冥府審判的故事？請列出三則並簡述。"),
        ("狐狸精怪的故事", "夷堅志裡有哪些關於狐狸與精怪的故事？請列出三則並簡述。"),
        ("夢境預兆的故事", "夷堅志裡有哪些以夢境預兆為主的故事？請列出三則並簡述。"),
    ],
    # 單一條目：示範可以指名某一篇追問內容
    [
        ("〈金釵辟鬼〉講什麼", "〈金釵辟鬼〉這則故事的內容是什麼？"),
        ("〈鐵塔神〉講什麼", "〈鐵塔神〉這則故事的內容是什麼？"),
        ("〈建德妖鬼〉講什麼", "〈建德妖鬼〉這則故事的內容是什麼？"),
        ("〈鹽官孝婦〉講什麼", "〈鹽官孝婦〉這則故事的內容是什麼？"),
    ],
    # 詮釋分析：示範可以問意涵，而不只是問情節
    [
        ("鬼神故事的社會心態", "夷堅志中關於冥府審判與鬼神的故事，反映了南宋什麼樣的社會心態？"),
        ("因果報應的觀念", "夷堅志中的因果報應觀念是如何呈現的？反映了當時什麼樣的價值觀？"),
        ("洪邁的寫作意圖", "洪邁編纂夷堅志的意圖是什麼？書中的志怪題材有何時代背景？"),
        ("夢境的文化意義", "夷堅志中頻繁出現的夢境與預兆，在南宋的信仰脈絡中有什麼意義？"),
    ],
    # 人物主題：示範可以就某類人物橫向提問
    [
        ("書中的女性形象", "夷堅志裡有哪些以女性為主角的故事？她們呈現什麼樣的形象？"),
        ("書中的僧道人物", "夷堅志裡的僧人與道士扮演什麼角色？有哪些相關故事？"),
        ("書中的官員與士人", "夷堅志裡關於官員與士人的故事有哪些？呈現了什麼樣的形象？"),
        ("書中的商賈與庶民", "夷堅志裡關於商賈與市井庶民的故事有哪些？"),
    ],
]


@cl.set_starters
async def set_starters():
    """首頁建議問題。

    讓使用者一眼看出這是可用自然語言提問的問答系統，而非關鍵字搜尋引擎。
    每組抽一題並打亂順序：四種問法一定都示範得到，但每次進來看到的題目不同，
    避免同一組問題看久了像是系統只答得出這四題。
    """
    picks = [random.choice(group) for group in STARTER_POOL]
    random.shuffle(picks)
    return [cl.Starter(label=label, message=message) for label, message in picks]


@cl.on_chat_start
async def on_chat_start():
    vector_store = get_shared_vector_store()
    cl.user_session.set("vector_store", vector_store)
    model = ChatOpenAI(streaming=True, model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是個研究文史學者，非常擅長於回答文史相關議題與討論，尤其是對於南宋洪邁著的《夷堅志》有非常深入的研究，請你盡量嘗試回答使用者的疑問並遵守以下原則:\n\
                1. 一律使用臺灣慣用的繁體中文回答，不得出現簡體字，也不要夾雜非必要的英文詞彙。\n\
                2. 只回答與使用者提問相關的內容，若使用者問題與《夷堅志》無關或是無法理解，請告訴使用者。\n\
                3. 區分事實與詮釋：敘述情節、人物、時地等事實時須依據參考資料原文，不得自行增添；\n\
                當使用者詢問「為什麼」、「意涵」、「反映了什麼」這類問題時，不要只複述情節，\n\
                應進一步就宋代的社會、宗教、司法或文化背景提出分析，並以「就此推測」、「可能反映」等\n\
                措辭標明哪些內容屬於你的詮釋，與原文事實有所區隔。\n\
                4. 參考資料以「[資料 N] 篇名：〈某某〉／原文：……」的形式逐則列出，每一則是獨立的\n\
                一篇故事。敘述情節時必須指明出自哪一篇，不同篇的內容不得合併成同一件事來講；\n\
                若不同篇講的是不同人、不同事，請分開敘述。篇名一律以「篇名：」後面標示的為準，\n\
                不可把原文的第一句話當成篇名。\n\
                5. 條目結尾常有一句交代這則故事從何聽來、或某物後來為何人所得的句子，\n\
                通常以「⋯⋯說」、「⋯⋯云」作結。那是記述來源與流傳經過，不是故事情節。\n\
                回答「誰做了某件事」時，須依據正文中描述該動作的句子，不可改用結尾這類\n\
                交代後續或出處的句子作答，也不可把其中的人物寫成故事裡的角色。\n\
                並非每一則都有這樣的結尾；沒有的時候就不要提及來源，更不可自行補上\n\
                一個來源。參考資料以外的人名一律不得出現，包括本規則所舉的例子。\n\
                6. 回答前先核對提問的前提是否與原文相符。若使用者的問題預設了原文沒有\n\
                記載的事，例如問某人為何而死、但原文並未記載其死亡，必須明確指出原文\n\
                未記載該事，不得順著前提推測或補述情節。\n\
                7. 盡可能提供詳細的答案。\n\
                8. 事實性內容只回答你有把握的部分，若問題超出你的知識範圍或無法回答，請告訴使用者。\n\
                9. 回答完畢後，另起一段以「你可以接著問：」開頭，列出 2 至 3 個承接本次回答、\n\
                且本書確實談得到的延伸問題，每個問題獨立一行並以「・」開頭。若使用者的提問與\n\
                《夷堅志》無關或你無法解答，仍請提供延伸問題，藉此把話題引回本書談得到的主題。\n\
                以下是一些可能對於回答問題有幫助的參考資料：{context}\n\
                在回答問題前，請自行判斷這些參考資料是否能幫助回答問題，若問題超出參考資料所能解答的範圍，請在有把握的範圍內嘗試利用你對《夷堅志》的了解回答使用者問題，若無把握或能正確回答問題，請告知使用者。",
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                "{question}",
            ),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)

    # 追問改寫：把「第 2 則呢？」這類依賴前文的問題，補成不看對話也能理解的完整問題，
    # 供向量檢索使用。屬內部步驟，不需串流；temperature 設 0 讓改寫結果穩定。
    #
    # 這裡刻意「不」用 MessagesPlaceholder 把歷史當成真實對話輪次送入：那樣模型會
    # 順著對話繼續作答而非改寫。改為把歷史當成資料貼在指令中，並把改寫指令放在
    # 最後（緊貼生成點），模型才會把這視為一次文字轉換任務。
    condense_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一個「問題改寫器」。你只做改寫，絕對不回答任何問題。"),
            (
                "human",
                "以下是使用者與《夷堅志》問答系統的對話紀錄：\n"
                "---\n{history_text}\n---\n\n"
                "使用者最新的一句話是：「{question}」\n\n"
                "請把這句話改寫成一個「不看對話紀錄也能理解」的完整問題，供全文檢索使用，並遵守以下原則:\n"
                "1. 只輸出改寫後的問句本身，不要回答問題，也不要加上說明、引號或前綴。\n"
                "2. 必須是單一個問句，以問號結尾，長度不超過 50 字。\n"
                "3. 若這句話用序號或代名詞指稱前文提過的故事（如「第 2 則」「那一篇」「他」），\n"
                "必須從對話紀錄找出它實際指的是哪一篇，並把序號或代名詞換成該篇名。\n"
                "只改動詞句、不要只換掉量詞——把「第 2 個」寫成「第 2 則」不算完成改寫。\n"
                "例：對話紀錄中系統列出三則故事，第 2 則是〈鐵塔神〉，使用者問「第 2 個故事的詳細內容？」，\n"
                "應改寫為「《夷堅志》〈鐵塔神〉這則故事的詳細內容是什麼？」。\n"
                "4. 只有在對話紀錄中確實找不到對應篇名時，才保留原本的序號或代名詞。\n"
                "5. 若原句本身已經完整、不看對話紀錄也能理解，就原樣輸出。\n"
                "6. 一律使用臺灣慣用的繁體中文。\n\n"
                "改寫後的問題：",
            ),
        ]
    )
    condense_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    cl.user_session.set("condense_runnable", condense_prompt | condense_model | StrOutputParser())

    cl.user_session.set("history", [])


@cl.on_message
async def on_message(message: cl.Message):
    await answer(message.content)


@cl.action_callback("ask_followup")
async def on_followup(action: cl.Action):
    """使用者點了延伸問題按鈕：先把它當成使用者訊息顯示，再走一次正常的回答流程。"""
    question = str(action.payload.get("question", "")).strip()
    if not question:
        return
    await cl.Message(content=question, type="user_message").send()
    await answer(question)


async def answer(question: str):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable
    condense_runnable = cast(Runnable, cl.user_session.get("condense_runnable"))
    vector_store = cast(QdrantVectorStore, cl.user_session.get("vector_store"))
    history = cast(list[BaseMessage], cl.user_session.get("history"))

    # 首輪直接用原句檢索；後續先依對話脈絡改寫，否則「第 2 個呢？」這類追問
    # 拿去查向量庫會檢索不到正確條目——模型即使有對話歷史，檢索端仍是瞎的。
    search_query = question
    if history:
        rewritten = (
            await condense_runnable.ainvoke(
                {"history_text": format_history_for_condense(history), "question": question}
            )
        ).strip()
        if rewritten and len(rewritten) <= MAX_SEARCH_QUERY_LENGTH:
            search_query = rewritten
            logger.info("檢索問題改寫: %r -> %r", question, search_query)
        else:
            logger.warning("改寫失敗（長度 %d），改用原句檢索: %r", len(rewritten), rewritten[:120])

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {
            "question": question,
            "context": format_context(vector_store.similarity_search(search_query, k=10)),
            "history": history,
        },
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

    # 延伸問題改以按鈕呈現，點了直接送出，使用者不必重打一次
    body, followups = split_followups(msg.content)
    if followups:
        # 保留「你可以接著問：」這行標題，否則按鈕看起來像三行沒頭沒尾的文字
        msg.content = f"{body}\n\n**{FOLLOWUP_MARKER}：**"
        msg.actions = [cl.Action(name="ask_followup", payload={"question": q}, label=q) for q in followups]
        await msg.update()

    # 存進歷史的是移除延伸問題後的本文，避免那幾行問句干擾之後的追問改寫
    history.extend([HumanMessage(content=question), AIMessage(content=body)])
    del history[:-MAX_HISTORY_MESSAGES]  # 只保留最近 MAX_HISTORY_MESSAGES 則
    cl.user_session.set("history", history)


if __name__ == "__main__":
    subprocess.run(["chainlit", "run", "app.py", "-h", "--host", str(settings.host), "--port", str(settings.port)])
