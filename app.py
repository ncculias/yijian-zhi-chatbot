import logging
import subprocess
from typing import cast

import chainlit as cl
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


def get_shared_vector_store() -> QdrantVectorStore | None:
    global _SHARED_VECTOR_STORE
    if _SHARED_VECTOR_STORE is None:
        _SHARED_VECTOR_STORE = get_or_create_vector_store()
    return _SHARED_VECTOR_STORE


@cl.set_starters
async def set_starters():
    """首頁建議問題。

    讓使用者一眼看出這是可用自然語言提問的問答系統，而非關鍵字搜尋引擎；
    四則分別示範主題檢索、單一條目、詮釋分析與人物主題四種問法。
    """
    return [
        cl.Starter(
            label="雷擊報應的故事",
            message="夷堅志裡有哪些關於雷擊報應的故事？請列出三則並簡述。",
        ),
        cl.Starter(
            label="〈金釵辟鬼〉講什麼",
            message="〈金釵辟鬼〉這則故事的內容是什麼？",
        ),
        cl.Starter(
            label="鬼神故事的社會心態",
            message="夷堅志中關於冥府審判與鬼神的故事，反映了南宋什麼樣的社會心態？",
        ),
        cl.Starter(
            label="書中的女性形象",
            message="夷堅志裡有哪些以女性為主角的故事？她們呈現什麼樣的形象？",
        ),
    ]


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
                4. 盡可能提供詳細的答案。\n\
                5. 事實性內容只回答你有把握的部分，若問題超出你的知識範圍或無法回答，請告訴使用者。\n\
                6. 回答完畢後，另起一段以「你可以接著問：」開頭，列出 2 至 3 個承接本次回答、\n\
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
                "3. 補上被省略的主語、篇名，或序號（如「第 2 則」）實際指涉的條目名稱。\n"
                "4. 若原句本身已經完整、不看對話紀錄也能理解，就原樣輸出。\n"
                "5. 一律使用臺灣慣用的繁體中文。\n\n"
                "改寫後的問題：",
            ),
        ]
    )
    condense_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    cl.user_session.set("condense_runnable", condense_prompt | condense_model | StrOutputParser())

    cl.user_session.set("history", [])


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable
    condense_runnable = cast(Runnable, cl.user_session.get("condense_runnable"))
    vector_store = cast(QdrantVectorStore, cl.user_session.get("vector_store"))
    history = cast(list[BaseMessage], cl.user_session.get("history"))

    # 首輪直接用原句檢索；後續先依對話脈絡改寫，否則「第 2 個呢？」這類追問
    # 拿去查向量庫會檢索不到正確條目——模型即使有對話歷史，檢索端仍是瞎的。
    search_query = message.content
    if history:
        rewritten = (
            await condense_runnable.ainvoke(
                {"history_text": format_history_for_condense(history), "question": message.content}
            )
        ).strip()
        if rewritten and len(rewritten) <= MAX_SEARCH_QUERY_LENGTH:
            search_query = rewritten
            logger.info("檢索問題改寫: %r -> %r", message.content, search_query)
        else:
            logger.warning("改寫失敗（長度 %d），改用原句檢索: %r", len(rewritten), rewritten[:120])

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {
            "question": message.content,
            "context": vector_store.similarity_search(search_query, k=10),
            "history": history,
        },
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

    history.extend([HumanMessage(content=message.content), AIMessage(content=msg.content)])
    del history[:-MAX_HISTORY_MESSAGES]  # 只保留最近 MAX_HISTORY_MESSAGES 則
    cl.user_session.set("history", history)


if __name__ == "__main__":
    subprocess.run(["chainlit", "run", "app.py", "-h", "--host", str(settings.host), "--port", str(settings.port)])
