import subprocess
from typing import cast

import chainlit as cl
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore

from index import get_or_create_vector_store
from settings import settings

_SHARED_VECTOR_STORE: QdrantVectorStore | None = None


def get_shared_vector_store() -> QdrantVectorStore | None:
    global _SHARED_VECTOR_STORE
    if _SHARED_VECTOR_STORE is None:
        _SHARED_VECTOR_STORE = get_or_create_vector_store()
    return _SHARED_VECTOR_STORE


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
                以下是一些可能對於回答問題有幫助的參考資料：{context}\n\
                在回答問題前，請自行判斷這些參考資料是否能幫助回答問題，若問題超出參考資料所能解答的範圍，請在有把握的範圍內嘗試利用你對《夷堅志》的了解回答使用者問題，若無把握或能正確回答問題，請告知使用者。",
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable
    vector_store = cast(QdrantVectorStore, cl.user_session.get("vector_store"))
    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content, "context": vector_store.similarity_search(message.content, k=10)},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()


if __name__ == "__main__":
    subprocess.run(["chainlit", "run", "app.py", "-h", "--host", str(settings.host), "--port", str(settings.port)])
