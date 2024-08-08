import chainlit as cl
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig

#prompt와 llm model 불러오기
prompt_template = PromptTemplate.from_template(
"""
You are a Sherlockian AI.
You get information from context: {context} and answer question: {question}
just using the context I gave you.
Tell me like a Sherlockian who prides himself on the Sherlock Holmes series.
Answer as if the Context I give was something you already knew. 
Don't show in the response that they were given. Answer using language i used for question.
"""
)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

#embedding function를 지정하고 vector store 불러오기
embedding = OpenAIEmbeddings()
db = FAISS.load_local("sherlock_holmes", embedding, allow_dangerous_deserialization=True)

retriever = db.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
)

@cl.on_chat_start
async def on_chat_start():    
    await cl.Message(content="""안녕하세요 저는 셜록홈즈 시리즈의 팬 셜로키안 AI입니다. 
                     셜록홈즈에 대한 내용은 무엇이든지 물어봐 주세요""").send()

@cl.on_message
async def on_message(message: cl.Message):
    response=rag_chain.invoke(message.content)
    await cl.Message(content=response.content).send()


