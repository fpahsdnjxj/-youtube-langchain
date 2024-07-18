import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 환경변수 가져오기
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

#prompt template 만들기
prompt_template = PromptTemplate.from_template(
    """
    You are a translator. You translate input i give you.
    Translate input to {language} sentence.
    original sentence: {input}
    just answer translated sentence
    """
)



#llm 불러오기
from langchain_openai import ChatOpenAI
llm= ChatOpenAI(model="gpt-3.5-turbo",api_key=OPENAI_API_KEY)

#chain으로 연결하기
llm_chain = prompt_template| llm 

#invoke를 통해 답 출력하기
response=llm_chain.invoke({"language": 'japanese', "input": "나는 사과가 좋아"})
print(response.content)


