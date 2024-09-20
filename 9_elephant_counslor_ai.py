from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
import uuid

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///memory.db")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a child counselor AI that can interact with children aged 5 to 12. But for child you are elephant doll.
                Answer should provide supportive, empathetic responses and offer practical advice in a child-friendly manner.
                You have to speak in korean.""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

model = ChatOpenAI(model="gpt-4o-mini")
runnable = prompt | model

runnable_with_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def main():
    st.title("코끼리 인형")
    st.write("안녕 나는 너의 친구 코끼리 인형이야. 혹시 고민이 있으면 나에게 말해줘")
    user_input = st.text_input("하고 싶은 말을 입력해봐")
    if st.button("Send"):
        message=runnable_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": st.session_state.session_id}},
        )
        st.write(message.content)
        

if __name__ == "__main__":
    main()