import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.chat_history import (BaseChatMessageHistory, InMemoryChatMessageHistory)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


model = ChatGroq(model="llama3-8b-8192")

prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "Hi! I'm Akash. I'm a software developer"),
        ("ai", "Hello Akash! How can I assist you today?"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ]
)

chain = prompt | model

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id is not None:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

config = {"configurable": {"session_id": "abc1"}}

response = chain_with_history.invoke({"question": "What's my name?"}, config=config)
print(response.content)

response = chain_with_history.invoke({"question": "What I do?"}, config=config)
print(response.content)
