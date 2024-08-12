import os
import yaml

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.chat_history import (BaseChatMessageHistory, InMemoryChatMessageHistory)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


model = ChatGroq(model="llama3-8b-8192")

def get_character_initial_prompt(character_name: str) -> str:
    """
        Returns: Character prompt message
    """
    with open('roles.yaml', 'r') as file:
        data = yaml.safe_load(file)
    return data["characters"][character_name]

def chat_with_character(character_name: str):
    """
        Returns: Character prompt template
    """
    character_prompt = get_character_initial_prompt(character_name)

    return ChatPromptTemplate.from_messages(
        [
            ("system", character_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

prompt = chat_with_character("lawyer")

chain = prompt | model

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id is not None:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc1"}}

response = chain_with_history.invoke({"messages": [HumanMessage(content="Hi! My name is Akash.")]}, config=config)
print(response.content)

response = chain_with_history.invoke({"messages": [HumanMessage(content="What is my name?")]}, config=config)
print(response.content)
