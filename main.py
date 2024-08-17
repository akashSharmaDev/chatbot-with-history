import os
import yaml

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import streamlit as st

from app.utils import get_session_history


# Load environment variables
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
MODEL = os.getenv("MODEL")

# Initialize the model
model = ChatGroq(model=MODEL)


def get_character_initial_prompt(character_name: str) -> str:
    """
    Returns the initial character prompt message.
    """
    with open('roles.yaml', 'r') as file:
        data = yaml.safe_load(file)
    return data["characters"][character_name]


def chat_with_character(character_name: str):
    """
    Returns the character prompt template.
    """
    character_prompt = get_character_initial_prompt(character_name)

    return ChatPromptTemplate.from_messages(
        [
            ("system", character_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


# Streamlit UI setup
st.title("Character Chatbot")

# Dropdown for selecting a character
character_name = st.selectbox("Choose a character:", ["lawyer", "doctor", "engineer"])

# Create the prompt template based on selected character
prompt = chat_with_character(character_name)
chain = prompt | model

# Setup the chain with history
chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="messages",
)

# Chat session history
if "history" not in st.session_state:
    st.session_state.history = []

# Input for user messages
user_input = st.text_input("You: ", key="input")

if st.button("Send"):
    if user_input:
        # Append user message to the history
        st.session_state.history.append(HumanMessage(content=user_input))

        # Create config for the session
        config = {"configurable": {"session_id": "abc1"}}

        # Invoke the chain and get the response
        response = chain_with_history.invoke(
            {"messages": st.session_state.history},
            config=config
        )

        # Display the chatbot response
        st.write(f"{character_name}: {response.content}")

        # Append the chatbot's response to the history
        st.session_state.history.append(response)

# Show the conversation history
st.write("## Conversation History")
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        st.write(f"You: {msg.content}")
    else:
        st.write(f"{character_name}: {msg.content}")
