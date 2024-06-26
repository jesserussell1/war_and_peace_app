import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
import openai


# Create a Settings object with the desired configuration
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5,
                      system_prompt="You are an expert on War and Peace and your job is to answer topical questions. Assume that all questions are related to War and Peace. Keep your answers topical and based on facts – do not hallucinate features.")


st.set_page_config(page_title="Chat with War and Peace", page_icon="🦙", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)

openai.api_key = st.secrets["openai_key"]

st.title("Chat with the text of War and Peace, Maude translation 💬")

st.write("Source: Project Gutenberg: https://www.gutenberg.org/ebooks/2600")


if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about War and Peace!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the text of War and Peace – hang tight! This should take 1-2 minutes."):
        # SimpleDirectoryReader will use the global settings
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(docs)
        return index

index = load_data()


if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)


