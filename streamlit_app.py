import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.llms import OpenAI
import openai
import os

# Function to load data and create the VectorStoreIndex
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the text of War and Peace – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5,
                       system_prompt="You are an expert on War and Peace and your job is to answer topical questions. Assume that all questions are related to War and Peace. Keep your answers topical and based on facts – do not hallucinate features.")
        )
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

# Streamlit app code
st.set_page_config(page_title="Chat with War and Peace", page_icon="🦙", layout="centered",
                   initial_sidebar_state="auto", menu_items=None)

# Ensure the OpenAI API key is set
openai.api_key = os.environ.get('OPENAI_KEY')

st.title("Chat with the text of War and Peace, Maude translation 💬")

st.write("Source: Project Gutenberg: https://www.gutenberg.org/ebooks/2600")

# Load or retrieve the index
index = load_data()

# Initialize the chat engine
if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# User input prompt
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)
