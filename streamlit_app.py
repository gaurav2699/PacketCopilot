# streamlit_app.py

import streamlit as st
from rag_system import RAGSystem
import os

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_rag_system():
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    qdrant_url = st.secrets["QDRANT_URL"]
    collection_name = "my_documents"
    
    rag_system = RAGSystem(openai_api_key, qdrant_url, collection_name)
    
    # Load documents (you might want to make this configurable)
    documents_directory = "path/to/your/documents"
    rag_system.add_documents(documents_directory)
    
    return rag_system

st.title("RAG Chatbot")

# Initialize RAG system if not already done
if st.session_state.rag_system is None:
    st.session_state.rag_system = initialize_rag_system()

# Chat interface
st.subheader("Chat")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get response from RAG system
    response = st.session_state.rag_system.query(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        st.write(response["answer"])
        st.write("Sources:")
        for source in response["sources"]:
            st.write(f"- {source}")

    st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})

# Sidebar for system information and controls
st.sidebar.title("System Info")
st.sidebar.info("This chatbot uses a Retrieval-Augmented Generation (RAG) system to provide informed answers based on a document collection.")

if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.experimental_rerun()
