# streamlit_app.py

import streamlit as st
from rag_system import RAGSystem
from db_operations import DatabaseManager
import os
import tempfile

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None

if 'db_manager' not in st.session_state:
    st.session_state.db_manager = None

if 'user_id' not in st.session_state:
    st.session_state.user_id = None

def initialize_systems():
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    qdrant_url = st.secrets["QDRANT_URL"]
    collection_name = "my_documents"
    db_url = st.secrets["DATABASE_URL"]
    
    rag_system = RAGSystem(openai_api_key, qdrant_url, collection_name)
    db_manager = DatabaseManager(db_url)
    
    return rag_system, db_manager

st.title("RAG Chatbot")

# Initialize systems if not already done
if st.session_state.rag_system is None or st.session_state.db_manager is None:
    st.session_state.rag_system, st.session_state.db_manager = initialize_systems()

# User authentication
username = st.text_input("Enter your username:")
if username:
    st.session_state.user_id = st.session_state.db_manager.get_or_create_user(username)

if st.session_state.user_id:
    # File Upload Section
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx'])
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Process and add the document to the RAG system
            st.session_state.rag_system.add_documents(tmp_file_path)
            st.success(f"File {uploaded_file.name} has been successfully added to the knowledge base.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
        finally:
            # Remove the temporary file
            os.unlink(tmp_file_path)

    # Chat interface
    st.subheader("Chat")

    # Load chat history
    chat_history = st.session_state.db_manager.get_chat_history(st.session_state.user_id)
    
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("What is your question?"):
        # Add user message to database
        st.session_state.db_manager.add_message(st.session_state.user_id, "user", prompt)
        
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

        # Add assistant response to database
        st.session_state.db_manager.add_message(st.session_state.user_id, "assistant", response["answer"])

    # Sidebar for system information and controls
    st.sidebar.title("System Info")
    st.sidebar.info("This chatbot uses a Retrieval-Augmented Generation (RAG) system to provide informed answers based on a document collection.")

    if st.sidebar.button("Clear Chat History"):
        # In a real application, you might want to add a confirmation dialog
        # and actually delete the history from the database
        st.experimental_rerun()
else:
    st.info("Please enter a username to start chatting.")
