# index_knowledge_base.py

import os
import argparse
from typing import List
from langchain.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredFileLoader,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import streamlit as st

# Import from your existing files
from rag_system import RAGSystem, pcap_to_json, process_large_file_line_by_line
from db_operations import DatabaseManager

# Load Streamlit secrets
st.secrets.load_secrets()

# Use the same Qdrant URL as in your RAG system
QDRANT_URL = st.secrets["QDRANT_URL"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
COLLECTION_NAME = "my_documents"  # You can change this if needed

def load_model():
    print("Loading Instructor XL Embeddings Model... This may take a few minutes.")
    return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

def get_file_loader(file_path: str):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.txt':
        return TextLoader(file_path)
    elif file_extension.lower() == '.pdf':
        return PyPDFLoader(file_path)
    elif file_extension.lower() in ['.docx', '.doc']:
        return Docx2txtLoader(file_path)
    elif file_extension.lower() == '.csv':
        return CSVLoader(file_path)
    elif file_extension.lower() == '.json':
        return JSONLoader(file_path, jq_schema='.', text_content=False)
    elif file_extension.lower() == '.etl':
        final_path = file_path + '.txt'
        command = f'pktmon etl2txt "{file_path}"'
        os.system(command)  # Note: This uses os.system instead of subprocess for simplicity
        txt_file = os.path.splitext(file_path)[0] + ".txt"
        process_large_file_line_by_line(txt_file, final_path, "filters.json")
        return TextLoader(final_path)
    elif file_extension.lower() == '.pcap':
        json_path = file_path + ".json"
        pcap_to_json(file_path, json_path)
        return JSONLoader(json_path, jq_schema='.', text_content=False)
    else:
        # For unknown file types, we'll use UnstructuredFileLoader which can handle many file types
        return UnstructuredFileLoader(file_path)

def process_directory(directory: str, embedding_model) -> List[Document]:
    loader = DirectoryLoader(directory, loader_cls=get_file_loader, recursive=True, use_multithreading=True)
    documents = loader.load()
    text_splitter = SemanticChunker(embedding_model)
    return text_splitter.split_documents(documents)

def initialize_vector_store(qdrant_url: str, collection_name: str, embedding_model):
    qdrant_client = QdrantClient(url=qdrant_url)
    collections = qdrant_client.get_collections().collections
    if not any(collection.name == collection_name for collection in collections):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),  # 768 for instructor-large
        )
    return Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embedding_model
    )

def main(directory: str):
    embedding_model = load_model()
    vector_store = initialize_vector_store(QDRANT_URL, COLLECTION_NAME, embedding_model)
    
    print(f"Processing documents in {directory}")
    documents = process_directory(directory, embedding_model)
    
    print(f"Adding {len(documents)} documents to the vector store")
    vector_store.add_documents(documents)
    
    print("Indexing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Index a directory of documents for RAG.')
    parser.add_argument('directory', type=str, help='Path to the directory containing documents to index.')
    
    args = parser.parse_args()
    
    main(args.directory)
