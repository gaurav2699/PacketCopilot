# rag_system.py

import os
import subprocess
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, JSONLoader
from langchain.schema import Document
from preprocess import *
import streamlit as st
from langchain_openai import AzureChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

@st.cache_resource
def load_model():
    with st.spinner("Downloading Instructor XL Embeddings Model locally....please be patient"):
        embedding_model=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda"})
    return embedding_model

class RAGSystem:
    def __init__(self, openai_api_key: str, qdrant_url: str, collection_name: str):
        self.openai_api_key = openai_api_key
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_dimension = 768 # Dimension of the embeddings
        self.embedding_model = load_model()
        os.environ["AZURE_OPENAI_API_KEY"] = openai_api_key
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://openai2699.openai.azure.com/"
        print(f"OpenAI API Key: {os.getenv('AZURE_OPENAI_API_KEY')}")

        self.qdrant_client = QdrantClient(url=self.qdrant_url)
        self.vector_store = self.initialize_vector_store()
        self.qa_chain = self.setup_qa_chain()


    def initialize_vector_store(self):
        collections = self.qdrant_client.get_collections().collections
        if not any(collection.name == self.collection_name for collection in collections):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dimension, distance=Distance.COSINE),
            )
        return Qdrant(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embeddings=self.embedding_model)

    def returnSystemText(pcap_data : str) -> str:
        PACKET_WHISPERER = f"""
            {st.session_state['system_message']}
            log_info : {pcap_data}
        """
        return PACKET_WHISPERER

    def pcap_to_json(pcap_path, json_path):
        command = f'tshark -nlr {pcap_path} -T json > {json_path}'
        subprocess.run(command, shell=True)

    def load_and_process_documents(self, file_path: str) -> List[Document]:
        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() == '.txt':
            # loader = TextLoader(file_path)
            loader = TextLoader(file_path)
        elif file_extension.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension.lower() in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        elif file_extension.lower() == '.etl':
            final_path = file_path + '.txt'
            command = f'pktmon etl2txt "{file_path}"'
            # Execute the PowerShell command
            subprocess.run(["powershell", "-Command", command], capture_output=True, text=True)
            txt_file = os.path.splitext(file_path)[0] + ".txt"
            process_large_file_line_by_line(txt_file, final_path, "filters.json")
            loader = TextLoader(final_path)
        elif file_extension.lower() == '.pcap':
            json_path = file_path + ".json"
            self.pcap_to_json(file_path, json_path)
            loader = JSONLoader(file_path)

        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        self.pages = loader.load_and_split()
        self.text_splitter = SemanticChunker(self.embedding_model)

        # documents = loader.load()

        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=10000,
        #     chunk_overlap=200,
        #     length_function=len,
        # )
        return self.text_splitter.split_documents(self.pages)

    def setup_qa_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        llm = AzureChatOpenAI(
            azure_deployment="packetcopilot2",  # or your deployment
            api_version="2023-06-01-preview",  # or your api version
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            verbose=True
        )

    def add_documents(self, file_path: str):
        documents = self.load_and_process_documents(file_path)
        self.vector_store.add_documents(documents)

    def query(self, question: str):
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"]
        }
