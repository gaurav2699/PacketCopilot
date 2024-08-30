# rag_system.py

import os
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class RAGSystem:
    def __init__(self, openai_api_key: str, qdrant_url: str, collection_name: str):
        self.openai_api_key = openai_api_key
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_dimension = 384  # Dimension for 'all-MiniLM-L6-v2' model

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
            embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )

    def load_and_process_documents(self, file_path: str) -> List[Document]:
        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() == '.txt':
            loader = TextLoader(file_path)
        elif file_extension.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension.lower() in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        return text_splitter.split_documents(documents)

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
            return_source_documents=True,
            verbose=True
        )

    def add_documents(self, file_path: str):
        documents = self.load_and_process_documents(file_path)
        self.vector_store.add_documents(documents)

    def query(self, question: str):
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata.get('source', 'Unknown source') for doc in result["source_documents"]]
        }
