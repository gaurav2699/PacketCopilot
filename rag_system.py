# rag_system.py

import os
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class RAGSystem:
    def __init__(self, openai_api_key: str, qdrant_url: str, collection_name: str):
        self.openai_api_key = openai_api_key
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_dimension = 384  # Dimension for 'all-MiniLM-L6-v2' model
        
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
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

    def load_and_process_documents(self, directory: str) -> List[Document]:
        loader = DirectoryLoader(directory, glob="**/*.txt")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        return text_splitter.split_documents(documents)

    def setup_qa_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        return RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )

    def add_documents(self, directory: str):
        documents = self.load_and_process_documents(directory)
        self.vector_store.add_documents(documents)

    def query(self, question: str):
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata.get('source', 'Unknown source') for doc in result["source_documents"]]
        }
