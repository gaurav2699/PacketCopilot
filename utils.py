import os
import qdrant_client
from qdrant_client.http.models import PointStruct
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Qdrant
from langchain.embeddings import SemanticChunker

def push_documents_to_qdrant(directory_path, embedding_model):
    # Initialize the Qdrant client
    client = qdrant_client.QdrantClient(host='localhost', port=6333)

    # Define the collection name
    collection_name = 'your_collection_name'

    # Load documents using DirectoryLoader
    loader = DirectoryLoader(directory_path, glob='*.json')  # Adjust the glob pattern as needed
    documents = loader.load()

    # Initialize the SemanticChunker
    chunker = SemanticChunker(embedding_model)

    # Chunk and embed documents
    points = []
    for doc in documents:
        chunks = chunker.chunk(doc.text)
        for chunk in chunks:
            point = PointStruct(
                id=chunk.metadata["id"],
                vector=chunk.vector,
                payload=chunk.metadata
            )
            points.append(point)

    # Push documents to Qdrant
    client.upsert(collection_name=collection_name, points=points)

    print("Documents have been successfully pushed to Qdrant!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python script.py <directory_path> <embedding_model>")
    else:
        directory_path = sys.argv
        embedding_model = sys.argv
        push_documents_to_qdrant(directory_path, embedding_model)