from chromadb import PersistentClient

def get_vector_db_collection():
    client = PersistentClient(path=".chromadb")
    collection = client.get_or_create_collection(
        "kai_documents",
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection
