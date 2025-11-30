# populate_db.py
from process_documents import process_documents
from setup_vector_db import get_vector_db_collection
from setup_embedding import embed_texts

if __name__ == "__main__":
    sources = [
        r"C:\Users\23022009\Desktop\fypdoc\olevel_electivehistory_2024_sa2_victoria - Copy-combined.pdf"
    ]
    docs = process_documents(sources)
    collection = get_vector_db_collection()
    if docs:
        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]
        embeddings = embed_texts(texts)
        # add to chroma
        ids = [f"doc_{i}" for i in range(len(texts))]
        collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=embeddings.tolist())
        print(f"✅ {len(texts)} document chunks added to ChromaDB and saved.")
    else:
        print("⚠️ No valid documents to add.")
