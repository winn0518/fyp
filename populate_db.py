from setup_vector_db import vector_db
from process_documents import texts

# Add documents to the vector database
ids = vector_db.add_documents(texts)
print(f"{len(ids)} documents added to the vector database")

