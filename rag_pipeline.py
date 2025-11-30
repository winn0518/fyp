# rag_pipeline.py
from setup_vector_db import get_vector_db
from setup_model import model
from llama_index import VectorIndexRetriever, ResponseSynthesizer

if __name__ == "__main__":
    query = input("Enter your question: ")
    vector_db = get_vector_db()
    
    retriever = VectorIndexRetriever(vector_store=vector_db)
    synthesizer = ResponseSynthesizer(retriever=retriever, llm=model)
    
    answer = synthesizer.query(query)
    print("\n🧠 AI Answer:")
    print(answer)
