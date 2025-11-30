import streamlit as st
from setup_vector_db import get_vector_db_collection
from setup_embedding import embed_text, embed_texts
from setup_watsonx import ask_watsonx
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="KAI - Knowledge AI", layout="wide")
st.title("💡 KAI - Knowledge AI")

client, collection = get_vector_db_collection()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- FILE UPLOAD ----------
st.header("📂 Upload Documents")

uploaded_file = st.file_uploader("Upload txt/md/pdf", type=["txt", "md", "pdf"])

if uploaded_file:
    raw = uploaded_file.read().decode("utf-8", errors="ignore")
    st.success(f"{uploaded_file.name} uploaded!")

    vector = embedder.encode(raw).tolist()

    collection.add(
        ids=[uploaded_file.name],
        documents=[raw],
        embeddings=[vector]
    )

    st.info("📦 Saved to vector database.")

# ---------- QUESTION SECTION ----------
st.header("💬 Ask Questions")
query = st.text_input("Ask something:")

if st.button("Ask"):
    if not query.strip():
        st.warning("Enter a question.")
    else:
        query_vec = embedder.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_vec],
            n_results=3
        )

        context = "\n\n".join([d[0] for d in results["documents"]])

        if not context:
            st.error("No relevant documents found.")
        else:
            prompt = f"""
You are a helpful AI. Use ONLY the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

            answer = ask_watsonx(prompt)

            st.subheader("🧠 Answer")
            st.write(answer)
