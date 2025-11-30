from sentence_transformers import SentenceTransformer
import numpy as np

_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str):
    return _embedder.encode([text])[0].tolist()

def embed_texts(texts):
    return _embedder.encode(texts).tolist()
