from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

embeddings_model_path = "ibm-granite/granite-embedding-30m-english"
embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model_path)
embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)
