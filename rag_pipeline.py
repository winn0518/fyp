from langchain.chains.retrieval import create_retrieval_chain
from ibm_granite_community.langchain.chains.combine_documents import create_stuff_documents_chain
from setup_model import model
from setup_vector_db import vector_db
from langchain_core.prompts import ChatPromptTemplate

query = "'It was Reagan who helped bring the Cold War to an end.' How far do you agree with this statement? Explain your answer."

prompt_template = ChatPromptTemplate.from_template("{input}")
combine_docs_chain = create_stuff_documents_chain(llm=model, prompt=prompt_template)
rag_chain = create_retrieval_chain(retriever=vector_db.as_retriever(), combine_docs_chain=combine_docs_chain)

output = rag_chain.invoke({"input": query})
print(output['answer'])
