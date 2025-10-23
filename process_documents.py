from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types.doc.labels import DocItemLabel
from langchain_core.documents import Document
from setup_embedding import embeddings_tokenizer

sources = r"C:\Users\23022009\Downloads\olevel_electivehistory_2024_sa2_victoria - Copy-combined.pdf"


converter = DocumentConverter()
doc_id = 0

texts = [
    Document(
        page_content=chunk.text,
        metadata={"doc_id": (doc_id:=doc_id+1), "source": source}
    )
    for source in sources
    for chunk in HybridChunker(tokenizer=embeddings_tokenizer).chunk(converter.convert(source=source).document)
    if any(filter(lambda c: c.label in [DocItemLabel.TEXT, DocItemLabel.PARAGRAPH], iter(chunk.meta.doc_items)))
]

print(f"{len(texts)} document chunks created")
for document in texts:
    print(f"Document ID: {document.metadata['doc_id']}")
    print(f"Source: {document.metadata['source']}")
    print(document.page_content[:200], "...\n")  # print first 200 chars
