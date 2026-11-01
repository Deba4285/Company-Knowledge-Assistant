from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

#load Pdf
doc=PyPDFLoader("data/fictional_company_rag_pages.pdf")
documents=doc.load()
#print(documents)

# chunking the pdf
chunking=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks=chunking.split_documents(documents)


#embedding the chunks
model=SentenceTransformer("BAAI/bge-small-en-v1.5")

texts=[chunk.page_content for chunk in chunks]
vector=model.encode(texts).tolist()

# 6. Connect to ChromaDB

client = chromadb.PersistentClient(
    path="./chroma_db"
)


# 7. Create collection

collection = client.get_or_create_collection(
    name="company_documents"
)


# 8. Store documents and embeddings

collection.add(
    ids=[str(i) for i in range(len(chunks))],
    documents=texts,
    embeddings=vector,
    metadatas=[chunk.metadata for chunk in chunks]
)

print("Documents stored successfully in ChromaDB!")
    