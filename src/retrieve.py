import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load embedding model

embedding_model = SentenceTransformer(
"BAAI/bge-small-en-v1.5"
)

# 2. Connect to ChromaDB

client = chromadb.PersistentClient(
path="./chroma_db"
)

# 3. Get existing collection

collection = client.get_collection(
name="company_documents"
)

# 4. Take user question

question = input("Ask your question: ")

# 5. Convert question into embedding

question_embedding = embedding_model.encode(
question,
normalize_embeddings=True
).tolist()

# 6. Search ChromaDB

results = collection.query(
query_embeddings=[question_embedding],
n_results=5
)

# 7. Display retrieved chunks

print("\nRelevant Documents:\n")

for document in results["documents"][0]:
    print(document)
print("-" * 50)
