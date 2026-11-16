import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="company_documents"
)

results = collection.get(
    include=["documents", "metadatas"]
)

print("Total chunks:", len(results["ids"]))

keywords = [
    "working hours",
    "annual leave",
    "work from home",
    "remote work",
    "health insurance",
    "password",
    "security",
    "employee"
]

print("\n========== FILTERED CHUNKS ==========\n")

for i, document in enumerate(results["documents"]):

    text = document.lower()

    if any(word in text for word in keywords):

        print("Chunk ID:", results["ids"][i])

        print("Page:", results["metadatas"][i].get("page_label"))

        print("Content:", document)

        print("-" * 70)