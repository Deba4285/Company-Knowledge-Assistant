import chromadb
from sentence_transformers import SentenceTransformer

from evaluation.eval_datasets import evaluation_dataset

# Load model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Connect ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    name="company_documents"
)

# Test different K values
k_values = [3, 5, 10]

for k in k_values:

    precision_scores = []
    recall_scores = []

    print("\n========== K =", k, "==========")

    for item in evaluation_dataset:

        question = item["question"]
        relevant_ids = set(item["relevant_chunk_ids"])

        question_embedding = model.encode(
            question,
            normalize_embeddings=True
        ).tolist()

        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=k
        )

        retrieved_ids = results["ids"][0]

        relevant_retrieved = (
            set(retrieved_ids) & relevant_ids
        )

        precision = len(relevant_retrieved) / k

        recall = (
            len(relevant_retrieved) / len(relevant_ids)
        )

        precision_scores.append(precision)
        recall_scores.append(recall)

    print(
        "Average Precision:",
        round(sum(precision_scores) / len(precision_scores), 3)
    )

    print(
        "Average Recall:",
        round(sum(recall_scores) / len(recall_scores), 3)
    )