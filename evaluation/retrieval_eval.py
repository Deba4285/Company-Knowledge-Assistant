import chromadb
from sentence_transformers import SentenceTransformer

from evaluation.eval_datasets import evaluation_dataset


# 1. Load embedding model

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


# 2. Connect to ChromaDB

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="company_documents"
)


# 3. Evaluation metrics

def precision_at_k(retrieved, relevant, k):

    retrieved_k = retrieved[:k]

    relevant_retrieved = set(retrieved_k) & set(relevant)

    return len(relevant_retrieved) / k


def recall_at_k(retrieved, relevant, k):

    retrieved_k = retrieved[:k]

    relevant_retrieved = set(retrieved_k) & set(relevant)

    if len(relevant) == 0:
        return 0

    return len(relevant_retrieved) / len(relevant)


def reciprocal_rank(retrieved, relevant):

    for rank, chunk_id in enumerate(retrieved, start=1):

        if chunk_id in relevant:

            return 1 / rank

    return 0


def hit_rate_at_k(retrieved, relevant, k):

    retrieved_k = retrieved[:k]

    return int(bool(set(retrieved_k) & set(relevant)))


# 4. Evaluate retrieval

K = 5

precision_scores = []
recall_scores = []
mrr_scores = []
hit_scores = []


print("\n========== RAG RETRIEVAL EVALUATION ==========\n")


for item in evaluation_dataset:

    question = item["question"]

    relevant_ids = item["relevant_chunk_ids"]

    # Convert question into embedding

    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # Retrieve documents

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=K
    )

    retrieved_ids = results["ids"][0]

    # Calculate metrics

    precision = precision_at_k(
        retrieved_ids,
        relevant_ids,
        K
    )

    recall = recall_at_k(
        retrieved_ids,
        relevant_ids,
        K
    )

    mrr = reciprocal_rank(
        retrieved_ids,
        relevant_ids
    )

    hit = hit_rate_at_k(
        retrieved_ids,
        relevant_ids,
        K
    )

    # Store scores

    precision_scores.append(precision)

    recall_scores.append(recall)

    mrr_scores.append(mrr)

    hit_scores.append(hit)

    # Display results

    print("Question:", question)

    print("Relevant IDs:", relevant_ids)

    print("Retrieved IDs:", retrieved_ids)

    print("Precision@5:", round(precision, 3))

    print("Recall@5:", round(recall, 3))

    print("MRR:", round(mrr, 3))

    print("Hit Rate@5:", hit)

    print("-" * 60)


# 5. Calculate average scores

print("\n========== FINAL EVALUATION RESULTS ==========\n")

print(
    "Average Precision@5:",
    round(sum(precision_scores) / len(precision_scores), 3)
)

print(
    "Average Recall@5:",
    round(sum(recall_scores) / len(recall_scores), 3)
)

print(
    "Average MRR:",
    round(sum(mrr_scores) / len(mrr_scores), 3)
)

print(
    "Average Hit Rate@5:",
    round(sum(hit_scores) / len(hit_scores), 3)
)