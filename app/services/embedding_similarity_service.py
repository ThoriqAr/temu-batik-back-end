import numpy as np

from app.core.embedding_database_registry import (
    get_embedding_database
)


def normalize_embedding(
    embedding: np.ndarray
):

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


def find_top_neighbors(
    similarities: np.ndarray,
    labels: np.ndarray,
    top_k: int
):

    top_indices = np.argpartition(
        similarities,
        -top_k
    )[-top_k:]

    top_indices = top_indices[
        np.argsort(
            similarities[top_indices]
        )[::-1]
    ]

    top_labels = labels[top_indices]

    top_similarities = similarities[top_indices]

    return (
        top_indices,
        top_labels,
        top_similarities
    )


def compute_margin(
    top_similarities: np.ndarray
):

    if len(top_similarities) < 2:
        return 0.0

    return float(
        top_similarities[0] -
        top_similarities[1]
    )


def compute_agreement(
    top_labels: np.ndarray
):

    unique_labels, counts = np.unique(
        top_labels,
        return_counts=True
    )

    majority_index = np.argmax(counts)

    majority_label = unique_labels[majority_index]

    agreement = (
        counts[majority_index] /
        len(top_labels)
    )

    return (
        str(majority_label),
        float(agreement)
    )


def compute_statistics(
    similarities: np.ndarray,
    top_similarities: np.ndarray,
    majority_label: str,
    agreement: float
):

    return {
        "max_similarity":float(top_similarities[0]),
        "mean_similarity":float(similarities.mean()),
        "mean_topk_similarity":float(np.mean(top_similarities)),
        "std_topk_similarity":float(np.std(top_similarities)),
        "margin":compute_margin(top_similarities),
        "agreement":agreement
    }


def build_neighbor_list(
    top_indices: np.ndarray,
    top_labels: np.ndarray,
    top_similarities: np.ndarray
):

    neighbors = []

    for rank, (
        index,
        label,
        similarity
    ) in enumerate(

        zip(
            top_indices,
            top_labels,
            top_similarities
        ),

        start=1

    ):
        
        similarity = float(similarity)

        neighbors.append({
            "rank": rank,
            "index": int(index),
            "label": str(label),
            "similarity": similarity,
            "distance": float(
                1.0 - similarity
            )
        })

    return neighbors


def compute_similarity(
    query_embedding: np.ndarray,
    model_name: str,
    top_k: int = 5
):

    database = get_embedding_database(
        model_name
    )

    if database is None:

        raise ValueError(
            f"Embedding database '{model_name}' not found."
        )

    embeddings = database["embeddings"]

    labels = database["labels"]

    query_embedding = normalize_embedding(
        query_embedding.astype(np.float32)
    )

    similarities = embeddings @ query_embedding

    (
        top_indices,
        top_labels,
        top_similarities
    ) = find_top_neighbors(
        similarities,
        labels,
        top_k
    )

    majority_label, agreement = compute_agreement(top_labels)

    statistics = compute_statistics(
        similarities,
        top_similarities,
        majority_label,
        agreement
    )


    neighbors = build_neighbor_list(
        top_indices,
        top_labels,
        top_similarities
    )


    return {

        "prediction": {
            "nearest_label":str(top_labels[0]),
            "majority_label":majority_label
        },

        "statistics":statistics,
        "neighbors":neighbors
    }