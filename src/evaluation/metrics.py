import math
from typing import Dict, List


def extract_ids(results: List[Dict]) -> List[int]:
    """
    Extract document IDs from retriever results.

    Expected result format:
        [
          {
            "score": float,
            "metadata": {"id": int, "text": str}
          },
          ...
        ]

    Returns:
        [id1, id2, id3 ...] in ranked order
    """
    return [item["metadata"]["id"] for item in results]


def hit_rate_at_k(results: List[Dict], ground_truth_ids: List[int], k: int) -> float:
    """
    Hit-Rate@K: whether at least one ground truth ID
    appears in the top-k results.
    """
    top_k_ids = extract_ids(results[:k])
    return 1.0 if any(gt in top_k_ids for gt in ground_truth_ids) else 0.0


def recall_at_k(results: List[Dict], ground_truth_ids: List[int], k: int) -> float:
    """
    Recall@K: (# of ground truths retrieved in top-k) / (# of total ground truths)
    """
    top_k_ids = extract_ids(results[:k])
    hit_count = sum(1 for gt in ground_truth_ids if gt in top_k_ids)
    return hit_count / len(ground_truth_ids) if ground_truth_ids else 0.0


def precision_at_k(results: List[Dict], ground_truth_ids: List[int], k: int) -> float:
    """
    Precision@K: (# of retrieved relevant docs in top-k) / k
    """
    top_k_ids = extract_ids(results[:k])
    hit_count = sum(1 for id_ in top_k_ids if id_ in ground_truth_ids)
    return hit_count / k if k > 0 else 0.0


def mean_reciprocal_rank(results: List[Dict], ground_truth_ids: List[int]) -> float:
    """
    MRR: reciprocal rank of the first relevant document.
    """
    ranked_ids = extract_ids(results)
    for idx, doc_id in enumerate(ranked_ids):
        if doc_id in ground_truth_ids:
            return 1.0 / (idx + 1)
    return 0.0


def ndcg_at_k(results: List[Dict], ground_truth_ids: List[int], k: int) -> float:
    """
    NDCG@K based on binary relevance (1 if in ground truth else 0).
    """
    ranked_ids = extract_ids(results[:k])

    dcg = 0.0
    for idx, doc_id in enumerate(ranked_ids):
        if doc_id in ground_truth_ids:
            dcg += 1.0 / math.log2(idx + 2)

    ideal_hits = min(len(ground_truth_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def average(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0
