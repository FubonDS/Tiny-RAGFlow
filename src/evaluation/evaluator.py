from typing import Any, Dict, List

from tqdm import tqdm

from ..retrievers.base_retriever import BaseRetriever
from .dataset_loader import EvaluationDataset
from .metrics import (average, hit_rate_at_k, mean_reciprocal_rank, ndcg_at_k,
                      precision_at_k, recall_at_k)


class RetrieverEvaluator:
    """
    Standard evaluation module for retrievers.

    Features:
    ---------
    - async evaluation loop
    - supports multiple top-k metrics
    - compatible with retriever.retrieve(query)
    - expects result format:
        [
          {
            "score": 0.98,
            "metadata": {"id": 12, "text": "..."}
          },
          ...
        ]
    """
    def __init__(self, retriever: BaseRetriever, eval_dataset: EvaluationDataset):
        self.retriever = retriever
        self.eval_dataset = eval_dataset
        
    async def evaluate(
        self,
        top_k: List[int] = [5],
        batch_size: int = 8
    ) -> Dict[str, Any]:

        max_k = max(top_k)

        hit_rates = {k: [] for k in top_k}
        recalls = {k: [] for k in top_k}
        precisions = {k: [] for k in top_k}
        ndcgs = {k: [] for k in top_k}
        mrr_scores = []

        total_samples = len(self.eval_dataset)

        for start in tqdm(range(0, total_samples, batch_size), desc="Evaluating Retriever"):
            end = min(start + batch_size, total_samples)
            batch_samples = self.eval_dataset._samples[start:end]

            batch_queries = [s.query for s in batch_samples]
            batch_ground_truth = [s.ground_truth_ids for s in batch_samples]

            batch_results = await self.retriever.retrieve_batch(batch_queries, top_k=max_k)

            for results, gt_ids in zip(batch_results, batch_ground_truth):

                for k in top_k:
                    hit_rates[k].append(hit_rate_at_k(results, gt_ids, k))
                    recalls[k].append(recall_at_k(results, gt_ids, k))
                    precisions[k].append(precision_at_k(results, gt_ids, k))
                    ndcgs[k].append(ndcg_at_k(results, gt_ids, k))

                mrr_scores.append(mean_reciprocal_rank(results, gt_ids))

        summary = {}

        for k in top_k:
            summary[f"HitRate@{k}"] = average(hit_rates[k])
            summary[f"Recall@{k}"] = average(recalls[k])
            summary[f"Precision@{k}"] = average(precisions[k])
            summary[f"NDCG@{k}"] = average(ndcgs[k])

        summary["MRR"] = average(mrr_scores)

        return summary