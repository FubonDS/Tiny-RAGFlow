import asyncio
from typing import List, Dict, Any, Optional

from .base_retriever import BaseRetriever

class HybridRetriever(BaseRetriever):
    
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        fusion_method: str = "rrf",
        weights: Optional[List[float]] = None,
        rrf_k: int = 60,
        top_k: int = 5,
    ):
        super().__init__(top_k=top_k)
        self.retrievers = retrievers
        self.fusion_method = fusion_method
        self.weights = weights
        self.rrf_k = rrf_k
        
        if self.fusion_method == "weighted":
            if weights is None or len(weights) != len(retrievers):
                raise ValueError("Weights must be provided and match the number of retrievers for weighted fusion.")
            
    
    async def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.top_k
            
        tasks = [r.retrieve(query, top_k=top_k) for r in self.retrievers]
        result_lists = await asyncio.gather(*tasks)
        
        fused = self._fuse_results(result_lists)
        
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k] if top_k else fused
    
    async def retrieve_batch(
        self, queries: List[str], top_k: int = None
    ):
        tasks = [r.retrieve_batch(queries, top_k=top_k or self.top_k) for r in self.retrievers]
        batch_result_lists = await asyncio.gather(*tasks)
        
        final = []
        Q = len(queries)
        for qi in range(Q):
            per_query_lists = [
                batch_result_lists[ri][qi] for ri in range(len(self.retrievers))
            ]
            fused = self._fuse_results(per_query_lists)
            fused.sort(key=lambda x: x["score"], reverse=True)
            final.append(fused[:top_k] if top_k else fused)
        
        return final
    
    def _fuse_results(self, result_lists: List[List[Dict[str, Any]]]):
        if self.fusion_method == "rrf":
            return self._fusion_rrf(result_lists)
        elif self.fusion_method == "weighted":
            return self._fusion_weighted(result_lists)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
    def _fusion_rrf(self, result_lists: List[List[Dict[str, Any]]]):
        """
        result_lists:
            [
                [ {score, metadata}, ... ],  # retriever1
                [ {score, metadata}, ... ],  # retriever2
            ]

        回傳：
            [
                { "score": fused_score, "metadata": {...} },
                ...
            ]
        """
        score_map = {}
        for results in result_lists:
            for rank, item in enumerate(results):
                doc = item['metadata']
                doc_id = doc.get("id")
                
                rrf_score = 1.0 / (self.rrf_k + rank + 1)
                
                if doc_id not in score_map:
                    score_map[doc_id] = {
                        "score": rrf_score,
                        "metadata": doc
                    }
                else:
                    score_map[doc_id]["score"] += rrf_score
        
        return list(score_map.values())
    
    def _fusion_weighted(self, result_lists: List[List[Dict[str, Any]]]):
        score_map = {}
        
        for retriever_idx, results in enumerate(result_lists):
            weight = self.weights[retriever_idx]
            
            scores = [item["score"] for item in results]
            min_s, max_s = min(scores), max(scores)
            denom = (max_s - min_s) or 1.0

            for item in results:
                doc = item["metadata"]
                doc_id = doc.get("id")

                normalized = (item["score"] - min_s) / denom
                fused_score = normalized * weight

                if doc_id not in score_map:
                    score_map[doc_id] = {
                        "score": fused_score,
                        "metadata": doc
                    }
                else:
                    score_map[doc_id]["score"] += fused_score
        
        return list(score_map.values())