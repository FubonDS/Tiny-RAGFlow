import asyncio
from typing import Any, Dict, List


from ..core.client.embedding_rerank_client import JinaForRerankingModel
from .base_reranker import BaseReranker
from .reranker_registry import RERANKER_REGISTRY


class JinaReranker(BaseReranker):
    reranker_type = "jina_reranker"
    def __init__(self, model_path: str):
        super().__init__()
        self.reranker = JinaForRerankingModel(
            model_path=model_path
        )
        self.logger.info(f"Initialized JinaReranker with model: {model_path}")
        
    @classmethod
    def from_config(cls, config: Dict):
        return cls(
            model_path=config["model_path"]
        )
        
    async def rerank(self, query: str, candidates: List[Dict[str, Any]]):
        """
        candidates:
        [
            {
                "score": float,
                "metadata": {"id":..., "text":...}
            },
            ...
        ]

        """
        texts = [c["metadata"]["text"] for c in candidates]
        
        new_scores = await self.reranker.rerank_documents(
            documents=texts,
            query=query
        )
        
        reranked_results = []
        for score, c in zip(new_scores, candidates):
            reranked_results.append({
                "score": float(score),
                "metadata": c["metadata"]
            })
            
        reranked_results = sorted(
            reranked_results,
            key=lambda x: x["score"],
            reverse=True
        )

        return reranked_results
    
    async def rerank_batch(
        self,
        queries: List[str],
        documents_list: List[List[Dict[str, Any]]]
    ) -> List[List[Dict[str, Any]]]:
        
        response = await self.reranker.rerank_documents_batch(
            documents_list=[
                [c["metadata"]["text"] for c in documents]
                for documents in documents_list
            ],
            query_list=queries
        )

        all_reranked_results = []
        for query_results, candidates in zip(response, documents_list):
            reranked_results = []
            for score, c in zip(query_results, candidates):
                reranked_results.append({
                    "score": float(score),
                    "metadata": c["metadata"]
                })
            reranked_results = sorted(
                reranked_results,
                key=lambda x: x["score"],
                reverse=True
            )
            all_reranked_results.append(reranked_results)   
        return all_reranked_results
    
RERANKER_REGISTRY[JinaReranker.reranker_type] = JinaReranker