import asyncio
from typing import Any, Dict, List


from ..core.client.embedding_rerank_client import RerankingModel
from .base_reranker import BaseReranker
from .reranker_registry import RERANKER_REGISTRY


class GeneralReranker(BaseReranker):
    reranker_type = "general_reranker"
    def __init__(self, model_name: str, config_path: str):
        super().__init__()
        self.reranker = RerankingModel(
            reranking_model=model_name,
            config_path=config_path
        )
        self.logger.info(f"Initialized GeneralReranker with model: {model_name}")
        
    @classmethod
    def from_config(cls, config: Dict):
        return cls(
            model_name=config["model_name"],
            config_path=config["config_path"]
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

        tasks = []

        for q, docs in zip(queries, documents_list):
            tasks.append(
                asyncio.create_task(self.rerank(q, docs))
            )

        results = await asyncio.gather(*tasks)

        return results
    
RERANKER_REGISTRY[GeneralReranker.reranker_type] = GeneralReranker