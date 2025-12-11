from typing import Any, Dict, List, Optional, Callable

from ..core.bm25_index import BM25Index
from .base_retriever import BaseRetriever
from .retriever_registry import RETRIEVER_REGISTRY


class BM25Retriever(BaseRetriever):
    retriever_type = "bm25"
    def __init__(
            self, 
            index: BM25Index, 
            top_k: int = 5,
            dedup_key: Optional[str] = None,
            dedup_fn: Optional[Callable] = None,
            config: Dict = None
        ):
        super().__init__(top_k=top_k, dedup_key=dedup_key, dedup_fn=dedup_fn)
        self.index = index
        self.logger.info("BM25Retriever initialized.")
        self.config = config or {}
        
    @classmethod
    def from_config(cls, config):
        index = BM25Index(config["index_config"], auto_load=True)
        return cls(
            index=index,
            top_k=config.get("top_k", 5),
            dedup_key=config.get("dedup_key", None),
            dedup_fn=config.get("dedup_fn", None),
            config=config
        )

    async def retrieve(
            self, 
            query: str, 
            top_k: int = None,
            max_retries: int = 3, 
            expansion_factor: int = 2,
            dedup_key: Optional[str] = None, 
            dedup_fn: Optional[Callable] = None
        ) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.top_k
        if dedup_key is None:
            dedup_key = self.dedup_key
        if dedup_fn is None:
            dedup_fn = self.dedup_fn

        scores, docs = self.index.search(
            query, 
            top_k,
            max_retries=max_retries,
            expansion_factor=expansion_factor,
            dedup_key=dedup_key,
            dedup_fn=dedup_fn
        )

        return [
            {
                "score": float(score),
                "metadata": doc
            }
            for score, doc in zip(scores, docs)
        ]

    async def retrieve_batch(
        self, 
        queries: List[str], 
        top_k: int = None,
        max_retries: int = 3, 
        expansion_factor: int = 2,
        dedup_key: Optional[str] = None, 
        dedup_fn: Optional[Callable] = None
    ) -> List[List[Dict[str, Any]]]:
        if top_k is None:
            top_k = self.top_k
        if dedup_key is None:
            dedup_key = self.dedup_key
        if dedup_fn is None:
            dedup_fn = self.dedup_fn

        scores_list, docs_list = self.index.search_batch(
            queries, 
            top_k,
            max_retries=max_retries,
            expansion_factor=expansion_factor,
            dedup_key=dedup_key,
            dedup_fn=dedup_fn
            )

        batch_results = []
        for scores, docs in zip(scores_list, docs_list):
            results = [
                {
                    "score": float(score),
                    "metadata": doc
                }
                for score, doc in zip(scores, docs)
            ]
            batch_results.append(results)

        return batch_results
    
RETRIEVER_REGISTRY[BM25Retriever.retriever_type] = BM25Retriever