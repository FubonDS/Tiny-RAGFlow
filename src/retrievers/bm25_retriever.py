from typing import Any, Dict, List

from ..core.bm25_index import BM25Index
from .base_retriever import BaseRetriever
from .retriever_registry import RETRIEVER_REGISTRY


class BM25Retriever(BaseRetriever):
    retriever_type = "bm25"
    def __init__(self, index: BM25Index, top_k: int = 5):
        super().__init__(top_k=top_k)
        self.index = index
        
    @classmethod
    def from_config(cls, config):
        index = BM25Index(config["index_config"], auto_load=True)
        return cls(
            index=index,
            top_k=config.get("top_k", 5)
        )

    async def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.top_k

        scores, docs = self.index.search(query, top_k)

        return [
            {
                "score": float(score),
                "metadata": doc
            }
            for score, doc in zip(scores, docs)
        ]

    async def retrieve_batch(
        self, queries: List[str], top_k: int = None
    ) -> List[List[Dict[str, Any]]]:
        if top_k is None:
            top_k = self.top_k

        scores_list, docs_list = self.index.search_batch(queries, top_k)

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