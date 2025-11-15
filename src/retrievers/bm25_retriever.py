from typing import List, Dict, Any

from .base_retriever import BaseRetriever
from ..core.bm25_index import BM25Index

class BM25Retriever(BaseRetriever):

    def __init__(self, index: BM25Index, top_k: int = 5):
        super().__init__(top_k=top_k)
        self.index = index

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