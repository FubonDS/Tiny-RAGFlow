from typing import List, Dict, Any
from .base_retriever import BaseRetriever
from ..rerankers.general_reranker import GeneralReranker

class RerankRetriever(BaseRetriever):
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: GeneralReranker,
        top_k: int = 5,
    ):
        super().__init__(top_k=top_k)
        self.base_retriever = base_retriever
        self.reranker = reranker
        
    async def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.top_k
            
        candidates = await self.base_retriever.retrieve(query, top_k=top_k)
        reranked = await self.reranker.rerank(query, candidates)
        
        return reranked[:top_k]
            
    async def retrieve_batch(
        self, queries: List[str], top_k: int = None
    ) -> List[List[Dict[str, Any]]]:
        if top_k is None:
            top_k = self.top_k
        candidates_batch = await self.base_retriever.retrieve_batch(queries, top_k=top_k)
        reranked_batch = await self.reranker.rerank_batch(
            queries=queries,
            documents_list=candidates_batch
        )
        return reranked_batch
        
