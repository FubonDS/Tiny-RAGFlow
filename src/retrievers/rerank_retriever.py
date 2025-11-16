from typing import List, Dict, Any
from .base_retriever import BaseRetriever
from ..rerankers.general_reranker import GeneralReranker
from .retriever_registry import RETRIEVER_REGISTRY


class RerankRetriever(BaseRetriever):
    retriever_type = "rerank"
    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: GeneralReranker,
        top_k: int = 5,
    ):
        super().__init__(top_k=top_k)
        self.base_retriever = base_retriever
        self.reranker = reranker
        
    @classmethod
    def from_config(cls, config: Dict):
        r_cfg = config["retriever"]
        r_type = r_cfg["type"]
        r_conf = r_cfg["config"]

        if r_type not in RETRIEVER_REGISTRY:
            raise ValueError(f"Unknown retriever type: {r_type}")

        BaseRetrieverClass = RETRIEVER_REGISTRY[r_type]
        base_retriever = BaseRetrieverClass.from_config(r_conf)

        reranker = GeneralReranker.from_config(config["reranker"])

        top_k = config.get("top_k", 5)

        return cls(
            base_retriever=base_retriever,
            reranker=reranker,
            top_k=top_k,
        )
        
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
        
