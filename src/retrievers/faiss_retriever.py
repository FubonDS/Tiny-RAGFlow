import re
from typing import Any, Dict, List, Optional, Callable

import numpy as np

from ..core.client.embedding_rerank_client import EmbeddingModel
from ..core.faiss_index import FaissIndex
from .base_retriever import BaseRetriever
from .retriever_registry import RETRIEVER_REGISTRY


class FaissRetriever(BaseRetriever):
    retriever_type = "faiss"
    def __init__(
        self,
        index: FaissIndex,
        embedder: EmbeddingModel,
        top_k: int = 5,
        dedup_key: Optional[str] = None,
        dedup_fn: Optional[Callable] = None
    ):
        super().__init__(top_k=top_k, dedup_key=dedup_key, dedup_fn=dedup_fn)
        self.index = index
        self.embedder = embedder
        
        self.re_emoji = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+")
    
    @classmethod
    def from_config(cls, config: Dict):
        index = FaissIndex(config["index_config"], auto_load=True)
        embedder = EmbeddingModel(
            embedding_model=config["embedding_model"],
            config_path=config["model_config_path"]
        )
        return cls(
            index=index,
            embedder=embedder,
            top_k=config.get("top_k", 5),
            dedup_key=config.get("dedup_key", None),
            dedup_fn=config.get("dedup_fn", None)
        )
    
    async def retrieve(
            self, query: str, 
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
            
        query = self.normalize(query)
            
        query_vec = await self.embedder.embed_documents([query])
        query_vec = np.array(query_vec)
        
        scores, docs = self.index.search(
            query_vec, 
            top_k=top_k,
            max_retries=max_retries,
            expansion_factor=expansion_factor,
            dedup_key=dedup_key,
            dedup_fn=dedup_fn
        )
        
        return [
            {
                "score": float(score),
                "metadata": doc,
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
        ):
        if top_k is None:
            top_k = self.top_k
        if dedup_key is None:
            dedup_key = self.dedup_key
        if dedup_fn is None:
            dedup_fn = self.dedup_fn
            
        queries = [self.normalize(q) for q in queries]
        
        query_vecs = await self.embedder.embed_documents(queries)
        query_vecs = np.array(query_vecs)

        scores_list, docs_list = self.index.search_batch(
            query_vecs, 
            top_k=top_k,
            max_retries=max_retries,
            expansion_factor=expansion_factor,
            dedup_key=dedup_key,
            dedup_fn=dedup_fn
        )

        final_results = []
        for scores, docs in zip(scores_list, docs_list):
            result = [
                {"score": float(score), "metadata": doc}
                for score, doc in zip(scores, docs)
            ]
            final_results.append(result)
            
        return final_results

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        text = self._fullwidth_to_halfwidth(text)  
        text = self.re_emoji.sub("", text)
        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        return text
    
    def _fullwidth_to_halfwidth(self, text):
        """全形 → 半形"""
        res = []
        for char in text:
            code = ord(char)
            if code == 0x3000:       
                code = 0x20
            elif 0xFF01 <= code <= 0xFF5E:  
                code -= 0xFEE0
            res.append(chr(code))
        return "".join(res)
    
RETRIEVER_REGISTRY[FaissRetriever.retriever_type] = FaissRetriever