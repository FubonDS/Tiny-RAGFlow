import re
from typing import Any, Dict, List

import numpy as np

from ..core.client.embedding_rerank_client import (EmbeddingModel,
                                                   MultiVectorModel)
from ..core.qdrant_index import QdrantIndex
from .base_retriever import BaseRetriever
from .retriever_registry import RETRIEVER_REGISTRY


class QdrantMultivectorRetriever(BaseRetriever):
    retriever_type = "qdrant_multivector"
    def __init__(
        self,
        index: QdrantIndex,
        embedder: MultiVectorModel,
        top_k: int = 5,
    ):
        super().__init__(top_k=top_k)
        self.index = index
        self.embedder = embedder

        self.re_emoji = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+")
    

    @classmethod
    def from_config(cls, config: Dict):
        index = QdrantIndex(config["index_config"], auto_load=True)
        embedder = MultiVectorModel(
            model_path=config["embedding_model_path"],
        )
        return cls(
            index=index,
            embedder=embedder,
            top_k=config.get("top_k", 5)
        )
    
    async def retrieve(self, query: str, top_k: int = None, allowed_ids: List[int] = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.top_k
            
        query = self.normalize(query)

        query_vec = await self.embedder.embed_query(query)
        scores, docs = self.index.search(query_vec, top_k=top_k, allowed_ids=allowed_ids)

        return [
            {
                "score": float(score),
                "metadata": doc,
            }
            for score, doc in zip(scores, docs)
        ]
    
    async def retrieve_batch(self, queries: List[str], top_k: int = None, allowed_ids_list: List[List[int]] = None):
        if top_k is None:
            top_k = self.top_k
            
        queries = [self.normalize(q) for q in queries]
        
        query_vecs = await self.embedder.embed_query_batch(queries)
        
        scores_list, docs_list = self.index.search_batch(query_vecs, top_k, allowed_ids_list=allowed_ids_list)
        
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

class QdrantRetriever(BaseRetriever):
    retriever_type = "qdrant"
    def __init__(
        self,
        index: QdrantIndex,
        embedder: EmbeddingModel,
        top_k: int = 5,
    ):
        super().__init__(top_k=top_k)
        self.index = index
        self.embedder = embedder
        
        self.re_emoji = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+")
    
    @classmethod
    def from_config(cls, config: Dict):
        index = QdrantIndex(config["index_config"], auto_load=True)
        embedder = EmbeddingModel(
            embedding_model=config["embedding_model"],
            config_path=config["model_config_path"]
        )
        return cls(
            index=index,
            embedder=embedder,
            top_k=config.get("top_k", 5)
        )
    
    async def retrieve(self, query: str, top_k: int = None, allowed_ids: List[int] = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.top_k
            
        query = self.normalize(query)
            
        query_vec = await self.embedder.embed_documents([query])
        query_vec = np.array(query_vec)
        
        scores, docs = self.index.search(query_vec, top_k=top_k, allowed_ids=allowed_ids)
        
        return [
            {
                "score": float(score),
                "metadata": doc,
            }
            for score, doc in zip(scores, docs)
        ]
        
    async def retrieve_batch(self, queries: List[str], top_k: int = None, allowed_ids_list: List[List[int]] = None):
        if top_k is None:
            top_k = self.top_k
            
        queries = [self.normalize(q) for q in queries]
        
        query_vecs = await self.embedder.embed_documents(queries)
        query_vecs = np.array(query_vecs)
        
        scores_list, docs_list = self.index.search_batch(query_vecs, top_k, allowed_ids_list=allowed_ids_list)
        
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

RETRIEVER_REGISTRY[QdrantRetriever.retriever_type] = QdrantRetriever
RETRIEVER_REGISTRY[QdrantMultivectorRetriever.retriever_type] = QdrantMultivectorRetriever