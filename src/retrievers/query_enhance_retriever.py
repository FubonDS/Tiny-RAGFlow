import re
from typing import Any, Dict, List, Optional, Callable

import numpy as np
import json
import asyncio

from ..core.client.embedding_rerank_client import EmbeddingModel
from ..core.faiss_index import FaissIndex
from .base_retriever import BaseRetriever
from .retriever_registry import RETRIEVER_REGISTRY
from ..core.client.llm_client import AsyncLLMChat

QUERY_EXPAND_PROMPT = """
You are an AI language model assistant. Your task is to generate 3 \n 
different versions of the given user question to retrieve relevant documents from a vector \n
database. By generating multiple perspectives on the user question, your goal is to help \n
the user overcome some of the limitations of the distance-based similarity search. \n

範例：
原始問題：我想知道信用卡年費怎麼收？
擴展查詢：
{{
  "original_query": "我想知道信用卡年費怎麼收？",
  "expanded_queries": [
    "信用卡的年費是怎麼計算的？",
    "哪些信用卡需要支付年費？費用是多少？",
    "我可以免除信用卡年費嗎？有什麼條件？"
  ]
}}

請根據以下原始問題產生擴展查詢，並以相同格式輸出：

原始問題：{user_query}
擴展查詢：
"""

class QueryEnhanceRetriever:
    retriever_type = "queryenhance"
    def __init__(self, 
                 retrievers,
                 llmchater,
                 fusion_method: str = "rrf",
                 top_k: int = 5,
                 rrf_k: int = 60,
                 weights: Optional[List[float]] = None
                ):
        self.retrievers = retrievers
        self.llmchater = llmchater
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.weights = weights
        self.top_k = top_k
    
    @classmethod
    def from_config(cls, config:Dict):
        """
        expansion_config = {
        "retrievers": [
                {
                    "type": "faiss",
                    "config": {
                        "index_config": "./config/faiss.yaml",
                        "embedding_model": "m3e-base",
                        "model_config_path": "./config/models.yaml",
                        "top_k": 3
                    }
                },
                {
                    "type": "bm25",
                    "config": {
                        "index_config": "./config/bm25.yaml",
                        "top_k": 3
                    }
                }
            ],
            "fusion_method": "rrf",
            "rrf_k": 60,
            "top_k": 5,

            "llm_model": "Qwen2.5-32B-Instruct-GPTQ-Int4",
            "model_config_path": "./config/models.yaml"
        }
        """
        llmchater = AsyncLLMChat(
            model=config["llm_model"],
            config_path=config["model_config_path"]
            )
        
        retrievers_cfg = config["retrievers"]
        retrievers = []
        for c_cfg in retrievers_cfg:
            r_type = c_cfg['type']
            r_conf = c_cfg['config']
            
            if r_type not in RETRIEVER_REGISTRY:
                raise ValueError(f"Unknown retriever type: {r_type}")
            
            RetrieverClass = RETRIEVER_REGISTRY[r_type]
            retriever = RetrieverClass.from_config(r_conf)
            retrievers.append(retriever)

        return cls(
                retrievers=retrievers,
                llmchater=llmchater,
                fusion_method=config.get("fusion_method", "rrf"),
                rrf_k=config.get("rrf_k", 60),
                weights=config.get("weights", None),
                top_k = config.get("top_k", 5)
                )
    
    async def retrieve(self, query: str) -> List[Dict[str, Any]]:
        top_k = self.top_k
        response = await self._expand_queries(query)
        all_queries = [query] + response

        res = None
        for query in all_queries:
            tasks = [r.retrieve(query) for r in self.retrievers]
            result_lists = await asyncio.gather(*tasks)
            if res is None:
                res = []
                for i in range(len(result_lists)):
                    res.append([result_lists[i]])
            else:
                for i in range(len(res)):
                    res[i].append(result_lists[i])
        
        for i in range(len(res)):
            fused = self._fuse_results(res[i])
            fused.sort(key=lambda x: x["score"], reverse=True)
            res[i] = fused[:top_k] if top_k else fused

        fused = self._fuse_results(res)
        
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k] if top_k else fused
    
    async def retrieve_batch(self, queries: List[str]):
        top_k = self.top_k
        for query in queries: 
            tasks = [self._expand_queries(query) for query in queries]
            expanded_queries_list = await asyncio.gather(*tasks)
            
            for i in range(len(expanded_queries_list)):
                    expanded_queries_list[i].extend([queries[i]])

        batch_final = []
        Q = len(expanded_queries_list)
        for qi in range(Q):
            for query in expanded_queries_list[qi]:
                tasks = [r.retrieve(query) for r in self.retrievers]
                qi_results = None
                result_lists = await asyncio.gather(*tasks)
                if qi_results is None:
                    qi_results = []
                    for i in range(len(result_lists)):
                        qi_results.append([result_lists[i]])
                else:
                    for i in range(len(qi_results)):
                        qi_results[i].append(result_lists[i])
            
            # TODO: fuse 在一次
            for i in range(len(qi_results)):
                fused = self._fuse_results(qi_results[i])
                fused.sort(key=lambda x: x["score"], reverse=True)
                qi_results[i] = fused[:top_k] if top_k else fused
            
            fused = self._fuse_results(qi_results)
            fused.sort(key=lambda x: x["score"], reverse=True)
            batch_final.append(fused[:top_k] if top_k else fused)
        
        return batch_final

    async def _expand_queries(self, query: str) -> List[str]:
        prompt = QUERY_EXPAND_PROMPT.format(user_query=query)
        response, _ = await self.llmchater.chat(prompt)
        response = self._normalize_json_response(response)
        response = json.loads(response)['expanded_queries']
        return response

    def _normalize_json_response(self, response: str) -> str:
        if not response:
            return "{}"
        
        response_clean = response.strip()
        
        json_block_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'`json\s*(.*?)\s*`',
            r'`\s*(.*?)\s*`'
        ]
        
        for pattern in json_block_patterns:
            match = re.search(pattern, response_clean, re.DOTALL)
            if match:
                response_clean = match.group(1).strip()
                break
        
        start_idx = response_clean.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(response_clean[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
            
            if brace_count == 0:
                response_clean = response_clean[start_idx:end_idx + 1]
        
        response_clean = re.sub(r"'([^']*)':", r'"\1":', response_clean)
        response_clean = re.sub(r":\s*'([^']*)'", r': "\1"', response_clean)
        response_clean = re.sub(r',\s*}', '}', response_clean)
        response_clean = re.sub(r',\s*]', ']', response_clean)
        
        return response_clean
    
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
    
RETRIEVER_REGISTRY[QueryEnhanceRetriever.retriever_type] = QueryEnhanceRetriever