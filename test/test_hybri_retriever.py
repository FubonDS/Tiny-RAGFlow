import asyncio
import os
import sys

from src.core.faiss_index import FaissIndex
from src.core.client.embedding_rerank_client import EmbeddingModel
from src.retrievers.faiss_retriever import FaissRetriever
from src.core.bm25_index import BM25Index
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever


async def test_hybrid_retriever():
    hybrid_config = {
        "retrievers": [
            {
                "type": "faiss",
                "config": {
                    "index_config": "./config/faiss.yaml",
                    "embedding_model": "m3e-base",
                    "model_config_path": "./config/models.yaml",
                    "top_k": 5,
                    "dedup_key": "metadata.A"
                }
            },
            {
                "type": "bm25",
                "config": {
                    "index_config": "./config/bm25.yaml",
                    "top_k": 5,
                    "dedup_key": "metadata.A"
                }
            }
        ],
        "fusion_method": "rrf",
        "rrf_k": 60,
        "top_k": 5
    }
    
    hybrid = HybridRetriever.from_config(hybrid_config)
    
    results = await hybrid.retrieve("How to apply for a new credit card from the bank?")
    print("\n==== Hybrid Retrieve Results ====")
    for item in results:
        score = item["score"]
        meta = item["metadata"]
        print(f"{score:.4f} → id={meta.get('id')}  text={meta.get('text')} metadata={meta.get('metadata')}")
        
    # 測試 batch query
    queries = ["How to apply for a new credit card from the bank?", "faiss is great"]
    print(f"\nBatch Queries = {queries}")
    batch_results = await hybrid.retrieve_batch(queries)
    print("\n==== Hybrid Batch Retrieve Results ====")
    for i, query_results in enumerate(batch_results):
        print(f"\n-- Results for Query: {queries[i]} --")
        for item in query_results:
            score = item["score"]
            meta = item["metadata"]
            print(f"{score:.4f} → id={meta.get('id')}  text={meta.get('text')} metadata={meta.get('metadata')}")
        
if __name__ == "__main__":
    asyncio.run(test_hybrid_retriever())