import asyncio
import os
import sys

from src.core.faiss_index import FaissIndex
from src.core.client.embedding_rerank_client import EmbeddingModel
from src.retrievers.faiss_retriever import FaissRetriever
from src.core.bm25_index import BM25Index
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.rerank_retriever import RerankRetriever
from src.rerankers.general_reranker import GeneralReranker


async def test_hybrid_retriever():
    faiss_index = FaissIndex("./config/faiss.yaml", auto_load=True)

    embedder = EmbeddingModel(
        embedding_model="m3e-base",
        config_path="./config/models.yaml"
    )

    faiss_retriever = FaissRetriever(
        index=faiss_index,
        embedder=embedder,
        top_k=3
    )
    
    bm25_index = BM25Index("./config/bm25.yaml", auto_load=True)
    bm25_retriever = BM25Retriever(bm25_index, top_k=3)
    
    hybrid = HybridRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        fusion_method="rrf"
    )
    
    reranker = GeneralReranker(
        model_name="bge-reranker-base",
        config_path="./config/models.yaml"
        )
    
    retriever = RerankRetriever(hybrid, reranker, top_k=3)
    
    # hybrid = HybridRetriever(
    #     retrievers=[faiss_retriever, bm25_retriever],
    #     fusion_method="weighted",
    #     weights=[0.7, 0.3]
    # )
    
    results = await retriever.retrieve("hello world", top_k=3)
    print("\n==== Hybrid Retrieve Results ====")
    for item in results:
        score = item["score"]
        meta = item["metadata"]
        print(f"{score:.4f} → id={meta.get('id')}  text={meta.get('text')}")
        
    # 測試 batch query
    queries = ["hello world", "faiss is great"]
    print(f"\nBatch Queries = {queries}")
    batch_results = await retriever.retrieve_batch(queries, top_k=3)
    print("\n==== Hybrid Batch Retrieve Results ====")
    for i, query_results in enumerate(batch_results):
        print(f"\n-- Results for Query: {queries[i]} --")
        for item in query_results:
            score = item["score"]
            meta = item["metadata"]
            print(f"{score:.4f} → id={meta.get('id')}  text={meta.get('text')}")
        
if __name__ == "__main__":
    asyncio.run(test_hybrid_retriever())