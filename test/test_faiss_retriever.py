import asyncio
import os
import sys

from src.core.faiss_index import FaissIndex
from src.core.client.embedding_rerank_client import EmbeddingModel
from src.retrievers.faiss_retriever import FaissRetriever


async def test_retriever():
    print("==== Testing FaissRetriever ====")

    index = FaissIndex("./config/faiss.yaml", auto_load=True)

    embedder = EmbeddingModel(
        embedding_model="m3e-base",
        config_path="./config/models.yaml"
    )

    retriever = FaissRetriever(
        index=index,
        embedder=embedder,
        top_k=3
    )

    # 測試 query
    query = "hello world"
    print(f"\nQuery = {query}")

    # 執行 retrieve
    results = await retriever.retrieve(query)

    print("\n==== Retrieve Results ====")
    for item in results:
        score = item["score"]
        meta = item["metadata"]
        print(f"{score:.4f} → id={meta.get('id')}  text={meta.get('text')}")
        
    # 測試 batch query
    queries = ["hello world", "faiss is great"]
    print(f"\nBatch Queries = {queries}")
    batch_results = await retriever.retrieve_batch(queries)
    print("\n==== Batch Retrieve Results ====")
    for i, query_results in enumerate(batch_results):
        print(f"\n-- Results for Query: {queries[i]} --")
        for item in query_results:
            score = item["score"]
            meta = item["metadata"]
            print(f"{score:.4f} → id={meta.get('id')}  text={meta.get('text')}")


if __name__ == "__main__":
    asyncio.run(test_retriever())
