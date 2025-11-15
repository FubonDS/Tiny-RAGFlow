import asyncio
import os
import sys

from src.core.bm25_index import BM25Index
from src.retrievers.bm25_retriever import BM25Retriever


async def test_bm25_retriever():
    index = BM25Index("./config/bm25.yaml", auto_load=True)
    retriever = BM25Retriever(index, top_k=3)

    query = "hello world"
    print(f"Query = {query}")

    results = await retriever.retrieve(query)

    print("\n===== BM25 Retrieve Results =====")
    for r in results:
        print(f"{r['score']:.4f} → id={r['metadata'].get('id')}  text={r['metadata'].get('text')}")
        
    # batch query test
    queries = ["hello world", "bm25 index"]
    print(f"\nBatch Queries = {queries}")
    batch_results = await retriever.retrieve_batch(queries)
    print("\n===== BM25 Batch Retrieve Results =====")
    for i, query_results in enumerate(batch_results):
        print(f"\n-- Results for Query: {queries[i]} --")
        for r in query_results:
            print(f"{r['score']:.4f} → id={r['metadata'].get('id')}  text={r['metadata'].get('text')}")


if __name__ == "__main__":
    asyncio.run(test_bm25_retriever())
