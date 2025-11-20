import asyncio

from src.retrievers.qdrant_retriever import QdrantRetriever


async def test_retriever():
    print("==== Testing QdrantRetriever ====")
    retriever = QdrantRetriever.from_config({
        "index_config": "./config/qdrant.yaml",
        "embedding_model": "m3e-base",
        "model_config_path": "./config/models.yaml",
        "top_k": 3
    })

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
