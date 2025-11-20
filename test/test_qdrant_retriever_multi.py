import asyncio

from src.retrievers.qdrant_retriever import QdrantRetriever, QdrantMultivectorRetriever


async def test_retriever():
    print("==== Testing QdrantRetriever ====")
    retriever = QdrantMultivectorRetriever.from_config({
        "index_config": "./config/qdrant.yaml",
        "embedding_model_path": "colbert-ir/colbertv2.0",
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
    queries = ["Spinal Tap, one of England's loudest bands, is chronicled by film director Marty DiBergi on what proves to be a fateful tour.", "faiss is great"]
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
