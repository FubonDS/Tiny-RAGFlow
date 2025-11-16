import asyncio

from src.retrievers.rerank_retriever import RerankRetriever


async def test():
    config = {
        "retriever": {
            "type": "faiss",
            "config": {
                "index_config": "./config/faiss.yaml",
                "embedding_model": "m3e-base",
                "model_config_path": "./config/models.yaml",
                "top_k": 3
            }
        },
        "reranker": {
            "model_name": "bge-reranker-base",
            "config_path": "./config/models.yaml"
        },
        "top_k": 3
    }
    retriever = RerankRetriever.from_config(config)

    query = "hello world"
    results = await retriever.retrieve(query)

    for r in results:
        print(r)
        
    queries = ["hello world", "faiss index"]
    batch_results = await retriever.retrieve_batch(queries)
    for i, query_results in enumerate(batch_results):
        print(f"\n-- Results for Query: {queries[i]} --")
        for r in query_results:
            print(r)

asyncio.run(test())
