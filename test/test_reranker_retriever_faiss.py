import asyncio

from src.core.faiss_index import FaissIndex
from src.core.client.embedding_rerank_client import EmbeddingModel
from src.retrievers.faiss_retriever import FaissRetriever
from src.retrievers.rerank_retriever import RerankRetriever
from src.rerankers.general_reranker import GeneralReranker


async def test():
    index = FaissIndex("./config/faiss.yaml", auto_load=True)
    embedder = EmbeddingModel("m3e-base", "./config/models.yaml")
    base_retriever = FaissRetriever(index=index, embedder=embedder, top_k=10)

    reranker = GeneralReranker(
        model_name="bge-reranker-base",
        config_path="./config/models.yaml"
    )

    retriever = RerankRetriever(base_retriever, reranker, top_k=5)

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
