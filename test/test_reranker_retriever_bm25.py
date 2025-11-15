import asyncio

from src.retrievers.bm25_retriever import BM25Retriever
from src.core.bm25_index import BM25Index
from src.retrievers.rerank_retriever import RerankRetriever
from src.rerankers.general_reranker import GeneralReranker


async def test():
    index = BM25Index("./config/bm25.yaml", auto_load=True)
    base_retriever = BM25Retriever(index=index, top_k=5)

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
