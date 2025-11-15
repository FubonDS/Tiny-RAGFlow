import asyncio

from src.core.faiss_index import FaissIndex
from src.core.client.embedding_rerank_client import EmbeddingModel
from src.retrievers.faiss_retriever import FaissRetriever
from src.retrievers.rerank_retriever import RerankRetriever
from src.rerankers.general_reranker import GeneralReranker
from src.query_enhancers.llm_rewrite_enhancer import LLMRewriteEnhancer
from src.pipelines.query_retrieval_pipeline import QueryRetrievalPipeline


async def test():
    enhancers = [LLMRewriteEnhancer()]
    index = FaissIndex("./config/faiss.yaml", auto_load=True)
    embedder = EmbeddingModel("m3e-base", "./config/models.yaml")
    base_retriever = FaissRetriever(index=index, embedder=embedder, top_k=10)

    reranker = GeneralReranker(
        model_name="bge-reranker-base",
        config_path="./config/models.yaml"
    )

    retriever = RerankRetriever(base_retriever, reranker, top_k=5)
    
    pipeline = QueryRetrievalPipeline(
        enhancers=enhancers,
        retriever=retriever,
    )


    results = await pipeline.retrieve("hello world", top_k=3)
    print("\n==== Query Enhance + Retrieve + Rerank Results ====")
    for item in results:
        score = item["score"]
        meta = item["metadata"]
        print(f"{score:.4f} → id={meta.get('id')}  text={meta.get('text')}")
        
    # 測試 batch query
    queries = ["hello world", "faiss is great"]
    print(f"\nBatch Queries = {queries}")
    batch_results = await pipeline.retrieve_batch(queries, top_k=3)
    for i, query_results in enumerate(batch_results):
        print(f"\n-- Results for Query: {queries[i]} --")
        for item in query_results:
            score = item["score"]
            meta = item["metadata"]
            print(f"{score:.4f} → id={meta.get('id')}  text={meta.get('text')}")

asyncio.run(test())