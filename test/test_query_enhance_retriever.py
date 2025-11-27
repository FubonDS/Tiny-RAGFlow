from src.retrievers.query_enhance_retriever import QueryEnhanceRetriever
from src.retrievers.faiss_retriever import FaissRetriever

import asyncio

hybrid_config = {
        "retrievers": [
                {
                    "type": "faiss",
                    "config": {
                        "index_config": "./config/faiss.yaml",
                        "embedding_model": "m3e-base",
                        "model_config_path": "./config/models.yaml",
                        "top_k": 3
                    }
                },
                {
                    "type": "bm25",
                    "config": {
                        "index_config": "./config/bm25.yaml",
                        "top_k": 3
                    }
                }
            ],
            "fusion_method": "rrf",
            "rrf_k": 60,
            "top_k": 5,

            "llm_model": "Qwen2.5-32B-Instruct-GPTQ-Int4",
            "model_config_path": "./config/models.yaml"
        }

async def test_query_enhance_retriever():
    print("==== Testing QueryEnhanceRetriever ====")
    retriever_enhance = QueryEnhanceRetriever.from_config(hybrid_config)

    # 測試 retrieve 方法
    query = "房貸提前清償本金"
    results =  await retriever_enhance.retrieve(
        query
        )
    print(results)

    # 測試 retrieve_batch 方法
    queries = [
        "信用卡的年費是怎麼計算的？",
        "房貸提前清償本金",
        ]

    batch_results = await retriever_enhance.retrieve_batch(queries)
    print(batch_results)
    
if __name__ == "__main__":
    asyncio.run(test_query_enhance_retriever())