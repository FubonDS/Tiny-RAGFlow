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
                        "top_k": 50
                    }
                },
                {
                    "type": "bm25",
                    "config": {
                        "index_config": "./config/bm25.yaml",
                        "top_k": 50
                    }
                }
            ],
            "fusion_method": "rrf",
            "rrf_k": 60,
            "top_k": 50,
            "query_extension_config": {
                "method": "sub_question",  # or "paraphrase" or "sub_question",
                "query_expand_number": 3,
                "fusion_method": "rrf", # or "round_robin"
                "alpha": 0.3
            },

            "llm_model": "gpt-5-nano",
            "model_config_path": "./config/models.yaml",
            "cache_config": {
                    "enable": True,
                    "cache_file": './cache/query_enhance/retriever_cache.json'
                }
        }

async def test_query_enhance_retriever():
    print("==== Testing QueryEnhanceRetriever ====")
    retriever_enhance = QueryEnhanceRetriever.from_config(hybrid_config)

    # 測試 retrieve 方法
    query = "如何解除預約解定存, 以及他行繳信用卡費？"
    results =  await retriever_enhance.retrieve(
        query
        )
    for item in results:
        score = item["score"]
        meta = item["metadata"]
        print(f"{score:.4f} → id={meta.get('id')}  text={meta.get('text')} metadata={meta.get('metadata')}")

    # 測試 retrieve_batch 方法
    queries = [
        "如何解除預約解定存, 以及他行繳信用卡費？",
        "如何解除預約解定存, 以及他行繳信用卡費？",
        "如何解除預約解定存, 以及他行繳信用卡費？",
        ]

    batch_results = await retriever_enhance.retrieve_batch(queries)
    print(batch_results)
    
if __name__ == "__main__":
    asyncio.run(test_query_enhance_retriever())