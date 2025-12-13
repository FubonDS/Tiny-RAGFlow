import asyncio
from src.evaluation.retriever_benchmark import RetrieverBenchmark
from src.retrievers.faiss_retriever import FaissRetriever
from src.evaluation.evaluator import RetrieverEvaluator
from src.evaluation.dataset_loader import EvaluationDataset
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.rerank_retriever import RerankRetriever
from src.retrievers.qdrant_retriever import QdrantRetriever
from src.retrievers.qdrant_retriever import QdrantRetriever, QdrantMultivectorRetriever


async def main():
    dataset = EvaluationDataset.from_json("./data/bonnieQA/evaluation_dataset.json")
    
    # faiss
    faiss_m3e_base_retriever = FaissRetriever.from_config({
        "index_config": "./config/faiss.yaml",
                "embedding_model": "m3e-base",
                "model_config_path": "./config/models.yaml",
                "top_k": 50,
                "dedup_key": "metadata.answer",
                "cache_config": {
                    "enable": True,
                    "cache_file": './cache/retriever_cache.json'
                }
    })
    
    config = {
        "retriever": {
            "type": "faiss",
            "config": {
                "index_config": "./config/faiss.yaml",
                "embedding_model": "m3e-base",
                "model_config_path": "./config/models.yaml",
                "top_k": 50,
                "dedup_key": "metadata.answer",
                "cache_config": {
                    "enable": True,
                    "cache_file": './cache/retriever_cache.json'
                }
            }
        },
        "reranker": {
            "type": "general_reranker",
            "config": {
                "model_name": "bge-reranker-base",
                "config_path": "./config/models.yaml",
                "conbine_metadata_keys": ["metadata.question"]
            }
        },
        "top_k": 3
    }
    
    faiss_m3e_base_retriever_rerank = RerankRetriever.from_config(config)
    
    
    config = {
        "retrievers": [
                {
                    "type": "faiss",
                    "config": {
                        "index_config": "./config/faiss.yaml",
                        "embedding_model": "m3e-base",
                        "model_config_path": "./config/models.yaml",
                        "top_k": 50,
                        "dedup_key": "metadata.answer"
                    }
                }
            ],
            "fusion_method": "rrf",
            "rrf_k": 60,
            "top_k": 50,

            "query_extension_config": {
                "method": "sub_question",  # or "paraphrase" or "sub_question",
                "query_expand_number": 3,
            },
            
            "llm_model": "gpt-5-nano",
            "model_config_path": "./config/models.yaml",
            "cache_config": {
                "enable": True,
                "cache_file": "./llm_cache.json"
            }
        }
    rerank_config = {
        "retriever": {
            "type": "queryenhance",
            "config": config
        },
        "reranker": {
            "type": "general_reranker",
            "config": {
                "model_name": "bge-reranker-base",
                "config_path": "./config/models.yaml"
            }
        },
        "top_k": 2
    }
    # pehybrid_retriever_rerank = RerankRetriever.from_config(rerank_config)
    
    retrievers = [
        # ("PEHybrid_Rerank", pehybrid_retriever_rerank, None),
        ("FAISS_m3e_base_Rerank", faiss_m3e_base_retriever_rerank, None),
        # ("FAISS_m3e_base", faiss_m3e_base_retriever, None),
    ]
    
    
    benchmark = RetrieverBenchmark(
        retrievers,
        dataset,
        type="single"
    )
    
    results = await benchmark.run(
        top_k=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20],
        batch_size=16,
        sort_by=("MRR", 5)
    )
        
if __name__ == "__main__":
    asyncio.run(main())