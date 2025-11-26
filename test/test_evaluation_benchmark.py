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
    dataset = EvaluationDataset.from_json("./data/evaluation_dataset.json")
    
    # faiss
    faiss_retriever = FaissRetriever.from_config({
        "index_config": "./config/faiss.yaml",
        "embedding_model": "m3e-base",
        "model_config_path": "./config/models.yaml",
        "top_k": 3,
        "dedup_key": "metadata.A"
    })
    # bm25
    bm25_retriever = BM25Retriever.from_config({
        "index_config": "./config/bm25.yaml",
        "top_k": 3,
        "dedup_key": "metadata.A"
    })
    
    # htbrid
    hybrid_config = {
        "retrievers": [
            {
                "type": "faiss",
                "config": {
                    "index_config": "./config/faiss.yaml",
                    "embedding_model": "m3e-base",
                    "model_config_path": "./config/models.yaml",
                    "top_k": 3,
                    "dedup_key": "metadata.A"
                }
            },
            {
                "type": "bm25",
                "config": {
                    "index_config": "./config/bm25.yaml",
                    "top_k": 3,
                    "dedup_key": "metadata.A"
                }
            }
        ],
        "fusion_method": "rrf",
        "rrf_k": 60,
        "top_k": 3
    }
    
    hybrid_retriever = HybridRetriever.from_config(hybrid_config)
    
    config = {
        "retriever": {
            "type": "hybrid",
            "config": {
                    "retrievers": [
                {
                    "type": "faiss",
                    "config": {
                        "index_config": "./config/faiss.yaml",
                        "embedding_model": "m3e-base",
                        "model_config_path": "./config/models.yaml",
                        "top_k": 3,
                        "dedup_key": "metadata.A"
                    }
                },
                {
                    "type": "bm25",
                    "config": {
                        "index_config": "./config/bm25.yaml",
                        "top_k": 3,
                        "dedup_key": "metadata.A"
                }
            }
        ],
        "fusion_method": "rrf",
        "rrf_k": 60,
        "top_k": 3
            }
        },
        "reranker": {
            "type": "general_reranker",
            "config": {
                "model_name": "bge-reranker-large",
                "config_path": "./config/models.yaml"
            }
        },
        "top_k": 3
    }
    
    reranker_retriever = RerankRetriever.from_config(config)
    
    # qdrant_retriever = QdrantMultivectorRetriever.from_config({
    #     "index_config": "./config/qdrant.yaml",
    #     "embedding_model_path": "colbert-ir/colbertv2.0",
    #     "top_k": 3
    # })
    qdrant_retriever = QdrantRetriever.from_config({
        "index_config": "./config/qdrant.yaml",
        "embedding_model": "m3e-base",
        "model_config_path": "./config/models.yaml",
        "top_k": 3,
        "dedup_key": "metadata.A"
    })
    
    
    retrievers = [
        ("FAISS", faiss_retriever),
        ("BM25", bm25_retriever),
        ("Hybrid", hybrid_retriever),
        ("Reranker-Hybrid", reranker_retriever),
        ("Qdrant-Single", qdrant_retriever)
    ]
    
    benchmark = RetrieverBenchmark(
        retrievers,
        dataset,
    )
    
    results = await benchmark.run(
        top_k=[3, 5],
        batch_size=3,
        sort_by=("MRR", 5)
    )
        
if __name__ == "__main__":
    asyncio.run(main())