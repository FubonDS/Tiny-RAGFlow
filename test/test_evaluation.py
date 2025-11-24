import asyncio

from src.retrievers.faiss_retriever import FaissRetriever
from src.evaluation.evaluator import RetrieverEvaluator
from src.evaluation.dataset_loader import EvaluationDataset
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.rerank_retriever import RerankRetriever
from src.retrievers.qdrant_retriever import QdrantRetriever
from src.retrievers.qdrant_retriever import QdrantRetriever, QdrantMultivectorRetriever




async def test_retriever():
    print("==== Testing FaissRetriever ====")
    
    faiss_retriever = FaissRetriever.from_config({
        "index_config": "./config/faiss.yaml",
        "embedding_model": "m3e-base",
        "model_config_path": "./config/models.yaml",
        "top_k": 3
    })
    
    dataset = EvaluationDataset.from_json("./data/evaluation_dataset.json")
    evaluator = RetrieverEvaluator(faiss_retriever, dataset)
    results = await evaluator.evaluate(batch_size=3)
    print("\n=== Batch faiss Evaluation Results ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
        
    bm25_retriever = BM25Retriever.from_config({
        "index_config": "./config/bm25.yaml",
        "top_k": 3
    })
    evaluator = RetrieverEvaluator(bm25_retriever, dataset)
    results = await evaluator.evaluate(batch_size=3)
    print("\n=== Batch bm25 Evaluation Results ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
        
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
        "top_k": 3
    }
    
    hybrid_retriever = HybridRetriever.from_config(hybrid_config)
    evaluator = RetrieverEvaluator(hybrid_retriever, dataset)
    results = await evaluator.evaluate(batch_size=3)
    print("\n=== Batch hybrid Evaluation Results ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
        
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
        "top_k": 3
            }
        },
        "reranker": {
            "type": "general_reranker",
            "config": {
                "model_name": "bge-reranker-base",
                "config_path": "./config/models.yaml"
            }
        },
        "top_k": 3
    }
    
    reranker_retriever = RerankRetriever.from_config(config)
    evaluator = RetrieverEvaluator(reranker_retriever, dataset)
    results = await evaluator.evaluate(batch_size=3)
    print("\n=== Batch reranker hybrid Evaluation Results ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
        
    # qdrant_retriever = QdrantRetriever.from_config({
    #     "index_config": "./config/qdrant.yaml",
    #     "embedding_model": "m3e-base",
    #     "model_config_path": "./config/models.yaml",
    #     "top_k": 3
    # })
    # evaluator = RetrieverEvaluator(qdrant_retriever, dataset)
    # results = await evaluator.evaluate(batch_size=3)
    # print("\n=== Batch qdrant Evaluation Results ===")
    # for metric, value in results.items():   
    #     print(f"{metric}: {value:.4f}")
        
    retriever = QdrantMultivectorRetriever.from_config({
        "index_config": "./config/qdrant.yaml",
        "embedding_model_path": "colbert-ir/colbertv2.0",
        "top_k": 3
    })
    evaluator = RetrieverEvaluator(retriever, dataset)
    results = await evaluator.evaluate(batch_size=3)
    print("\n=== Batch qdrant multivector Evaluation Results ===")
    for metric, value in results.items():   
        print(f"{metric}: {value:.4f}")
    
    
    
    
if __name__ == "__main__":
    asyncio.run(test_retriever())