from src.core.client.embedding_rerank_client import RerankingModel
import asyncio
import numpy as np

reranker = RerankingModel(reranking_model="bge-reranker-base", config_path='./config/models.yaml')
documents = ["The food was delicious and the waiter was friendly.",
                 "The service was slow and the food was not very good."]
query = "The food was delicious and the waiter was friendly."
document_reranking = asyncio.run(reranker.rerank_documents(documents, query))
document_reranking = np.array(document_reranking)
print("==== Reranking Test ====")
print("Query:", query)
print("Documents:", documents)
print("Reranking Scores:", document_reranking)