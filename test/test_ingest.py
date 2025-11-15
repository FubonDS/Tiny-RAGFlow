from src.core.faiss_index import FaissIndex
from src.core.client.embedding_rerank_client import EmbeddingModel
from src.pipelines.faiss_ingestion import FaissIngestionPipeline
import asyncio
import numpy as np

index = FaissIndex("./config/faiss.yaml")
embedder = EmbeddingModel(embedding_model="m3e-base", config_path='./config/models.yaml')
pipeline = FaissIngestionPipeline(embedder, index)

dataset = [
    {"id": 1, "text": "hello world"},
    {"id": 2, "text": "how are you"},
    {"id": 3, "text": "faiss is great"},
]

asyncio.run(pipeline.ingest_dataset(dataset, batch_size=3))

query = ["hello world"]   
query_vec = asyncio.run(embedder.embed_documents(query))   
query_vec = np.array(query_vec) 

scores, docs = index.search(query_vec, top_k=3)

print("\n==== Retrieval Test ====")
print("Query:", query)
print("Scores:", scores)
print("Docs:", docs)