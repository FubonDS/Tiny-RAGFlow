from src.core.client.embedding_rerank_client import EmbeddingModel
import asyncio
import numpy as np

embmodel = EmbeddingModel(embedding_model="m3e-base", config_path='./config/models.yaml')
input = "The food was delicious and the waiter was friendly."
emb = asyncio.run(embmodel.embed_query(input))
query_embedding = np.array(emb)
print("Query embedding shape:", query_embedding.shape)

documents = ["The food was delicious and the waiter was friendly.",
                 "The service was slow and the food was not very good."]
doc_embeddings = asyncio.run(embmodel.embed_documents(documents))
for i, doc_emb in enumerate(doc_embeddings):
    doc_embedding =  np.array(doc_emb)
    print(f"Document {i} embedding shape:", doc_embedding.shape)