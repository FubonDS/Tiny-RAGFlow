from transformers import AutoTokenizer, AutoModel
import torch
from src.core.client.jina.jina_for_ranking import JinaForRanking

model_name = "./models/jina-reranker-v3"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = JinaForRanking.from_pretrained(
    model_name,
    dtype="auto").to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

query = "What are the health benefits of green tea?"
documents = [
    "What are the health benefits of green tea?",
    "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
    "El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro.",
    "Studies show that drinking green tea regularly can improve brain function and boost metabolism.",
    "Basketball is one of the most popular sports in the United States.",
    "绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。",
    "Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale.",
]

documents2 = [
    "What are the health benefits of green tea?",
    "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
    "El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro."
]

query_list = [query, query]
document_list = [documents, documents2]

# Rerank documents
# results = model.rerank(query, documents)
results = model.rerank_batch(query_list, document_list, batch_size=1)



# Results are sorted by relevance score (highest first)
for result in results:
    print("Reranking results:")
    for r in result:
        print(f"Score: {r['relevance_score']:.4f} - Document: {r['document']}")

# save
# folder_name = "reranker_models"
# model.save_pretrained(f"./{folder_name}/jinaai/jina-reranker-v3")
# tokenizer.save_pretrained(f"./{folder_name}/jinaai/jina-reranker-v3")

