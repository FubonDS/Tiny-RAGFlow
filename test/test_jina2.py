from src.core.client.embedding_rerank_client import JinaForRerankingModel
import asyncio



async def main():
    reranker = JinaForRerankingModel(model_path="./models/jina-reranker-v3")
    documents = ["The food was delicious and the waiter was friendly.",
                     "The service was slow and the food was not very good."]
    query = "The food was delicious and the waiter was friendly."
    document_reranking = await reranker.rerank_documents(documents, query)
    print("==== Jina Reranking Test ====")
    print("Query:", query)
    print("Documents:", documents)
    print("Reranking Scores:", document_reranking)
    
    # Test batch reranking
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
    batch_reranking = await reranker.rerank_documents_batch(documents_list=document_list, query_list=query_list, batch_size=1)
    print("==== Jina Batch Reranking Test ====")
    for i, scores in enumerate(batch_reranking):
        print(f"Query {i+1}: {query_list[i]}")
        print("Documents:", document_list[i])
        print("Reranking Scores:", scores)
        print()

if __name__ == "__main__":
    asyncio.run(main())