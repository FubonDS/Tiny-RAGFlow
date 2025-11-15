import asyncio
from src.rerankers.general_reranker import GeneralReranker

async def test():
    reranker = GeneralReranker(
        model_name="bge-reranker-base",
        config_path="./config/models.yaml"
    )

    documents = [
        "The food was delicious and the waiter was friendly.",
        "The service was slow and the food was not very good."
    ]

    query = "The food was delicious and the waiter was friendly."

    scores = await reranker.rerank(query, documents)
    print(scores)
    docs = [
        {"id": 1, "text": "The food was delicious and the waiter was friendly."},
        {"id": 2, "text": "The service was slow and the food was not very good."}
    ]

    results = await reranker.rerank_with_text(query, docs)

    for r in results:
        print(r)

asyncio.run(test())
