import os
from src.core.bm25_index import BM25Index

def test_bm25_index():

    index = BM25Index(config_path="./config/bm25.yaml", auto_load=False)


    texts = [
        "人工智慧正在改變世界。",
        "台灣的金融科技正在快速發展。",
        "機器學習是人工智慧的一個重要分支。",
        "今天台北下雨，天氣有點冷。",
    ]

    metadata = [
        {"id": 1, "text": texts[0]},
        {"id": 2, "text": texts[1]},
        {"id": 3, "text": texts[2]},
        {"id": 4, "text": texts[3]},
    ]

    index.add_batch(texts, metadata)
    index._build_bm25() 

    query = "台灣"
    scores, docs = index.search(query, top_k=3)
    print("\n=== Query ===")
    print(query)

    print("\n=== Search Result ===")
    for s, d in zip(scores, docs):
        print(f"score={s}  →  id={d['id']}  text={d['text']}")
    for s, d in zip(scores, docs):
        print(f"raw score={s}")

    index.save()

    print("\n=== Reload Index ===")
    loaded = BM25Index(config_path="./config/bm25.yaml", auto_load=True)

    scores2, docs2 = loaded.search(query, top_k=3)
    print("再查一次:")
    for s, d in zip(scores2, docs2):
        print(f"score={s:.4f}  →  id={d['id']}  text={d['text']}")
        
    
    # batch search test
    queries = ["天氣", "台灣"]
    scores, docs = loaded.search_batch(queries, top_k=2)
    print("\n=== Batch Search Result ===")
    for i, query in enumerate(queries):
        print(f"\n--- Query: {query} ---")
        for s, d in zip(scores[i], docs[i]):
            print(f"score={s:.4f}  →  id={d['id']}  text={d['text']}")
    
    

if __name__ == "__main__":
    test_bm25_index()
