import numpy as np
from src.core.faiss_index import FaissIndex

index = FaissIndex(config_path='./config/faiss.yaml')
index = FaissIndex("./config/faiss.yaml")

# 加入資料
index.add(np.array([[1,0,0,0]]), {"id": 1})
index.add(np.array([[0,1,0,0]]), {"id": 2})
index.add(np.array([[0,0,1,0]]), {"id": 3})

# 批量查詢
queries = np.array([
    [1,0,0,0],
    [0,1,0,0],
])

scores, docs = index.search_batch(queries, top_k=2)

print(scores)
print(docs)
# vectors = np.random.rand(5000, 1024).astype(np.float32)
# metadatas = [{"id": i, "text": f"doc {i}"} for i in range(5000)]

# index.add_batch(vectors, metadatas)


# vec1 = np.array([[1,0,0,0]], dtype=np.float32)
# vec2 = np.array([[0,1,0,0]], dtype=np.float32)

# index.add(vec1, {"id": 1})
# index.add(vec2, {"id": 2})

# scores, docs = index.search(vec1, 2)
# print(scores, docs)


# vec1 = np.array([[1, 0, 0, 0]], dtype=np.float32)
# vec2 = np.array([[0.8, 0.1, 0, 0]], dtype=np.float32)

# index.add(vec1, {"id": 1, "text": "hello world"})
# index.add(vec2, {"id": 2, "text": "vietnam economy"})

# query = np.array([[1, 0, 0, 0]], dtype=np.float32)
# scores, docs = index.search(query, top_k=2)
# print(scores)
# print(docs)
# index.save()

# new_index = FaissIndex(config_path='./config/faiss.yaml')
# new_index.load()

# scores, docs = new_index.search(query, top_k=2)
# print(scores, docs)