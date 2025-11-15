from src.pipelines.bm25_ingestion import BM25IngestionPipeline
from src.core.bm25_index import BM25Index

bm25 = BM25Index("./config/bm25.yaml", auto_load=False)

pipeline = BM25IngestionPipeline(bm25)

dataset = [
    {"id": 0, "text": "人工智慧正在改變世界。"},
    {"id": 1, "text": "台灣的金融科技正在快速發展。"},
    {"id": 2, "text": "機器學習是人工智慧的重要分支。"},
]

pipeline.ingest_dataset(dataset)
