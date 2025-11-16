import asyncio
from src.rerankers.general_reranker import GeneralReranker

async def test():
    reranker = GeneralReranker.from_config({
        "model_name": "bge-reranker-base",
        "config_path": "./config/models.yaml"
    })
    

asyncio.run(test())
