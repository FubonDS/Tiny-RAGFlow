from typing import List
from .base_query_enhancer import BaseQueryEnhancer
import asyncio


class LLMRewriteEnhancer(BaseQueryEnhancer):

    def __init__(self):
        super().__init__()
    
    async def enhance(self, query: str) -> List[str]:
        await asyncio.sleep(0.05)  

        rewrites = [
            query,  
            f"{query} 的介紹",
            f"{query} 是什麼？",
            f"關於 {query} 的資訊"
        ]

        return rewrites
