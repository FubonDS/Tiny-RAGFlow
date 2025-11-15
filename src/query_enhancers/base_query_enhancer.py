import asyncio
from abc import ABC, abstractmethod
from typing import List


class BaseQueryEnhancer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    async def enhance(self, query: str) -> List[str]:
        """
        子類別必須實作：
        - 接收一個 query
        - 回傳一組 query list（可能包含自己本身）

        範例：
        [
            "什麼是人工智慧？",
            "人工智慧簡介",
            "AI 是什麼？"
        ]
        """
        pass
    
    async def enhance_batch(self, queries: List[str]) -> List[List[str]]:
        tasks = [self.enhance(q) for q in queries]
        return await asyncio.gather(*tasks)
