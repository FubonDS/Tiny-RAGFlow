from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseRetriever(ABC):
    """
    通用 Retriever 基類
    所有 Retriever 都應該繼承這個 class
    """
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict):
        pass
    
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        傳回格式：
        [
            {
                "score": float,
                "metadata": {...}
            },
            ...
        ]
        """
        pass
    
    @abstractmethod
    async def retrieve_batch(self, queries: List[str], top_k: int = None) -> List[List[Dict[str, Any]]]:
        """
        傳回格式：
        [
            [   # for query1
                {"score":..., "metadata":...},
                ...
            ],
            [   # for query2
                {"score":..., "metadata":...},
                ...
            ]
        ]
        """
        pass
    
    