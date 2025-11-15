import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseIndex(ABC):
    def __init__(self, config: Dict):
        """
        config: 來自 YAML 配置的 dict（例如 bm25.yaml 或 faiss.yaml 解析後的內容）
        """
        self.config = config
        self.logger = self._setup_logger()
        self.metadata: List[Dict[str, Any]] = []
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @abstractmethod
    def add(self, item, metadata):
        pass

    @abstractmethod
    def add_batch(self, items, metadatas):
        pass

    @abstractmethod
    def search(self, query, top_k):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass
    
    def search_batch(self, queries, top_k):
        return [self.search(q, top_k) for q in queries]
