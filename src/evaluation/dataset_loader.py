import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class EvaluationSample:
    query: str
    ground_truth_ids: List[int]
    metadata: Optional[Dict[str, Any]] = None
    
    
class EvaluationDataset:
    """
    用來載入與管理 evaluation dataset 的類別。

    預期檔案格式 (JSON array) 範例：
    [
      {
        "query": "I lost my credit card, what should I do?",
        "ground_truth_ids": [2],
        "metadata": {"category": "credit_card"}
      },
      {
        "query": "How do I apply for a credit card?",
        "ground_truth_ids": [1]
      }
    ]
    """
    def __init__(self, samples: List[EvaluationSample]):
        self._samples = samples
        
    @classmethod
    def from_json(cls, path: str) -> "EvaluationDataset":
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        samples: List[EvaluationSample] = []
        for item in raw_data:
            query = item["query"]
            ground_truth_ids = item["ground_truth_ids"]
            metadata = item.get("metadata")
            samples.append(
                EvaluationSample(
                    query=query,
                    ground_truth_ids=ground_truth_ids,
                    metadata=metadata,
                )
            )

        return cls(samples)
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def __getitem__(self, idx: int) -> EvaluationSample:
        return self._samples[idx]
    
    def __iter__(self) -> Iterable[EvaluationSample]:
        return iter(self._samples)
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        result = []
        for s in self._samples:
            item: Dict[str, Any] = {
                "query": s.query,
                "ground_truth_ids": s.ground_truth_ids,
            }
            if s.metadata is not None:
                item["metadata"] = s.metadata
            result.append(item)
        return result