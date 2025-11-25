import os
import pickle
from typing import Any, Dict, List

import yaml
from rank_bm25 import BM25Okapi

from .base_index import BaseIndex
from .tokenizer import Tokenizer


class BM25Index(BaseIndex):
    def __init__(self, config_path: str, auto_load=False):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        super().__init__(config)
        
        self.index_path = config['index_path']
        self.tokenizer_config = config.get("tokenizer", {})
        self.k1 = config['params'].get('k1', 1.5)
        self.b = config['params'].get('b', 0.75)
        self.epsilon = config['params'].get('epsilon', 0.25)
        
        self.tokenizer = Tokenizer(
            backend=self.tokenizer_config.get("type", "jieba"),
            mode=self.tokenizer_config.get("mode", "search"),
            stopwords_path=self.tokenizer_config.get("stopwords_path", None),
            user_dict_path=self.tokenizer_config.get("user_dict_path", None),
            normalize_config=self.tokenizer_config.get("normalize", {}),
            filter_config=self.tokenizer_config.get("filter", {})
        )
        
        self.bm25 = None
        self.corpus_tokens = []
        
        if auto_load:
            self.load()
        else:
            self.logger.info(f"BM25 index initialized (empty).")
            
    def add(self, text: str, metadata: Dict[str, Any]):
        # BM25 needs to rebuild the index after adding new documents
        tokens = self.tokenizer.tokenize(text)
        self.corpus_tokens.append(tokens)
        self.metadata.append(metadata)
        self.logger.warning("BM25 index needs to be rebuilt after adding new documents.")    
    
    def add_batch(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        if len(texts) != len(metadatas):
            raise ValueError("Length of texts and metadatas must be the same.")
        
        # batch tokenization
        tokens_list = self.tokenizer.tokenize_batch(texts)
        self.corpus_tokens.extend(tokens_list)
        self.metadata.extend(metadatas)
        self.logger.warning("BM25 index needs to be rebuilt after adding new documents.")
        
    def search(self, query: str, top_k: int):
        if self.bm25 is None:
            self.logger.info("BM25 index not built yet. Building now...")
            self._build_bm25()
            
        q_tokens = self.tokenizer.tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        
        # rank
        ranked_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        ranked_docs = [self.metadata[i].copy() for i in ranked_ids]
        ranked_scores = [scores[i] for i in ranked_ids]
        return ranked_scores, ranked_docs
    
    def search_batch(self, queries: List[str], top_k: int):
        if self.bm25 is None:
            self.logger.info("BM25 index not built yet. Building now...")
            self._build_bm25()

        query_tokens_list = self.tokenizer.tokenize_batch(queries)

        all_scores = []
        all_docs = []

        for q_tokens in query_tokens_list:
            scores = self.bm25.get_scores(q_tokens)

            ranked_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

            ranked_docs = [self.metadata[i] for i in ranked_ids]
            ranked_scores = [scores[i] for i in ranked_ids]

            all_scores.append(ranked_scores)
            all_docs.append(ranked_docs)

        return all_scores, all_docs
        
            
    def _build_bm25(self):
        if len(self.corpus_tokens) == 0:
            self.logger.warning("Corpus is empty. BM25 index not built.")
            return
        
        self.bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b, epsilon=self.epsilon)
        self.logger.info(f"BM25 index built. Corpus size = {len(self.corpus_tokens)}")
        
    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        state = {
            "corpus_tokens": self.corpus_tokens,
            "metadata": self.metadata,
        }

        with open(self.index_path, "wb") as f:
            pickle.dump(state, f)

        self.logger.info(f"BM25 index saved to {self.index_path}")
            
    def load(self):
        with open(self.index_path, "rb") as f:
            state = pickle.load(f)
            
        self.corpus_tokens = state['corpus_tokens']
        self.metadata = state['metadata']
        
        self.logger.info(f"BM25 index loaded. Corpus size={len(self.corpus_tokens)}")
        
        self._build_bm25()
        
        