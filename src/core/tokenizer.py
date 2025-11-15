import os
import re
import logging
import jieba

class JiebaTokenizer:
    def __init__(
        self,
        mode=None,
        stopwords_path=None,
        user_dict_path=None,
        normalize_config=None,
        filter_config=None
    ):
        self.logger = self._setup_logger()
        self.mode = mode
        self.stopwords = self._load_stopwords(stopwords_path)
        self.normalize_cfg = normalize_config or {}
        self.filter_cfg = filter_config or {}
        
        if user_dict_path and os.path.exists(user_dict_path):
            self.logger.info(f"Loading user dictionary from {user_dict_path}")
            jieba.load_userdict(user_dict_path)
            
        self.re_punct = re.compile(r"[^\w\s\u4e00-\u9fff]+")
        self.re_emoji = re.compile(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+")

        
    def _load_stopwords(self, path):
        if not path or not os.path.exists(path):
            return set()
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('JiebaTokenizer')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _fullwidth_to_halfwidth(self, text):
        """全形 → 半形"""
        res = []
        for char in text:
            code = ord(char)
            if code == 0x3000:       
                code = 0x20
            elif 0xFF01 <= code <= 0xFF5E:  
                code -= 0xFEE0
            res.append(chr(code))
        return "".join(res)
    
    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)

        cfg = self.normalize_cfg

        if cfg.get("fullwidth_to_halfwidth", False):
            text = self._fullwidth_to_halfwidth(text)
            
        if cfg.get("remove_emoji", False):
            text = self.re_emoji.sub("", text)

        if cfg.get("strip_whitespace", True):
            text = text.strip()

        if cfg.get("lower", False):
            text = text.lower()

        if cfg.get("remove_punctuation", False):
            text = self.re_punct.sub(" ", text)

        text = re.sub(r"\s+", " ", text)

        return text
    
    def _remove_stopwords(self, tokens):
        if not self.stopwords:
            return tokens
        return [t for t in tokens if t not in self.stopwords]
    
    def _filter_tokens(self, tokens):
        drop_empty = self.filter_cfg.get("drop_empty_token", True)
        min_len = self.filter_cfg.get("min_token_length", 1)

        filtered = []
        for t in tokens:
            if drop_empty and not t.strip():
                continue
            if len(t) < min_len:
                continue
            filtered.append(t)
        return filtered
    
    def tokenize(self, text: str):
        if text is None:
            return []
        # Normalize text
        text = self.normalize(text)
        # Tokenize using jieba
        if self.mode == "search":
            tokens = list(jieba.cut_for_search(text))
        else:
            tokens = list(jieba.cut(text))
        # Remove stopwords
        tokens = self._remove_stopwords(tokens)
        # Filter tokens
        tokens = self._filter_tokens(tokens)
        return tokens
    
    def tokenize_batch(self, texts):
        return [self.tokenize(text) for text in texts]
    
    def __call__(self, text: str):
        return self.tokenize(text)