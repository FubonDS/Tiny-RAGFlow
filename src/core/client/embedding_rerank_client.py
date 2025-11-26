import logging

import yaml
from openai import AsyncOpenAI
import httpx


class BaseModel:
    def __init__(self, model_type, model_name, config_path='./configs/models.yaml'):
        self.config = self.load_config(config_path)
        self.model_config = self.config[model_type][model_name]
        self.model = self.model_config['model']
        self.logger = self._setup_logger()
        self.local_api_key = self.model_config['local_api_key']
        self.local_base_url = self.model_config['local_base_url']
        self.logger.info(f'[{model_type}] Initializing Model: {model_name}')
        self.client = AsyncOpenAI(api_key=self.local_api_key, base_url=self.local_base_url)
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('EmbeddingRerankClient')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_config(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
        
class EmbeddingModel(BaseModel):
    def __init__(self, embedding_model, config_path='./configs/models.yaml'):
        super().__init__('embedding_models', embedding_model, config_path)
    
    async def embed_query(self, query=None):
        response = await self.client.embeddings.create(
            input=query,
            model=self.model
        )
        return response.data[0].embedding
    
    async def embed_documents(self, documents=None):
        response = await self.client.embeddings.create(
            input=documents,
            model=self.model
        )
        embs =  [response.data[i].embedding for i in range(len(response.data))]
        return embs

class RerankingModel(BaseModel):
    def __init__(self, model_type , reranking_model, config_path='./llm_tools/configs/models.yaml'):
        if reranking_model in ['bge-reranker-large']:
            super().__init__(model_type, reranking_model, config_path)
            self.client_type = 'openai'
        elif reranking_model in ['qwen3_rerank_06B']:
            self.config = self.load_config(config_path)
            self.model_config = self.config[model_type][reranking_model]
            self.model = self.model_config['model']
            self.logger = self._setup_logger()
            self.local_api_key = self.model_config['local_api_key']
            self.local_base_url = self.model_config['local_base_url']
            self.logger.info(f'[{model_type}] Initializing Model: {reranking_model}')
            self.client_type = 'vllm'
            self.headers = {
            "Content-Type": "application/json"
            }
    
    async def rerank_query(self, input=None, query=None):
        if self.client_type == 'openai':
            response = await self.client.embeddings.create(
                model=self.model,
                input=input,
                extra_body={"query": query}
            )
            return response.data[0].embedding
        elif self.client_type == 'vllm':
            payload = {
                "model": self.model,
                "encoding_format": "float",
                "query": query,
                "documents": [input]
                }
            async with httpx.AsyncClient() as client:
                    response = await client.post(self.local_base_url, headers=self.headers, json=payload)
                    json_response = response.json()
                    results = json_response.get('results', [])[0]
                    return [results.get('relevance_score')]
            
    async def rerank_documents(self, documents=None, query=None):
        if self.client_type == 'openai':
            response = await self.client.embeddings.create(
                model=self.model,
                input=documents,
                extra_body={"query": query}
            )
            scores = [response.data[i].embedding for i in range(len(response.data))]
            return scores
        elif self.client_type == 'vllm':
            payload = {
                "model": self.model,
                "encoding_format": "float",
                "query": query,
                "documents": [doc for doc in documents]
                }
            async with httpx.AsyncClient() as client:
                response = await client.post(self.local_base_url, headers=self.headers, json=payload)
                json_response = response.json()
                results = json_response.get('results', [])
                sorted_results = sorted(results, key=lambda item: item.get('index'))
                scores = [result.get('relevance_score') for result in sorted_results]
                return scores

class MultiVectorModel():
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.logger = self._setup_logger()
        self.logger.info(f'Initializing MultiVector Model from: {model_path}')
        from fastembed import LateInteractionTextEmbedding
        self.embedding_model = LateInteractionTextEmbedding(model_path)
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def embed_query(self, query=None):
        embed = self.embedding_model.query_embed(query)
        return list(embed)[0]
    
    async def embed_query_batch(self, queries=None):
        embeds = self.embedding_model.query_embed(queries)
        embeds = list(embeds)
        return embeds
    
    async def embed_documents(self, documents=None):
        embeds = list(self.embedding_model.embed(documents))
        return embeds
