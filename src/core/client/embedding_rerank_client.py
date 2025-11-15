import logging

import yaml
from openai import AsyncOpenAI

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
    def __init__(self, reranking_model, config_path='./llm_tools/configs/models.yaml'):
        super().__init__('reranking_models', reranking_model, config_path)
    
    async def rerank_query(self, input=None, query=None):
        response = await self.client.embeddings.create(
            model=self.model,
            input=input,
            extra_body={"query": query}
        )
        return response.data[0].embedding
    
    async def rerank_documents(self, documents=None, query=None):
        response = await self.client.embeddings.create(
            model=self.model,
            input=documents,
            extra_body={"query": query}
        )
        scores = [response.data[i].embedding for i in range(len(response.data))]
        return scores