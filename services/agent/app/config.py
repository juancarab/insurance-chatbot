from __future__ import annotations
from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class AgentSettings(BaseSettings):
    """
    Config del AGENTE (independiente del backend).
    Lee variables desde entorno (.env via docker-compose).
    """
    # Gemini (LangChain wrapper ChatGoogleGenerativeAI)
    gemini_api_key: Optional[str] = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_model: Optional[str] = Field(default=None, validation_alias="GEMINI_MODEL")
    gemini_temperature: Optional[float] = Field(default=None, validation_alias="GEMINI_TEMPERATURE")
    gemini_top_p: Optional[float] = Field(default=None, validation_alias="GEMINI_TOP_P")
    gemini_max_output_tokens: Optional[int] = Field(default=None, validation_alias="GEMINI_MAX_OUTPUT_TOKENS")

    # Tavily
    tavily_api_key: Optional[str] = Field(default=None, validation_alias="TAVILY_API_KEY")

    # OpenSearch
    opensearch_host: str = Field(default="http://opensearch", validation_alias="OPENSEARCH_HOST")
    opensearch_port: int = Field(default=9200, validation_alias="OPENSEARCH_PORT")
    opensearch_index: str = Field(default="policies", validation_alias="OPENSEARCH_INDEX")
    opensearch_embed_dim: int = Field(default=384, validation_alias="OPENSEARCH_EMBED_DIM")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L-6-v2", validation_alias="EMBEDDING_MODEL")

    retrieval_top_k: int = Field(default=40, validation_alias="RETRIEVAL_TOP_K")
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", validation_alias="RERANK_MODEL")
    rerank_batch_size: int = Field(default=32, validation_alias="RERANK_BATCH_SIZE")
    rerank_top_k: int = Field(default=10, validation_alias="RERANK_TOP_K")

    opensearch_user: Optional[str] = Field(default=None, validation_alias="OPENSEARCH_USER")
    opensearch_password: Optional[str] = Field(default=None, validation_alias="OPENSEARCH_PASSWORD")
    opensearch_use_ssl: bool = Field(default=False, validation_alias="OPENSEARCH_USE_SSL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

@lru_cache()
def get_settings() -> AgentSettings:
    return AgentSettings()