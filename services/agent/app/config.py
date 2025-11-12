from __future__ import annotations
from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class AgentSettings(BaseSettings):
    """
    Configuración del agente (independiente del backend).
    Lee variables de entorno (.env o docker-compose).
    """

    # --- Gemini (LangChain wrapper ChatGoogleGenerativeAI) ---
    gemini_api_key: Optional[str] = Field(default=None, validation_alias="GEMINI_API_KEY") [cite: 276, 544]
    gemini_model: Optional[str] = Field(default=None, validation_alias="GEMINI_MODEL") [cite: 276, 544]
    gemini_temperature: Optional[float] = Field(default=None, validation_alias="GEMINI_TEMPERATURE") [cite: 276, 544]
    gemini_top_p: Optional[float] = Field(default=None, validation_alias="GEMINI_TOP_P") [cite: 276, 544]
    gemini_max_output_tokens: Optional[int] = Field(default=None, validation_alias="GEMINI_MAX_OUTPUT_TOKENS") [cite: 276, 544]

    # --- Tavily ---
    tavily_api_key: Optional[str] = Field(default=None, validation_alias="TAVILY_API_KEY") [cite: 276, 544]

    # --- OpenSearch (índice principal) ---
    opensearch_host: str = Field(default="http://opensearch", validation_alias="OPENSEARCH_HOST") [cite: 276, 544]
    opensearch_port: int = Field(default=9200, validation_alias="OPENSEARCH_PORT") [cite: 276, 544]
    opensearch_index: str = Field(default="policies", validation_alias="OPENSEARCH_INDEX") [cite: 276, 544]
    opensearch_embed_dim: int = Field(default=384, validation_alias="OPENSEARCH_EMBED_DIM") [cite: 277, 544]

    # --- Modelo de embeddings ---
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    ) [cite: 277, 545]

    # --- Retrieval + Reranking (Combinación de ambas ramas) ---
    retrieval_top_k: int = Field(default=40, validation_alias="RETRIEVAL_TOP_K") # Valor más alto para pre-filtrado [cite: 277]
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", validation_alias="RERANK_MODEL") [cite: 277, 545]
    rerank_batch_size: int = Field(default=32, validation_alias="RERANK_BATCH_SIZE") [cite: 277, 545]
    rerank_top_k: int = Field(default=10, validation_alias="RERANK_TOP_K") # El valor de main, más prudente que 4 [cite: 277]

    # --- OpenSearch (índice de resúmenes para el router semántico) ---
    policy_summaries_index: str = Field(
        default="policy_summaries_index",
        validation_alias="POLICY_SUMMARIES_INDEX",
    ) [cite: 546]

    # --- Seguridad/SSL opcional para OpenSearch ---
    opensearch_user: Optional[str] = Field(default=None, validation_alias="OPENSEARCH_USER") [cite: 277, 546]
    opensearch_password: Optional[str] = Field(default=None, validation_alias="OPENSEARCH_PASSWORD") [cite: 277, 546]
    opensearch_use_ssl: bool = Field(default=False, validation_alias="OPENSEARCH_USE_SSL") [cite: 277, 546]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> AgentSettings:
    """Carga la configuración desde .env (cacheada)[cite: 278, 546]."""
    return AgentSettings()