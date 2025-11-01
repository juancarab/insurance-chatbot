"""Application configuration management for environment variables."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings

    
class Settings(BaseSettings):
    """Centralised configuration for the Insurance Chatbot backend."""
    formatter: Literal["mock", "langchain", "gemini"] = Field(
        "mock", validation_alias="INSURANCE_CHATBOT_FORMATTER"
    )
    langchain_runner: Optional[str] = Field(
        default="services.agent.app.langchain_runner:run_langchain_agent",
        validation_alias="INSURANCE_CHATBOT_LANGCHAIN_RUNNER",
    )
    gemini_api_key: Optional[str] = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_model: Optional[str] = Field(default=None, validation_alias="GEMINI_MODEL")
    gemini_temperature: Optional[float] = Field(
        default=None, validation_alias="GEMINI_TEMPERATURE"
    )
    gemini_top_p: Optional[float] = Field(default=None, validation_alias="GEMINI_TOP_P")
    gemini_max_output_tokens: Optional[int] = Field(
        default=None, validation_alias="GEMINI_MAX_OUTPUT_TOKENS"
    )
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    @validator("formatter", pre=True)
    def _normalise_formatter(cls, value: str | None) -> str:
        """Ensure formatter values match the expected lowercase literals."""

        return (value or "mock").lower()

@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()
