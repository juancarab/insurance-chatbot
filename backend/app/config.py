"""Application configuration management for environment variables."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Centralised configuration for the Insurance Chatbot backend."""

    formatter: Literal["mock", "langchain", "gemini"] = Field(
        "mock", env="INSURANCE_CHATBOT_FORMATTER"
    )
    langchain_runner: Optional[str] = Field(
        default=None, env="INSURANCE_CHATBOT_LANGCHAIN_RUNNER"
    )
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    gemini_model: Optional[str] = Field(default=None, env="GEMINI_MODEL")
    gemini_temperature: Optional[float] = Field(
        default=None, env="GEMINI_TEMPERATURE"
    )
    gemini_top_p: Optional[float] = Field(default=None, env="GEMINI_TOP_P")
    gemini_max_output_tokens: Optional[int] = Field(
        default=None, env="GEMINI_MAX_OUTPUT_TOKENS"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("formatter", pre=True)
    def _normalise_formatter(cls, value: str | None) -> str:
        """Ensure formatter values match the expected lowercase literals."""

        return (value or "mock").lower()


@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of application settings."""

    return Settings()
