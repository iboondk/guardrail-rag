"""Environment-driven configuration (env prefix GR_)."""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GR_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    qdrant_url: str | None = None  # None/'' => local mode
    qdrant_path: str = ":memory:"
    collection: str = "guardrail_rag"

    embed_backend: Literal["hash", "ollama"] = "hash"
    embed_model: str = "all-minilm:l6-v2"
    embed_dim: int = 384

    llm_backend: Literal["fake", "ollama", "anthropic", "groq"] = "fake"
    judge_backend: Literal["", "fake", "ollama", "anthropic", "groq"] = ""

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3:1b"

    anthropic_model: str = ""
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    groq_model: str = ""
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    candidate_k: int = 20

    rerank_backend: Literal["none", "lexical", "cross_encoder"] = "lexical"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)
