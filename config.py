#from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore",case_sensitive=False,)

    # App metadata
    app_name: str = "HR Policy Assistant"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # Required secrets
    openai_api_key: SecretStr
    google_api_key: SecretStr

    # See what is the use of ingestion_api_key
    # ingestion_api_key: SecretStr

    # Model config
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-large"
    gemini_chat_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    embedding_dimension: int = Field(default=1536, gt=0)

    qdrant_url: str = "http://localhost:6333"
    # Optional secret
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "hr_policies"

    sparse_embedding_model: str = "Qdrant/bm25"
    cache_dir : str | None = None
    retrieval_limit: int = Field(default=12, ge=1, le=50)

    # For Future use.
    # rerank_limit: int = Field(default=5, ge=1, le=20)
    # relevance_threshold: float = Field(default=0.15, ge=-1.0, le=1.0)
    # rerank_concurrency: int = Field(default=4, ge=1, le=12)

    chunk_size: int = Field(default=700, ge=200, le=3000)
    chunk_overlap: int = Field(default=0, ge=0, le=600)
    data_dir: Path = Path("data")

    llm_temperature : float = Field(default= 0.5, ge= 0.0, le = 1.0)
    max_llm_retries : int = Field(default=3, ge=0, le=5)

    # First-time ingestion -> set to true in .env. 
    # Later run's set to false in .env
    re_ingest_docs: bool = False

    @property
    def openai_api_key_value(self) -> str:
        return self.openai_api_key.get_secret_value()

    @property
    def google_api_key_value(self) -> str:
        return self.google_api_key.get_secret_value()

    # @property
    # def ingestion_api_key_value(self) -> str:
    #    return self.ingestion_api_key.get_secret_value()
    
    @property
    def qdrant_api_key_value(self) -> str | None:
        return self.qdrant_api_key.get_secret_value() if self.qdrant_api_key else None



# @lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
