from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    embedding_base_url: str
    embedding_api_key: str
    vector_store: str
    qdrant_url: str
    qdrant_path: str

    @property
    def database_path(self) -> Path:
        return self.data_dir / "knowledge.db"

    @property
    def generation_enabled(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)


def load_settings() -> Settings:
    load_dotenv()
    data_dir = Path(os.getenv("AEC_DATA_DIR", "./data")).resolve()
    return Settings(
        data_dir=data_dir,
        llm_base_url=os.getenv("AEC_LLM_BASE_URL", "").rstrip("/"),
        llm_model=os.getenv("AEC_LLM_MODEL", ""),
        llm_api_key=os.getenv("AEC_LLM_API_KEY", ""),
        embedding_provider=os.getenv("AEC_EMBEDDING_PROVIDER", "hashing"),
        embedding_model=os.getenv("AEC_EMBEDDING_MODEL", ""),
        embedding_dimension=int(os.getenv("AEC_EMBEDDING_DIMENSION", "384")),
        embedding_base_url=os.getenv("AEC_EMBEDDING_BASE_URL", "").rstrip("/"),
        embedding_api_key=os.getenv("AEC_EMBEDDING_API_KEY", ""),
        vector_store=os.getenv("AEC_VECTOR_STORE", "sqlite"),
        qdrant_url=os.getenv("AEC_QDRANT_URL", ""),
        qdrant_path=os.getenv("AEC_QDRANT_PATH", ""),
    )
