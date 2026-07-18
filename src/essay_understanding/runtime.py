from __future__ import annotations

from functools import lru_cache

from .config import load_settings
from .engine import MemoryEngine
from .providers import build_embedder, build_generation_provider
from .repository import Repository
from .vector_store import build_vector_store


@lru_cache(maxsize=1)
def engine() -> MemoryEngine:
    settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    repository = Repository(settings.database_path)
    embedder = build_embedder(settings)
    vectors = build_vector_store(settings, repository, embedder.dimension)
    generation = build_generation_provider(settings)
    return MemoryEngine(repository, embedder, vectors, generation)

