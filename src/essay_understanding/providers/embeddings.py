from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import httpx
import numpy as np

from ..config import Settings


class Embedder(ABC):
    name: str
    dimension: int

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return normalized float32 vectors with shape (len(texts), dimension)."""


class HashingEmbedder(Embedder):
    """Dependency-free demo embedder; deterministic, local, and not production quality."""

    def __init__(self, dimension: int = 384):
        self.name = f"hashing-{dimension}"
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        output = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            normalized = re.sub(r"\s+", " ", text.lower()).strip()
            features = re.findall(r"\w+", normalized, flags=re.UNICODE)
            features.extend(normalized[index:index + 3] for index in range(max(0, len(normalized) - 2)))
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                output[row, value % self.dimension] += 1.0 if value & 1 else -1.0
            norm = np.linalg.norm(output[row])
            if norm:
                output[row] /= norm
        return output


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str):
        if not model_name:
            raise ValueError("AEC_EMBEDDING_MODEL is required for sentence_transformers")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("Install the sentence-transformers extra") from error
        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


class OpenAIEmbeddingProvider(Embedder):
    def __init__(self, base_url: str, model: str, api_key: str, dimension: int):
        if not base_url or not model:
            raise ValueError("Embedding base URL and model are required")
        self.base_url = base_url.rstrip("/")
        self.name = model
        self.api_key = api_key
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = httpx.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json={"model": self.name, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        vectors = np.asarray([item["embedding"] for item in response.json()["data"]], dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("Embedding endpoint returned an invalid shape")
        self.dimension = int(vectors.shape[1])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider.lower()
    if provider == "hashing":
        return HashingEmbedder(settings.embedding_dimension)
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    if provider == "openai":
        return OpenAIEmbeddingProvider(
            settings.embedding_base_url or settings.llm_base_url,
            settings.embedding_model,
            settings.embedding_api_key or settings.llm_api_key,
            settings.embedding_dimension,
        )
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")

