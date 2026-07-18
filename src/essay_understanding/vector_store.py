from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings
from .repository import Repository, now


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, namespace: str, item_id: str, vector: np.ndarray, payload: dict[str, Any], model: str) -> None: ...

    @abstractmethod
    def search(self, namespace: str, query: np.ndarray, limit: int, model: str,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...

    @abstractmethod
    def delete(self, namespace: str, item_id: str, model: str) -> None: ...

    def close(self) -> None:
        """Release optional backend resources."""


class SQLiteVectorStore(VectorStore):
    def __init__(self, repository: Repository):
        self.repository = repository

    def upsert(self, namespace: str, item_id: str, vector: np.ndarray, payload: dict[str, Any], model: str) -> None:
        values = np.asarray(vector, dtype=np.float32).tolist()
        with self.repository.connection() as db:
            db.execute("""INSERT OR REPLACE INTO vectors
              (item_id,namespace,model,dimension,vector_json,payload_json,updated_at)
              VALUES(?,?,?,?,?,?,?)""", (
                item_id, namespace, model, len(values), json.dumps(values),
                json.dumps(payload, ensure_ascii=False), now()))

    def search(self, namespace: str, query: np.ndarray, limit: int, model: str,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        rows = self.repository.rows(
            "SELECT * FROM vectors WHERE namespace=? AND model=?", (namespace, model))
        query = np.asarray(query, dtype=np.float32)
        hits = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            if filters and any(payload.get(key) not in values for key, values in filters.items()):
                continue
            vector = np.asarray(json.loads(row["vector_json"]), dtype=np.float32)
            if vector.shape != query.shape:
                continue
            hits.append({"item_id": row["item_id"], "score": float(vector @ query), **payload})
        return sorted(hits, key=lambda hit: hit["score"], reverse=True)[:limit]

    def delete(self, namespace: str, item_id: str, model: str) -> None:
        with self.repository.connection() as db:
            db.execute("DELETE FROM vectors WHERE namespace=? AND item_id=? AND model=?",
                       (namespace, item_id, model))


class QdrantVectorStore(VectorStore):
    def __init__(self, settings: Settings, dimension: int):
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as error:
            raise RuntimeError("Install the qdrant extra") from error
        self.models = models
        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url)
        else:
            path = settings.qdrant_path or str(settings.data_dir / "qdrant")
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=path)
        self.dimension = dimension

    def _collection(self, namespace: str, model: str) -> str:
        import hashlib
        suffix = hashlib.sha1(model.encode()).hexdigest()[:10]
        return f"aec_{namespace}_{suffix}"

    def _ensure(self, collection: str) -> None:
        if not self.client.collection_exists(collection):
            self.client.create_collection(collection, vectors_config=self.models.VectorParams(
                size=self.dimension, distance=self.models.Distance.COSINE))

    def upsert(self, namespace: str, item_id: str, vector: np.ndarray, payload: dict[str, Any], model: str) -> None:
        collection = self._collection(namespace, model)
        self._ensure(collection)
        self.client.upsert(collection, [self.models.PointStruct(
            id=item_id, vector=np.asarray(vector).tolist(), payload=payload)], wait=True)

    def search(self, namespace: str, query: np.ndarray, limit: int, model: str,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        collection = self._collection(namespace, model)
        if not self.client.collection_exists(collection):
            return []
        must = []
        for key, values in (filters or {}).items():
            must.append(self.models.FieldCondition(key=key, match=self.models.MatchAny(any=list(values))))
        found = self.client.query_points(
            collection, query=np.asarray(query).tolist(), limit=limit,
            query_filter=self.models.Filter(must=must) if must else None, with_payload=True).points
        return [{"item_id": str(point.id), "score": float(point.score), **(point.payload or {})} for point in found]

    def delete(self, namespace: str, item_id: str, model: str) -> None:
        collection = self._collection(namespace, model)
        if self.client.collection_exists(collection):
            self.client.delete(collection, self.models.PointIdsList(points=[item_id]), wait=True)

    def close(self) -> None:
        self.client.close()


def build_vector_store(settings: Settings, repository: Repository, dimension: int) -> VectorStore:
    if settings.vector_store == "sqlite":
        return SQLiteVectorStore(repository)
    if settings.vector_store == "qdrant":
        return QdrantVectorStore(settings, dimension)
    raise ValueError(f"Unknown vector store: {settings.vector_store}")
