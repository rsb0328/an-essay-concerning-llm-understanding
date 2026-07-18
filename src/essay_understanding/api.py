from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import load_settings
from .models import (
    LayerCreate, LayerTypeCreate, MappingCreate, NodeCreate, OntologyBundle,
    QueryRequest, RelationTypeCreate, ShortcutCreate,
)
from .runtime import engine
from .processing import KnowledgeProcessor


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    engine().vectors.close()


app = FastAPI(
    title="An Essay Concerning LLM Understanding",
    description="Model-agnostic multi-layer vector storage with extensible domain ontologies and a peer shortcut layer.",
    version="0.1.0",
    lifespan=lifespan,
)


class TextIngest(BaseModel):
    layer_id: str
    title: str
    text: str = Field(min_length=1)
    chunk_chars: int = Field(default=1200, ge=200, le=10000)
    overlap: int = Field(default=120, ge=0, le=2000)
    provenance: dict[str, Any] = Field(default_factory=dict)


class AbstractionRequest(BaseModel):
    source_layer_id: str
    target_name: str
    target_description: str = "Machine-derived semantic units"
    target_layer_type: str = "machine_derived"


class LayerMappingRequest(BaseModel):
    reference_layer_ids: list[str] = Field(min_length=1)
    target_layer_ids: list[str] = Field(min_length=1)
    per_target: int = Field(default=4, ge=1, le=20)
    accept: bool = False


@app.get("/status")
def status():
    settings = load_settings()
    core = engine()
    return {
        "generation": {"available": core.generation.available, "provider": core.generation.name},
        "embedding": {"provider": settings.embedding_provider, "model": core.embedder.name,
                      "dimension": core.embedder.dimension},
        "vector_store": settings.vector_store,
        "data_dir": str(settings.data_dir),
    }


@app.post("/layers")
def create_layer(item: LayerCreate):
    return {"id": engine().create_layer(item)}


@app.get("/layers")
def list_layers():
    return engine().repository.rows("SELECT * FROM layers ORDER BY created_at")


@app.get("/ontology")
def get_ontology():
    return engine().repository.ontology_snapshot()


@app.post("/ontology/layer-types")
def register_layer_type(item: LayerTypeCreate):
    return {"id": engine().repository.register_layer_type(item)}


@app.post("/ontology/relation-types")
def register_relation_type(item: RelationTypeCreate):
    return {"id": engine().repository.register_relation_type(item)}


@app.post("/ontology/import")
def import_ontology(item: OntologyBundle):
    return engine().repository.import_ontology(item)


@app.post("/nodes")
def create_node(item: NodeCreate):
    return {"id": engine().create_node(item)}


@app.post("/ingest/text")
def ingest_text(item: TextIngest):
    ids = engine().ingest_text(item.layer_id, item.title, item.text, item.chunk_chars,
                               item.overlap, item.provenance)
    return {"node_ids": ids, "count": len(ids)}


@app.post("/mappings")
def create_mapping(item: MappingCreate):
    return {"id": engine().create_mapping(item)}


@app.post("/process/abstract")
def abstract_layer(item: AbstractionRequest):
    return KnowledgeProcessor(engine()).abstract_layer(
        item.source_layer_id, item.target_name, item.target_description, item.target_layer_type)


@app.post("/process/derive")
def derive_layer(item: AbstractionRequest):
    """Domain-neutral alias for the backward-compatible abstraction endpoint."""
    return KnowledgeProcessor(engine()).abstract_layer(
        item.source_layer_id, item.target_name, item.target_description, item.target_layer_type)


@app.post("/process/map-layers")
def map_layers(item: LayerMappingRequest):
    return KnowledgeProcessor(engine()).map_layers(
        item.reference_layer_ids, item.target_layer_ids, item.per_target, item.accept)


@app.post("/shortcuts")
def create_shortcut(item: ShortcutCreate):
    return {"id": engine().create_shortcut(item)}


@app.get("/shortcuts")
def list_shortcuts():
    return [engine().repository.decode_shortcut(row) for row in
            engine().repository.rows("SELECT * FROM shortcuts ORDER BY created_at")]


@app.post("/query")
def query(item: QueryRequest):
    return engine().query(item)


@app.get("/export")
def export_memory():
    return engine().repository.export_all()
