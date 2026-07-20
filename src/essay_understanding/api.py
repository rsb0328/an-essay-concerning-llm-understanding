from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import load_settings
from .models import (
    LayerCreate, LayerTypeCreate, MappingCreate, NodeCreate, OntologyBundle,
    QueryRequest, RelationTypeCreate, SchemaApprovalRequest, ShortcutCreate,
)
from .runtime import engine
from .processing import KnowledgeProcessor


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    engine().vectors.close()


app = FastAPI(
    title="An Essay Concerning LLM Understanding",
    description="Embedding-provider-pluggable multi-layer vector storage with extensible domain ontologies and a peer shortcut layer.",
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


class SchemaDiscoveryRequest(BaseModel):
    source_layer_ids: list[str] = Field(min_length=1)
    namespace: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    sample_limit: int = Field(default=24, ge=4, le=32)
    max_chars_per_node: int = Field(default=4000, ge=200, le=4000)


class ReadinessRequest(BaseModel):
    layer_ids: list[str] = Field(min_length=1)


class PlacementRequest(BaseModel):
    source_layer_ids: list[str] = Field(min_length=1)
    material_origin: str
    sample_limit: int = Field(default=24, ge=1, le=32)


class SchemaCleaningRequest(BaseModel):
    discovery_id: str
    source_node_ids: list[str] | None = None
    max_nodes: int = Field(default=24, ge=1, le=32)
    placement_plan_id: str | None = None


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
        "abstraction_thresholds": {
            "min_nodes": settings.schema_min_nodes,
            "min_chars": settings.schema_min_chars,
            "short_record_nodes": settings.schema_short_record_nodes,
            "required_surveys": settings.schema_required_surveys,
        },
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


@app.get("/ontology/schema-discoveries")
def list_schema_discoveries(status: str | None = None):
    return engine().repository.schema_discoveries(status)


@app.get("/ontology/schema-discoveries/{discovery_id}")
def get_schema_discovery(discovery_id: str):
    found = engine().repository.schema_discovery(discovery_id)
    if not found:
        raise HTTPException(status_code=404, detail="Schema discovery not found")
    return found


@app.post("/ontology/schema-discoveries/{discovery_id}/approve")
def approve_schema_discovery(discovery_id: str, item: SchemaApprovalRequest):
    try:
        return engine().repository.approve_schema_discovery(discovery_id, item.candidate_keys)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/ontology/schema-discoveries/{discovery_id}/reject")
def reject_schema_discovery(discovery_id: str):
    try:
        return engine().repository.reject_schema_discovery(discovery_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


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
    try:
        return KnowledgeProcessor(engine()).abstract_layer(
            item.source_layer_id, item.target_name, item.target_description, item.target_layer_type)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/process/derive")
def derive_layer(item: AbstractionRequest):
    """Domain-neutral alias for the backward-compatible abstraction endpoint."""
    try:
        return KnowledgeProcessor(engine()).abstract_layer(
            item.source_layer_id, item.target_name, item.target_description, item.target_layer_type)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/process/map-layers")
def map_layers(item: LayerMappingRequest):
    return KnowledgeProcessor(engine()).map_layers(
        item.reference_layer_ids, item.target_layer_ids, item.per_target, item.accept)


@app.post("/process/discover-schema")
def discover_schema(item: SchemaDiscoveryRequest):
    try:
        return KnowledgeProcessor(engine()).discover_schema(
            item.source_layer_ids, item.namespace, item.sample_limit, item.max_chars_per_node)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/process/abstraction-readiness")
def abstraction_readiness(item: ReadinessRequest):
    try:
        return KnowledgeProcessor(engine()).abstraction_readiness(item.layer_ids)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/process/plan-placement")
def plan_placement(item: PlacementRequest):
    try:
        return KnowledgeProcessor(engine()).plan_material_placement(
            item.source_layer_ids, item.material_origin, item.sample_limit)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/placement-plans")
def list_placement_plans(status: str | None = None):
    return engine().repository.placement_plans(status)


@app.post("/placement-plans/{plan_id}/approve")
def approve_placement_plan(plan_id: str):
    try:
        return engine().repository.decide_placement_plan(plan_id, "approved")
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/placement-plans/{plan_id}/reject")
def reject_placement_plan(plan_id: str):
    try:
        return engine().repository.decide_placement_plan(plan_id, "rejected")
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/process/clean-with-schema")
def clean_with_schema(item: SchemaCleaningRequest):
    try:
        return KnowledgeProcessor(engine()).clean_with_schema(
            item.discovery_id, item.source_node_ids, item.max_nodes, item.placement_plan_id)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


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


@app.post("/indexes/rebuild")
def rebuild_indexes():
    return engine().rebuild_indexes()


@app.get("/export")
def export_memory():
    return engine().repository.export_all()
