from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ShortcutStatus = Literal["candidate", "active", "retired"]
TYPE_ID_PATTERN = r"^[a-z][a-z0-9_-]*(?::[a-z][a-z0-9_-]*)?$"


class LayerTypeCreate(BaseModel):
    id: str = Field(pattern=TYPE_ID_PATTERN)
    label: str = Field(min_length=1, max_length=100)
    description: str = ""
    namespace: str = Field(default="workspace", pattern=r"^[a-z][a-z0-9_-]*$")


class RelationTypeCreate(BaseModel):
    id: str = Field(pattern=TYPE_ID_PATTERN)
    label: str = Field(min_length=1, max_length=100)
    description: str = ""
    namespace: str = Field(default="workspace", pattern=r"^[a-z][a-z0-9_-]*$")
    inverse_type: str | None = Field(default=None, pattern=TYPE_ID_PATTERN)
    directional: bool = True
    symmetric: bool = False
    transitive: bool = False
    temporal: bool = False
    default_traversal_weight: float = Field(default=0.65, ge=0, le=1)
    allowed_source_types: list[str] = Field(default_factory=list)
    allowed_target_types: list[str] = Field(default_factory=list)
    validators: list[str] = Field(default_factory=list)


class OntologyBundle(BaseModel):
    name: str
    description: str = ""
    layer_types: list[LayerTypeCreate] = Field(default_factory=list)
    relation_types: list[RelationTypeCreate] = Field(default_factory=list)


class LayerCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    origin_type: str = Field(pattern=TYPE_ID_PATTERN)


class NodeCreate(BaseModel):
    layer_id: str
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1)
    node_type: str = "semantic_unit"
    provenance: dict[str, Any] = Field(default_factory=dict)
    maturity: str = "tentative"


class MappingCreate(BaseModel):
    source_node_id: str
    target_node_id: str
    relation_type: str = Field(pattern=TYPE_ID_PATTERN)
    confidence: float = Field(ge=0, le=1)
    evidence: str = ""
    status: str = "suggested"
    attributes: dict[str, Any] = Field(default_factory=dict)
    valid_from: str | None = None
    valid_to: str | None = None


class ShortcutPlan(BaseModel):
    start_layer_ids: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)
    max_depth: int = Field(default=2, ge=0, le=10)
    breadth: int = Field(default=5, ge=1, le=50)
    seed_nodes: int = Field(default=8, ge=1, le=100)
    stop_min_information_gain: float = Field(default=0.03, ge=0, le=1)


class ShortcutCreate(BaseModel):
    name: str
    description: str
    trigger_examples: list[str] = Field(min_length=1)
    plan: ShortcutPlan
    preconditions: list[str] = Field(default_factory=list)
    validators: list[str] = Field(default_factory=lambda: ["citations_exist"])
    failure_conditions: list[str] = Field(default_factory=list)
    status: ShortcutStatus = "candidate"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    max_depth: int = Field(default=2, ge=0, le=10)
    breadth: int = Field(default=5, ge=1, le=50)
    seed_nodes: int = Field(default=8, ge=1, le=100)
    layer_ids: list[str] | None = None
    relation_types: list[str] | None = None
    shortcut_threshold: float = Field(default=0.72, ge=0, le=1)
    learn_shortcut: bool = True


class Citation(BaseModel):
    node_id: str
    claim: str


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
