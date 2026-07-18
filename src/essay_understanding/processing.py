from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .engine import MemoryEngine
from .config import load_settings
from .models import LayerCreate, MappingCreate, NodeCreate, SchemaDiscoveryResult
from .protocols import (
    ABSTRACTION_SYSTEM, LAYER_PLACEMENT_SYSTEM, RELATION_SYSTEM, SCHEMA_CLEANING_SYSTEM,
    SCHEMA_DISCOVERY_SYSTEM,
)


class AbstractUnit(BaseModel):
    title: str
    content: str = Field(min_length=1)
    node_type: str
    source_node_ids: list[str] = Field(min_length=1)


class InternalRelation(BaseModel):
    source_unit: int = Field(ge=0)
    target_unit: int = Field(ge=0)
    relation_type: str
    confidence: float = Field(ge=0, le=1)
    evidence: str = ""


class AbstractionResult(BaseModel):
    units: list[AbstractUnit]
    relations: list[InternalRelation] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_indexes(self):
        for relation in self.relations:
            if relation.source_unit >= len(self.units) or relation.target_unit >= len(self.units):
                raise ValueError("Abstraction relation refers to an unknown unit")
        return self


class RelationDecision(BaseModel):
    pair_id: str
    relation_type: str
    confidence: float = Field(ge=0, le=1)
    evidence: str = ""


class RelationBatch(BaseModel):
    relations: list[RelationDecision]


class RoutedUnit(BaseModel):
    target_layer_type: str
    target_layer_name: str
    target_existing_layer_id: str | None = None
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    node_type: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    source_node_ids: list[str] = Field(min_length=1)


class RoutedRelation(BaseModel):
    source_unit: int = Field(ge=0)
    target_unit: int = Field(ge=0)
    relation_type: str
    confidence: float = Field(ge=0, le=1)
    evidence: str = ""


class RoutedCleaningResult(BaseModel):
    units: list[RoutedUnit] = Field(max_length=200)
    relations: list[RoutedRelation] = Field(default_factory=list, max_length=500)

    @model_validator(mode="after")
    def valid_indexes(self):
        for relation in self.relations:
            if relation.source_unit >= len(self.units) or relation.target_unit >= len(self.units):
                raise ValueError("Cleaned relation refers to an unknown unit")
        return self


class PlacementDecision(BaseModel):
    source_node_ids: list[str] = Field(min_length=1)
    action: Literal["append_existing_layer", "create_peer_layer", "create_derived_layer",
                    "link_only", "hold_for_review"]
    target_existing_layer_id: str | None = None
    target_layer_type: str | None = None
    target_layer_name: str | None = None
    same_source_continuation: bool = False
    independence_signals: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class PlacementPlanResult(BaseModel):
    summary: str
    decisions: list[PlacementDecision] = Field(min_length=1, max_length=200)


@dataclass(frozen=True)
class AbstractionReadinessPolicy:
    """Configurable Alpha guardrail, not a universal statistical theorem."""

    min_nodes: int = 12
    min_chars: int = 24_000
    short_record_nodes: int = 50
    required_surveys: int = 2


class KnowledgeProcessor:
    """Model-assisted processing with application-side validation and persistence."""

    def __init__(self, engine: MemoryEngine, policy: AbstractionReadinessPolicy | None = None):
        self.engine = engine
        if policy is None:
            settings = load_settings()
            policy = AbstractionReadinessPolicy(
                min_nodes=settings.schema_min_nodes,
                min_chars=settings.schema_min_chars,
                short_record_nodes=settings.schema_short_record_nodes,
                required_surveys=settings.schema_required_surveys,
            )
        self.policy = policy

    def abstraction_readiness(self, layer_ids: list[str]) -> dict[str, Any]:
        if not layer_ids:
            raise ValueError("At least one source layer is required")
        unique_ids = sorted(set(layer_ids))
        marks = ",".join("?" for _ in unique_ids)
        layers = self.engine.repository.rows(
            f"SELECT id,name,origin_type,is_initial FROM layers WHERE id IN ({marks})", unique_ids)
        if len(layers) != len(unique_ids):
            raise ValueError("One or more source layers do not exist")
        totals = self.engine.repository.rows(
            f"""SELECT COUNT(*) AS node_count,
            COALESCE(SUM(LENGTH(content)),0) AS total_chars
            FROM nodes WHERE layer_id IN ({marks})""", unique_ids)[0]
        node_count = int(totals["node_count"])
        total_chars = int(totals["total_chars"])
        enough_nodes = node_count >= self.policy.min_nodes
        enough_information = (total_chars >= self.policy.min_chars
                              or node_count >= self.policy.short_record_nodes)
        failures = []
        if not enough_nodes:
            failures.append(f"need at least {self.policy.min_nodes} nodes")
        if not enough_information:
            failures.append(
                f"need at least {self.policy.min_chars} characters or "
                f"{self.policy.short_record_nodes} short records")
        return {
            "eligible": enough_nodes and enough_information,
            "layer_ids": unique_ids,
            "layers": layers,
            "node_count": node_count,
            "total_chars": total_chars,
            "thresholds": {
                "min_nodes": self.policy.min_nodes,
                "min_chars": self.policy.min_chars,
                "short_record_nodes": self.policy.short_record_nodes,
                "required_surveys": self.policy.required_surveys,
            },
            "failed_conditions": failures,
            "interpretation": (
                "Historical initial status records provenance only; it does not make a layer "
                "a permanent semantic root or retrieval priority."
            ),
        }

    def discover_schema(self, source_layer_ids: list[str], namespace: str,
                        sample_limit: int = 24, max_chars_per_node: int = 4000) -> dict[str, Any]:
        if not self.engine.generation.available:
            raise RuntimeError("A generation provider is required for schema discovery")
        if not source_layer_ids:
            raise ValueError("At least one source layer is required")
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", namespace):
            raise ValueError("Namespace must begin with a letter and contain letters, numbers, '_' or '-'")
        readiness = self.abstraction_readiness(source_layer_ids)
        if not readiness["eligible"]:
            raise ValueError("Dataset is not ready for abstraction: " + "; ".join(readiness["failed_conditions"]))
        layers = self.engine.repository.rows(
            f"SELECT id,name,description,origin_type FROM layers WHERE id IN ({','.join('?' for _ in source_layer_ids)})",
            source_layer_ids,
        )
        if len(layers) != len(set(source_layer_ids)):
            raise ValueError("One or more source layers do not exist")
        survey_round = self.engine.repository.next_schema_survey_round(source_layer_ids, namespace)
        sample = self._representative_sample(
            source_layer_ids, sample_limit, max_chars_per_node, survey_round)
        if not sample:
            raise ValueError("Selected layers contain no nodes")
        active = self.engine.repository.ontology_snapshot()
        raw = self.engine.generation.structured(
            system=SCHEMA_DISCOVERY_SYSTEM,
            task="schema_discovery",
            payload={
                "requested_namespace": namespace,
                "source_layers": layers,
                "representative_raw_sample": sample,
                "active_ontology": active,
                "abstraction_readiness": readiness,
                "survey_round": survey_round,
                "decision_rules": {
                    "layer_type": "Use only for an independently selected, governed, or lifecycle-distinct context.",
                    "node_type": "Use for a stable kind of semantic unit with identity.",
                    "attribute": "Use for a value describing one unit; do not create a layer for scalar values.",
                    "relation_type": "Use for a verifiable connection between two units.",
                },
            },
            schema=SchemaDiscoveryResult.model_json_schema(),
        )
        result = SchemaDiscoveryResult.model_validate(raw)
        prefix = f"{namespace}:"
        keys: set[str] = set()
        for candidate in result.candidates:
            if not candidate.id.startswith(prefix):
                raise ValueError(f"Schema candidate must use namespace {namespace}: {candidate.id}")
            key = f"{candidate.kind}:{candidate.id}"
            if key in keys:
                raise ValueError(f"Duplicate schema candidate: {key}")
            keys.add(key)
        comparisons = self._compare_schema_candidates(result, active)
        discovery_id = self.engine.repository.create_schema_discovery(
            source_layer_ids=source_layer_ids,
            sample_node_ids=[item["id"] for item in sample],
            namespace=namespace,
            dataset_summary=result.dataset_summary,
            candidates=[item.model_dump() for item in result.candidates],
            comparisons=comparisons,
            cleaning_guidance=result.cleaning_guidance,
            generation_provider=self.engine.generation.name,
            readiness=readiness,
            survey_round=survey_round,
        )
        return self.engine.repository.schema_discovery(discovery_id) or {}

    def _representative_sample(self, layer_ids: list[str], sample_limit: int,
                               max_chars_per_node: int, survey_round: int = 1) -> list[dict[str, Any]]:
        per_layer = max(1, sample_limit // len(layer_ids))
        sample: list[dict[str, Any]] = []
        for layer_id in layer_ids:
            count = self.engine.repository.rows(
                "SELECT COUNT(*) AS count FROM nodes WHERE layer_id=?", (layer_id,))[0]["count"]
            if not count:
                continue
            take = min(per_layer, count)
            # Deterministic rotating strata: repeated surveys cover different positions while
            # remaining reproducible for a given round.
            shift = ((survey_round - 1) * 0.38196601125) % 1.0
            offsets = sorted({round((((index + .5) / take + shift) % 1.0) * (count - 1))
                              for index in range(take)})
            for offset in offsets:
                row = self.engine.repository.rows(
                    """SELECT id,layer_id,title,content,node_type,provenance_json FROM nodes
                    WHERE layer_id=? ORDER BY created_at LIMIT 1 OFFSET ?""", (layer_id, offset))[0]
                row["content"] = row["content"][:max_chars_per_node]
                sample.append(row)
        return sample[:sample_limit]

    def _compare_schema_candidates(self, result: SchemaDiscoveryResult,
                                   active: dict[str, Any]) -> list[dict[str, Any]]:
        dimensions = active.get("semantic_dimensions", [])
        output: list[dict[str, Any]] = []
        for candidate in result.candidates:
            existing = (active["layer_types"] if candidate.kind == "layer_type" else
                        active["relation_types"] if candidate.kind == "relation_type" else
                        [item for item in dimensions if item["kind"] == candidate.kind])
            exact = next((item for item in existing if item["id"] == candidate.id), None)
            if exact:
                output.append({
                    "candidate_key": f"{candidate.kind}:{candidate.id}",
                    "classification": "existing", "nearest_existing_id": exact["id"],
                    "similarity": 1.0, "recommendation": "reuse_existing",
                })
                continue
            nearest_id, similarity = None, 0.0
            if existing:
                candidate_text = f"{candidate.label}\n{candidate.description}\n{candidate.rationale}"
                existing_texts = [f"{item['label']}\n{item.get('description', '')}" for item in existing]
                query_vector = self.engine.embedder.encode([candidate_text])[0]
                scores = self.engine.embedder.encode(existing_texts) @ query_vector
                best_index = int(np.argmax(scores))
                nearest_id, similarity = existing[best_index]["id"], float(scores[best_index])
            overlap = similarity >= 0.82
            output.append({
                "candidate_key": f"{candidate.kind}:{candidate.id}",
                "classification": "possible_overlap" if overlap else "novel",
                "nearest_existing_id": nearest_id, "similarity": similarity,
                "recommendation": "review_merge" if overlap else "add",
            })
        return output

    def plan_material_placement(self, source_layer_ids: list[str], material_origin: str,
                                sample_limit: int = 24) -> dict[str, Any]:
        if not self.engine.generation.available:
            raise RuntimeError("A generation provider is required for material placement planning")
        if material_origin not in {"primary_source", "external_source", "machine_derived"}:
            raise ValueError("material_origin must be primary_source, external_source, or machine_derived")
        # Placement can be planned before abstraction readiness; it does not activate a schema or create a layer.
        self.abstraction_readiness(source_layer_ids)
        sample = self._representative_sample(source_layer_ids, sample_limit, 4000)
        if not sample:
            raise ValueError("Selected layers contain no nodes")
        layers = self.engine.repository.rows(
            "SELECT id,name,description,origin_type,is_initial FROM layers ORDER BY created_at")
        ontology = self.engine.repository.ontology_snapshot()
        raw = self.engine.generation.structured(
            system=LAYER_PLACEMENT_SYSTEM,
            task="layer_placement",
            payload={
                "material_origin": material_origin,
                "source_nodes": sample,
                "existing_layers": layers,
                "active_layer_types": ontology["layer_types"],
                "rules": {
                    "append_initial": "Only a genuine continuation of the same source.",
                    "external_independence": "Different author/source/version/viewpoint/access/lifecycle -> peer layer.",
                    "machine_derivation": "Never append machine-derived material to the historical initial layer.",
                },
            },
            schema=PlacementPlanResult.model_json_schema(),
        )
        result = PlacementPlanResult.model_validate(raw)
        source_ids = {item["id"] for item in sample}
        covered: list[str] = []
        layers_by_id = {item["id"]: item for item in layers}
        layer_types = {item["id"] for item in ontology["layer_types"]}
        for decision in result.decisions:
            if not set(decision.source_node_ids).issubset(source_ids):
                raise ValueError("Placement decision refers to a source outside the sampled material")
            covered.extend(decision.source_node_ids)
            if decision.action == "append_existing_layer":
                target = layers_by_id.get(decision.target_existing_layer_id or "")
                if not target:
                    raise ValueError("append_existing_layer requires a valid target_existing_layer_id")
                if target["is_initial"] and not decision.same_source_continuation:
                    raise ValueError("Only a genuine same-source continuation may append to the initial layer")
                if material_origin == "machine_derived" and target["is_initial"]:
                    raise ValueError("Machine-derived material cannot be appended to the initial layer")
            if decision.action in {"create_peer_layer", "create_derived_layer"}:
                if not decision.target_layer_name or decision.target_layer_type not in layer_types:
                    raise ValueError("New-layer placement requires a name and an active target layer type")
            if material_origin == "machine_derived" and decision.action == "create_peer_layer":
                raise ValueError("Machine-derived material must use create_derived_layer")
        if set(covered) != source_ids:
            raise ValueError("Every sampled source node must appear in at least one placement decision")
        plan_id = self.engine.repository.create_placement_plan(
            source_layer_ids=source_layer_ids,
            source_node_ids=sorted(source_ids),
            material_origin=material_origin,
            plan=result.model_dump(),
            generation_provider=self.engine.generation.name,
        )
        return self.engine.repository.placement_plan(plan_id) or {}

    def clean_with_schema(self, discovery_id: str, source_node_ids: list[str] | None = None,
                          max_nodes: int = 24, placement_plan_id: str | None = None) -> dict[str, Any]:
        if not self.engine.generation.available:
            raise RuntimeError("A generation provider is required for schema-guided cleaning")
        discovery = self.engine.repository.schema_discovery(discovery_id)
        if not discovery:
            raise ValueError("Schema discovery not found")
        if discovery["status"] != "approved":
            raise ValueError("Schema discovery must be approved before cleaning")
        selected_ids = source_node_ids or discovery["sample_node_ids"]
        if not selected_ids:
            raise ValueError("No source nodes selected for cleaning")
        if len(selected_ids) > max_nodes:
            raise ValueError(f"Cleaning batch exceeds max_nodes={max_nodes}")
        marks = ",".join("?" for _ in selected_ids)
        sources = self.engine.repository.rows(
            f"""SELECT id,layer_id,title,content,node_type,attributes_json,provenance_json
            FROM nodes WHERE id IN ({marks}) ORDER BY created_at""", selected_ids)
        if len(sources) != len(set(selected_ids)):
            raise ValueError("One or more selected source nodes do not exist")
        allowed_layers = set(discovery["source_layer_ids"])
        if any(item["layer_id"] not in allowed_layers for item in sources):
            raise ValueError("Selected nodes must belong to the discovery source layers")
        placement = None
        if placement_plan_id:
            placement = self.engine.repository.placement_plan(placement_plan_id)
            if not placement or placement["status"] != "approved":
                raise ValueError("Placement plan must exist and be approved before constrained cleaning")
            if not set(selected_ids).issubset(set(placement["source_node_ids"])):
                raise ValueError("Selected nodes must be covered by the approved placement plan")
        ontology = self.engine.repository.ontology_snapshot()
        raw = self.engine.generation.structured(
            system=SCHEMA_CLEANING_SYSTEM,
            task="schema_guided_cleaning",
            payload={
                "approved_schema_discovery": {
                    "dataset_summary": discovery["dataset_summary"],
                    "cleaning_guidance": discovery["cleaning_guidance"],
                },
                "active_ontology": ontology,
                "source_nodes": sources,
                "approved_placement_plan": placement,
            },
            schema=RoutedCleaningResult.model_json_schema(),
        )
        result = RoutedCleaningResult.model_validate(raw)
        allowed_source_ids = {item["id"] for item in sources}
        layer_types = {item["id"] for item in ontology["layer_types"]}
        node_types = {item["id"] for item in ontology["semantic_dimensions"]
                      if item["kind"] == "node_type"}
        attributes = {item["id"] for item in ontology["semantic_dimensions"]
                      if item["kind"] == "attribute"}
        relations = {item["id"]: item for item in ontology["relation_types"]}
        decisions_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if placement:
            for decision in placement["plan"]["decisions"]:
                for source_id in decision["source_node_ids"]:
                    decisions_by_source[source_id].append(decision)
        routed_source_ids: set[str] = set()
        for unit in result.units:
            if unit.target_layer_type not in layer_types:
                raise ValueError(f"Cleaning proposed unknown layer type: {unit.target_layer_type}")
            if unit.node_type not in node_types:
                raise ValueError(f"Cleaning proposed unknown node type: {unit.node_type}")
            unknown_attributes = sorted(set(unit.attributes) - attributes)
            if unknown_attributes:
                raise ValueError(f"Cleaning proposed unknown attributes: {', '.join(unknown_attributes)}")
            if not set(unit.source_node_ids).issubset(allowed_source_ids):
                raise ValueError("Cleaning unit refers to a source outside this batch")
            routed_source_ids.update(unit.source_node_ids)
            if placement:
                decision_groups = [decisions_by_source.get(source_id, []) for source_id in unit.source_node_ids]
                if any(not group for group in decision_groups):
                    raise ValueError("Cleaning unit is not covered by the approved placement plan")
                signature_groups = [{(decision["action"], decision.get("target_existing_layer_id"),
                                      decision.get("target_layer_type"), decision.get("target_layer_name"))
                                     for decision in group} for group in decision_groups]
                signatures = set.intersection(*signature_groups)
                if not signatures:
                    raise ValueError("One cleaned unit cannot combine incompatible placement decisions")
                signature = ("append_existing_layer", unit.target_existing_layer_id, None, None) \
                    if unit.target_existing_layer_id else (
                        "create_derived_layer" if any(sig[0] == "create_derived_layer" for sig in signatures)
                        else "create_peer_layer", None, unit.target_layer_type, unit.target_layer_name)
                if signature not in signatures:
                    raise ValueError("Cleaning output violates approved layer placement")
                decision = next(decision for decision in decision_groups[0]
                                if (decision["action"], decision.get("target_existing_layer_id"),
                                    decision.get("target_layer_type"), decision.get("target_layer_name")) == signature)
                if decision["action"] in {"link_only", "hold_for_review"}:
                    raise ValueError(f"Placement action {decision['action']} forbids creating a cleaned unit")
                if decision["action"] == "append_existing_layer":
                    if unit.target_existing_layer_id != decision.get("target_existing_layer_id"):
                        raise ValueError("Cleaning output violates approved existing-layer placement")
                else:
                    if (unit.target_existing_layer_id is not None
                            or unit.target_layer_type != decision.get("target_layer_type")
                            or unit.target_layer_name != decision.get("target_layer_name")):
                        raise ValueError("Cleaning output violates approved new-layer placement")
        if placement:
            required_routing = {source_id for source_id in selected_ids
                                if any(decision["action"] not in {"link_only", "hold_for_review"}
                                       for decision in decisions_by_source[source_id])}
            if not required_routing.issubset(routed_source_ids):
                raise ValueError("Cleaning omitted material that the placement plan requires routing")
        for relation in result.relations:
            if relation.relation_type not in relations:
                raise ValueError(f"Cleaning proposed unknown relation type: {relation.relation_type}")
            definition = relations[relation.relation_type]
            source_type = result.units[relation.source_unit].target_layer_type
            target_type = result.units[relation.target_unit].target_layer_type
            if definition["allowed_source_types"] and source_type not in definition["allowed_source_types"]:
                raise ValueError(f"Relation {relation.relation_type} rejects cleaned source layer type {source_type}")
            if definition["allowed_target_types"] and target_type not in definition["allowed_target_types"]:
                raise ValueError(f"Relation {relation.relation_type} rejects cleaned target layer type {target_type}")
            if ("distinct_endpoints" in definition["validators"]
                    and relation.source_unit == relation.target_unit):
                raise ValueError(f"Relation {relation.relation_type} requires distinct cleaned units")

        layer_ids: dict[tuple[str, str], str] = {}
        created_nodes: list[str] = []
        mapping_ids: list[str] = []
        for unit in result.units:
            layer_key = (unit.target_existing_layer_id or unit.target_layer_type, unit.target_layer_name)
            if layer_key not in layer_ids:
                layer_ids[layer_key] = unit.target_existing_layer_id or self.engine.create_layer(LayerCreate(
                    name=unit.target_layer_name,
                    description=f"Schema-guided output from discovery {discovery_id}",
                    origin_type=unit.target_layer_type))
            node_id = self.engine.create_node(NodeCreate(
                layer_id=layer_ids[layer_key], title=unit.title, content=unit.content,
                node_type=unit.node_type, attributes=unit.attributes,
                provenance={"schema_discovery_id": discovery_id,
                            "source_node_ids": unit.source_node_ids,
                            "placement_plan_id": placement_plan_id,
                            "generation_provider": self.engine.generation.name},
            ))
            created_nodes.append(node_id)
            for source_id in unit.source_node_ids:
                mapping_ids.append(self.engine.create_mapping(MappingCreate(
                    source_node_id=node_id, target_node_id=source_id,
                    relation_type="core:derived_from", confidence=1.0,
                    evidence="Explicit source pointer from schema-guided cleaning", status="accepted")))
        for relation in result.relations:
            mapping_ids.append(self.engine.create_mapping(MappingCreate(
                source_node_id=created_nodes[relation.source_unit],
                target_node_id=created_nodes[relation.target_unit],
                relation_type=relation.relation_type, confidence=relation.confidence,
                evidence=relation.evidence, status="accepted")))
        return {
            "schema_discovery_id": discovery_id,
            "layer_ids": list(layer_ids.values()),
            "node_ids": created_nodes,
            "mapping_ids": mapping_ids,
            "source_node_ids": selected_ids,
        }

    def abstract_layer(self, source_layer_id: str, target_name: str,
                       target_description: str = "Machine-derived semantic units",
                       target_layer_type: str = "machine_derived") -> dict[str, Any]:
        if not self.engine.generation.available:
            raise RuntimeError("A generation provider is required for abstraction")
        readiness = self.abstraction_readiness([source_layer_id])
        if not readiness["eligible"]:
            raise ValueError("Dataset is not ready for abstraction: " + "; ".join(readiness["failed_conditions"]))
        sources = self.engine.repository.rows(
            "SELECT id,title,content,node_type,provenance_json FROM nodes WHERE layer_id=? ORDER BY created_at",
            (source_layer_id,),
        )
        if not sources:
            raise ValueError("Source layer has no nodes")
        allowed_sources = {item["id"] for item in sources}
        raw = self.engine.generation.structured(
            system=ABSTRACTION_SYSTEM, task="semantic_abstraction",
            payload={"source_layer_id": source_layer_id, "source_nodes": sources,
                     "abstraction_readiness": readiness,
                     "active_ontology": self.engine.repository.ontology_snapshot()},
            schema=AbstractionResult.model_json_schema(),
        )
        result = AbstractionResult.model_validate(raw)
        valid_units = []
        old_to_new = {}
        for index, unit in enumerate(result.units):
            valid_ids = sorted(set(unit.source_node_ids) & allowed_sources)
            if not valid_ids:
                continue
            old_to_new[index] = len(valid_units)
            valid_units.append(unit.model_copy(update={"source_node_ids": valid_ids}))
        if not valid_units:
            raise ValueError("Abstraction returned no units linked to valid source nodes")
        target_layer_id = self.engine.create_layer(LayerCreate(
            name=target_name, description=target_description, origin_type=target_layer_type))
        unit_ids = []
        for unit in valid_units:
            node_id = self.engine.create_node(NodeCreate(
                layer_id=target_layer_id, title=unit.title, content=unit.content,
                node_type=unit.node_type,
                provenance={"source_layer_id": source_layer_id,
                            "source_node_ids": unit.source_node_ids,
                            "generation_provider": self.engine.generation.name},
            ))
            unit_ids.append(node_id)
            for source_node_id in unit.source_node_ids:
                self.engine.create_mapping(MappingCreate(
                    source_node_id=node_id, target_node_id=source_node_id,
                    relation_type="core:derived_from", confidence=1.0,
                    evidence="Explicit input pointer returned by transformation protocol", status="accepted"))
        internal_ids = []
        proposed_relation_types = set()
        for relation in result.relations:
            if relation.source_unit not in old_to_new or relation.target_unit not in old_to_new:
                continue
            if not self.engine.repository.relation_type(relation.relation_type):
                proposed_relation_types.add(relation.relation_type)
                continue
            internal_ids.append(self.engine.create_mapping(MappingCreate(
                source_node_id=unit_ids[old_to_new[relation.source_unit]],
                target_node_id=unit_ids[old_to_new[relation.target_unit]],
                relation_type=relation.relation_type, confidence=relation.confidence,
                evidence=relation.evidence,
            )))
        return {"layer_id": target_layer_id, "node_ids": unit_ids, "mapping_ids": internal_ids,
                "proposed_relation_types": sorted(proposed_relation_types)}

    def semantic_candidates(self, reference_layer_ids: list[str], target_layer_ids: list[str],
                            per_target: int = 4, minimum_similarity: float = 0.30) -> list[dict[str, Any]]:
        if not reference_layer_ids or not target_layer_ids:
            raise ValueError("Reference and target layer selections are required")
        marks_ref = ",".join("?" for _ in reference_layer_ids)
        marks_target = ",".join("?" for _ in target_layer_ids)
        reference = self.engine.repository.rows(
            f"SELECT id,layer_id,title,content FROM nodes WHERE layer_id IN ({marks_ref})", reference_layer_ids)
        targets = self.engine.repository.rows(
            f"SELECT id,layer_id,title,content FROM nodes WHERE layer_id IN ({marks_target})", target_layer_ids)
        if not reference or not targets:
            return []
        ref_vectors = self.engine.embedder.encode([item["content"] for item in reference])
        target_vectors = self.engine.embedder.encode([item["content"] for item in targets])
        scores = target_vectors @ ref_vectors.T
        candidates = []
        for target_index, target in enumerate(targets):
            ranked = np.argsort(scores[target_index])[::-1][:per_target]
            for ref_index in ranked:
                score = float(scores[target_index, ref_index])
                if score >= minimum_similarity:
                    candidates.append({
                        "source_node_id": reference[int(ref_index)]["id"],
                        "target_node_id": target["id"],
                        "source_text": reference[int(ref_index)]["content"],
                        "target_text": target["content"], "similarity": score,
                    })
        return candidates

    def classify_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.engine.generation.available:
            return [{**item, "relation_type": "core:semantic_candidate",
                     "confidence": item["similarity"], "evidence": "Embedding similarity only"}
                    for item in candidates]
        aliases = {f"P{index}": item for index, item in enumerate(candidates, 1)}
        raw = self.engine.generation.structured(
            system=RELATION_SYSTEM, task="cross_layer_relations",
            payload={"active_relation_types": self.engine.repository.ontology_snapshot()["relation_types"],
                     "instructions": "Use an active relation ID, 'unrelated', or propose a new namespaced ID for review.",
                     "pairs": [{
                "pair_id": alias, "source": item["source_text"], "target": item["target_text"],
                "semantic_similarity": item["similarity"],
            } for alias, item in aliases.items()]},
            schema=RelationBatch.model_json_schema(),
        )
        decisions = RelationBatch.model_validate(raw)
        output = []
        for decision in decisions.relations:
            if decision.pair_id not in aliases:
                continue
            registered = self.engine.repository.relation_type(decision.relation_type)
            output.append({**aliases[decision.pair_id],
                           "relation_type": decision.relation_type,
                           "confidence": decision.confidence, "evidence": decision.evidence,
                           "ontology_status": "registered" if registered else
                           ("unrelated" if decision.relation_type == "unrelated" else "proposed")})
        return output

    def map_layers(self, reference_layer_ids: list[str], target_layer_ids: list[str],
                   per_target: int = 4, accept: bool = False) -> dict[str, Any]:
        candidates = self.semantic_candidates(reference_layer_ids, target_layer_ids, per_target)
        judged = self.classify_candidates(candidates)
        mapping_ids = []
        proposed_relation_types = set()
        for item in judged:
            if item["relation_type"] == "unrelated":
                continue
            if not self.engine.repository.relation_type(item["relation_type"]):
                proposed_relation_types.add(item["relation_type"])
                continue
            mapping_ids.append(self.engine.create_mapping(MappingCreate(
                source_node_id=item["source_node_id"], target_node_id=item["target_node_id"],
                relation_type=item["relation_type"], confidence=item["confidence"],
                evidence=item["evidence"], status="accepted" if accept else "suggested",
            )))
        return {"candidate_count": len(candidates), "relations": judged, "mapping_ids": mapping_ids,
                "proposed_relation_types": sorted(proposed_relation_types)}
