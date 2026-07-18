from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .engine import MemoryEngine
from .models import LayerCreate, MappingCreate, NodeCreate
from .protocols import ABSTRACTION_SYSTEM, RELATION_SYSTEM


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


class KnowledgeProcessor:
    """Model-assisted processing with application-side validation and persistence."""

    def __init__(self, engine: MemoryEngine):
        self.engine = engine

    def abstract_layer(self, source_layer_id: str, target_name: str,
                       target_description: str = "Machine-derived semantic units",
                       target_layer_type: str = "machine_derived") -> dict[str, Any]:
        if not self.engine.generation.available:
            raise RuntimeError("A generation provider is required for abstraction")
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
