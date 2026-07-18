from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from typing import Any

import numpy as np

from .models import (
    GroundedAnswer, LayerCreate, MappingCreate, NodeCreate, QueryRequest,
    ShortcutCreate, ShortcutPlan,
)
from .protocols import ANSWER_SYSTEM
from .providers.embeddings import Embedder
from .providers.generation import GenerationProvider
from .repository import Repository
from .vector_store import VectorStore


RELATION_WEIGHTS = {
    "abstracted_from": 1.0, "cites": 1.0, "defines": 0.95, "supports": 0.95,
    "contradicts": 0.95, "qualifies": 0.90, "interprets": 0.90,
    "derives_from": 0.90, "depends_on": 0.90, "extends": 0.80,
    "analogous_to": 0.70, "equivalent_to": 0.90, "refers_to": 0.75,
    "semantic_candidate": 0.45,
}


class MemoryEngine:
    def __init__(self, repository: Repository, embedder: Embedder,
                 vectors: VectorStore, generation: GenerationProvider):
        self.repository = repository
        self.embedder = embedder
        self.vectors = vectors
        self.generation = generation

    def create_layer(self, item: LayerCreate) -> str:
        return self.repository.create_layer(item)

    def create_node(self, item: NodeCreate) -> str:
        node_id = self.repository.create_node(item)
        vector = self.embedder.encode([item.content])[0]
        self.vectors.upsert("nodes", node_id, vector, {"layer_id": item.layer_id}, self.embedder.name)
        return node_id

    def create_mapping(self, item: MappingCreate) -> str:
        return self.repository.create_mapping(item)

    def ingest_text(self, layer_id: str, title: str, text: str,
                    chunk_chars: int = 1200, overlap: int = 120,
                    provenance: dict[str, Any] | None = None) -> list[str]:
        normalized = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n")).strip()
        if not normalized:
            raise ValueError("Text is empty")
        chunks, start = [], 0
        while start < len(normalized):
            end = min(len(normalized), start + chunk_chars)
            if end < len(normalized):
                boundary = max(normalized.rfind("\n", start, end), normalized.rfind("。", start, end))
                if boundary > start + chunk_chars // 2:
                    end = boundary + 1
            chunks.append(normalized[start:end].strip())
            if end >= len(normalized):
                break
            start = max(start + 1, end - overlap)
        document_hash = hashlib.sha256(normalized.encode()).hexdigest()
        ids = []
        for index, content in enumerate(filter(None, chunks)):
            ids.append(self.create_node(NodeCreate(
                layer_id=layer_id, title=f"{title} · {index + 1}", content=content,
                provenance={**(provenance or {}), "document_sha256": document_hash, "chunk": index},
            )))
        return ids

    def create_shortcut(self, item: ShortcutCreate) -> str:
        signature = self._plan_signature(item.plan)
        shortcut_id = self.repository.create_shortcut(item, signature)
        trigger_vectors = self.embedder.encode(item.trigger_examples)
        vector = np.mean(trigger_vectors, axis=0)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        self.vectors.upsert("shortcuts", shortcut_id, vector, {}, self.embedder.name)
        return shortcut_id

    @staticmethod
    def _shortcut_text(name: str, description: str, examples: list[str]) -> str:
        return "\n".join([name, description, *examples])

    @staticmethod
    def _plan_signature(plan: ShortcutPlan) -> str:
        canonical = {
            "layers": sorted(plan.start_layer_ids), "relations": sorted(plan.relation_types),
            "depth": plan.max_depth, "breadth": plan.breadth,
        }
        return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()

    def find_shortcut(self, question: str, threshold: float) -> dict[str, Any] | None:
        query_vector = self.embedder.encode([question])[0]
        for hit in self.vectors.search("shortcuts", query_vector, 8, self.embedder.name):
            shortcut = self.repository.shortcut(hit["item_id"])
            if not shortcut or shortcut["status"] != "active":
                continue
            maturity = min(1.0, shortcut["confirmations"] / 5)
            composite = 0.75 * hit["score"] + 0.20 * shortcut["reliability"] + 0.05 * maturity
            if composite >= threshold:
                return {**shortcut, "semantic_similarity": hit["score"], "match_confidence": composite}
        return None

    def retrieve(self, question: str, plan: ShortcutPlan) -> dict[str, Any]:
        query_vector = self.embedder.encode([question])[0]
        filters = {"layer_id": set(plan.start_layer_ids)} if plan.start_layer_ids else None
        initial = self.vectors.search("nodes", query_vector, max(30, plan.seed_nodes * 4),
                                      self.embedder.name, filters)
        layer_scores: dict[str, list[float]] = defaultdict(list)
        for hit in initial:
            layer_scores[hit["layer_id"]].append(hit["score"])
        ranked_layers = sorted(
            ({"layer_id": layer_id, "score": 0.7 * max(scores) + 0.3 * sum(sorted(scores, reverse=True)[:3]) / min(3, len(scores))}
             for layer_id, scores in layer_scores.items()),
            key=lambda item: item["score"], reverse=True,
        )
        allowed_layers = set(plan.start_layer_ids) if plan.start_layer_ids else {
            item["layer_id"] for item in ranked_layers[:3]}
        seeds = [hit for hit in initial if hit["layer_id"] in allowed_layers][:plan.seed_nodes]
        if not seeds:
            return {"question": question, "nodes": [], "edges": [], "selected_layers": [],
                    "depth_reached": 0, "stop_reason": "no_seed_nodes"}

        records = {row["id"]: row for row in self.repository.get_nodes([hit["item_id"] for hit in seeds])}
        best = {}
        for hit in seeds:
            if hit["item_id"] in records:
                best[hit["item_id"]] = {
                    **records[hit["item_id"]], "node_id": hit["item_id"], "score": hit["score"],
                    "retrieval_score": hit["score"], "path_score": hit["score"], "depth": 0,
                }
        frontier = list(best)
        edges: list[dict[str, Any]] = []
        edge_ids: set[str] = set()
        depth_reached = 0
        stop_reason = "max_depth"
        information_gain: list[float] = []

        for level in range(1, plan.max_depth + 1):
            mappings = self.repository.mappings_from(frontier, plan.relation_types or None)
            proposed: dict[str, dict[str, Any]] = {}
            for mapping in mappings:
                if mapping["source_node_id"] in frontier:
                    current, neighbor, direction = mapping["source_node_id"], mapping["target_node_id"], "forward"
                else:
                    current, neighbor, direction = mapping["target_node_id"], mapping["source_node_id"], "reverse"
                if mapping["id"] not in edge_ids:
                    edges.append({**mapping, "from_node_id": current, "to_node_id": neighbor,
                                  "direction": direction, "depth": level})
                    edge_ids.add(mapping["id"])
                if neighbor in best or current not in best:
                    continue
                path = (best[current]["path_score"] * mapping["confidence"]
                        * RELATION_WEIGHTS.get(mapping["relation_type"], 0.65) * 0.88)
                if neighbor not in proposed or path > proposed[neighbor]["path_score"]:
                    proposed[neighbor] = {"path_score": path, "parent_node_id": current}
            if not proposed:
                stop_reason = "frontier_exhausted"
                break
            neighbor_records = {row["id"]: row for row in self.repository.get_nodes(list(proposed))}
            valid = [node_id for node_id in proposed if node_id in neighbor_records]
            semantic = self.embedder.encode([neighbor_records[node_id]["content"] for node_id in valid]) @ query_vector
            ranked = []
            for node_id, semantic_score in zip(valid, semantic):
                combined = 0.6 * float(semantic_score) + 0.4 * proposed[node_id]["path_score"]
                ranked.append((combined, node_id, float(semantic_score)))
            ranked.sort(reverse=True)
            gain = sum(max(0.0, score - 0.40) for score, _, _ in ranked[:plan.breadth]) / max(1, min(plan.breadth, len(ranked)))
            information_gain.append(gain)
            if level > 1 and gain < plan.stop_min_information_gain:
                stop_reason = "low_information_gain"
                break
            frontier = []
            for combined, node_id, semantic_score in ranked[:plan.breadth]:
                best[node_id] = {
                    **neighbor_records[node_id], "node_id": node_id, "score": combined,
                    "retrieval_score": semantic_score, "path_score": proposed[node_id]["path_score"],
                    "parent_node_id": proposed[node_id]["parent_node_id"], "depth": level,
                }
                frontier.append(node_id)
            depth_reached = level
        nodes = sorted(best.values(), key=lambda item: (item["depth"], -item["score"]))
        return {
            "question": question, "nodes": nodes, "edges": edges,
            "selected_layers": ranked_layers[:3], "depth_reached": depth_reached,
            "information_gain": information_gain, "stop_reason": stop_reason,
        }

    def _answer(self, question: str, graph: dict[str, Any]) -> GroundedAnswer | None:
        if not self.generation.available:
            return None
        ranked = sorted(graph["nodes"], key=lambda node: node["score"], reverse=True)[:10]
        aliases = {f"E{index}": node["node_id"] for index, node in enumerate(ranked, 1)}
        evidence = [{
            "node_id": alias, "layer": node["layer_name"], "origin_type": node["origin_type"],
            "title": node["title"], "content": node["content"], "depth": node["depth"],
        } for alias, node in zip(aliases, ranked)]
        result = GroundedAnswer.model_validate(self.generation.structured(
            system=ANSWER_SYSTEM, task="grounded_answer",
            payload={"question": question, "evidence": evidence},
            schema=GroundedAnswer.model_json_schema(),
        ))
        unknown = {citation.node_id for citation in result.citations} - set(aliases)
        if unknown:
            raise ValueError(f"Generation provider cited unknown evidence: {sorted(unknown)}")
        return result.model_copy(update={
            "citations": [citation.model_copy(update={"node_id": aliases[citation.node_id]})
                          for citation in result.citations]
        })

    def _learn_candidate(self, question: str, graph: dict[str, Any], plan: ShortcutPlan) -> str | None:
        if not graph["nodes"]:
            return None
        traversed_relations = sorted({edge["relation_type"] for edge in graph["edges"]})
        traversed_layers = sorted({node["layer_id"] for node in graph["nodes"] if node["depth"] == 0})
        learned_plan = plan.model_copy(update={
            "start_layer_ids": traversed_layers,
            "relation_types": traversed_relations,
            "max_depth": graph["depth_reached"],
        })
        signature = self._plan_signature(learned_plan)
        existing = self.repository.candidate_by_signature(signature)
        if existing:
            self.repository.reinforce_shortcut_candidate(existing["id"])
            return existing["id"]
        item = ShortcutCreate(
            name=f"Route learned from: {question[:60]}",
            description="Candidate retrieval route derived from a successful query; awaiting repeated confirmation.",
            trigger_examples=[question], plan=learned_plan,
            preconditions=["A semantically similar query is asked"],
            validators=["citations_exist", "evidence_is_traceable"],
            failure_conditions=["No seed evidence is found", "The route yields insufficient evidence"],
            status="candidate",
        )
        return self.create_shortcut(item)

    def query(self, request: QueryRequest) -> dict[str, Any]:
        started = time.perf_counter()
        shortcut = self.find_shortcut(request.question, request.shortcut_threshold)
        if shortcut:
            plan = ShortcutPlan.model_validate(shortcut["plan"])
            if request.layer_ids:
                plan = plan.model_copy(update={"start_layer_ids": request.layer_ids})
            plan = plan.model_copy(update={"max_depth": min(request.max_depth, plan.max_depth)})
            mode = "shortcut_guided"
        else:
            plan = ShortcutPlan(
                start_layer_ids=request.layer_ids or [], relation_types=request.relation_types or [],
                max_depth=request.max_depth, breadth=request.breadth, seed_nodes=request.seed_nodes,
            )
            mode = "free_exploration"
        graph = self.retrieve(request.question, plan)
        answer = self._answer(request.question, graph)
        elapsed_ms = (time.perf_counter() - started) * 1000
        candidate_id = None
        if request.learn_shortcut and not shortcut and graph["nodes"]:
            candidate_id = self._learn_candidate(request.question, graph, plan)
        if shortcut:
            self.repository.record_shortcut_run(
                shortcut["id"], request.question, shortcut["semantic_similarity"],
                bool(graph["nodes"]), elapsed_ms, {"depth": graph["depth_reached"]})
        answer_data = answer.model_dump() if answer else None
        run_id = self.repository.record_query(
            question=request.question, mode=mode, shortcut_id=shortcut["id"] if shortcut else None,
            requested_depth=request.max_depth, reached_depth=graph["depth_reached"],
            evidence_ids=[node["node_id"] for node in graph["nodes"]], answer=answer_data,
            latency_ms=elapsed_ms, success=bool(graph["nodes"]),
        )
        return {
            "query_run_id": run_id, "mode": mode,
            "generation_mode": "grounded_answer" if answer else "evidence_only",
            "shortcut": shortcut, "candidate_shortcut_id": candidate_id,
            "graph": graph, "answer": answer_data, "latency_ms": round(elapsed_ms, 3),
        }
