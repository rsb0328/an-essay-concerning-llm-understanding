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

    def rebuild_indexes(self, batch_size: int = 64) -> dict[str, int | str]:
        """Recreate current-model node and shortcut vectors from canonical SQLite records."""
        nodes = self.repository.rows("SELECT id,layer_id,content FROM nodes ORDER BY created_at")
        node_count = 0
        for start in range(0, len(nodes), batch_size):
            batch = nodes[start:start + batch_size]
            vectors = self.embedder.encode([item["content"] for item in batch])
            for item, vector in zip(batch, vectors):
                self.vectors.upsert(
                    "nodes", item["id"], vector, {"layer_id": item["layer_id"]}, self.embedder.name)
                node_count += 1
        shortcuts = [self.repository.decode_shortcut(row) for row in self.repository.rows(
            "SELECT * FROM shortcuts ORDER BY created_at")]
        shortcut_count = 0
        for shortcut in shortcuts:
            trigger_vectors = self.embedder.encode(shortcut["trigger_examples"])
            vector = np.mean(trigger_vectors, axis=0)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            self.vectors.upsert("shortcuts", shortcut["id"], vector, {}, self.embedder.name)
            shortcut_count += 1
        return {"model": self.embedder.name, "nodes": node_count, "shortcuts": shortcut_count}

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
                node_type="passage",
                provenance={**(provenance or {}), "document_sha256": document_hash, "chunk": index},
            )))
        return ids

    def create_shortcut(self, item: ShortcutCreate) -> str:
        if item.plan.start_layer_ids:
            marks = ",".join("?" for _ in item.plan.start_layer_ids)
            known_layers = {row["id"] for row in self.repository.rows(
                f"SELECT id FROM layers WHERE id IN ({marks})", item.plan.start_layer_ids)}
            unknown_layers = sorted(set(item.plan.start_layer_ids) - known_layers)
            if unknown_layers:
                raise ValueError(f"Shortcut references unknown layers: {', '.join(unknown_layers)}")
        unknown_relations = sorted({relation for relation in item.plan.relation_types
                                    if not self.repository.relation_type(relation)})
        if unknown_relations:
            raise ValueError(f"Shortcut references unknown relation types: {', '.join(unknown_relations)}")
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
            "depth": plan.max_depth, "breadth": plan.breadth, "seed_nodes": plan.seed_nodes,
            "stop_min_information_gain": plan.stop_min_information_gain,
            "semantic_weight": plan.semantic_weight, "path_decay": plan.path_decay,
        }
        return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()

    def find_shortcut(self, question: str, threshold: float,
                      query_vector: np.ndarray | None = None) -> dict[str, Any] | None:
        query_vector = query_vector if query_vector is not None else self.embedder.encode([question])[0]
        matches = []
        for hit in self.vectors.search("shortcuts", query_vector, 8, self.embedder.name):
            shortcut = self.repository.shortcut(hit["item_id"])
            if not shortcut or shortcut["status"] != "active":
                continue
            maturity = min(1.0, shortcut["confirmations"] / 5)
            composite = 0.75 * hit["score"] + 0.20 * shortcut["reliability"] + 0.05 * maturity
            if composite >= threshold:
                matches.append({**shortcut, "semantic_similarity": hit["score"],
                                "match_confidence": composite})
        return max(matches, key=lambda item: item["match_confidence"]) if matches else None

    def retrieve(self, question: str, plan: ShortcutPlan,
                 query_vector: np.ndarray | None = None,
                 initial_hits: list[dict[str, Any]] | None = None,
                 as_of: str | None = None) -> dict[str, Any]:
        query_vector = query_vector if query_vector is not None else self.embedder.encode([question])[0]
        filters = {"layer_id": set(plan.start_layer_ids)} if plan.start_layer_ids else None
        initial = initial_hits if initial_hits is not None else self.vectors.search(
            "nodes", query_vector, max(30, plan.seed_nodes * 4), self.embedder.name, filters)
        if filters:
            initial = [hit for hit in initial if hit.get("layer_id") in filters["layer_id"]]
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
        initial_scores = np.asarray([float(hit["score"]) for hit in initial], dtype=np.float32)
        score_baseline = float(np.median(initial_scores)) if len(initial_scores) else 0.0
        score_scale = max(float(np.max(initial_scores)) - score_baseline, 1e-6) if len(initial_scores) else 1.0

        for level in range(1, plan.max_depth + 1):
            mappings = self.repository.mappings_from(frontier, plan.relation_types or None, as_of)
            proposed: dict[str, dict[str, Any]] = {}
            for mapping in mappings:
                if mapping["source_node_id"] in frontier:
                    current, neighbor, direction = mapping["source_node_id"], mapping["target_node_id"], "forward"
                else:
                    if mapping["directional"] and not mapping["symmetric"]:
                        continue
                    current, neighbor, direction = mapping["target_node_id"], mapping["source_node_id"], "reverse"
                if neighbor in best or current not in best:
                    continue
                path = (best[current]["path_score"] * mapping["confidence"]
                        * self.repository.relation_weight(mapping["relation_type"]) * plan.path_decay)
                if neighbor not in proposed or path > proposed[neighbor]["path_score"]:
                    proposed[neighbor] = {"path_score": path, "parent_node_id": current,
                                          "mapping": mapping, "direction": direction}
            if not proposed:
                stop_reason = "frontier_exhausted"
                break
            neighbor_records = {row["id"]: row for row in self.repository.get_nodes(list(proposed))}
            valid = [node_id for node_id in proposed if node_id in neighbor_records]
            if not valid:
                stop_reason = "frontier_records_missing"
                break
            stored = self.vectors.fetch("nodes", valid, self.embedder.name)
            missing = [node_id for node_id in valid if node_id not in stored]
            if missing:
                repaired = self.embedder.encode([neighbor_records[node_id]["content"] for node_id in missing])
                for node_id, vector in zip(missing, repaired):
                    stored[node_id] = vector
                    self.vectors.upsert("nodes", node_id, vector,
                                        {"layer_id": neighbor_records[node_id]["layer_id"]}, self.embedder.name)
            semantic = np.asarray([stored[node_id] for node_id in valid]) @ query_vector
            ranked = []
            for node_id, semantic_score in zip(valid, semantic):
                combined = (plan.semantic_weight * float(semantic_score)
                            + (1 - plan.semantic_weight) * proposed[node_id]["path_score"])
                ranked.append((combined, node_id, float(semantic_score)))
            ranked.sort(reverse=True)
            selected_semantic = [semantic_score for _, _, semantic_score in ranked[:plan.breadth]]
            gain = sum(min(1.0, max(0.0, (score - score_baseline) / score_scale))
                       for score in selected_semantic) / max(1, len(selected_semantic))
            information_gain.append(gain)
            if level > 1 and gain < plan.stop_min_information_gain:
                stop_reason = "low_information_gain"
                break
            frontier = []
            for combined, node_id, semantic_score in ranked[:plan.breadth]:
                selected_mapping = proposed[node_id]["mapping"]
                if selected_mapping["id"] not in edge_ids:
                    edges.append({**selected_mapping,
                                  "from_node_id": proposed[node_id]["parent_node_id"],
                                  "to_node_id": node_id,
                                  "direction": proposed[node_id]["direction"], "depth": level})
                    edge_ids.add(selected_mapping["id"])
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

    def _learn_candidate(self, question: str, graph: dict[str, Any], plan: ShortcutPlan,
                         query_vector: np.ndarray, answer: GroundedAnswer | None,
                         similarity_threshold: float,
                         novelty_ceiling: float) -> str | None:
        # Automatic route learning requires a grounded outcome. Merely returning any
        # node is not evidence that the route answered the question.
        if not graph["nodes"] or answer is None or not answer.citations:
            return None
        traversed_relations = sorted({edge["relation_type"] for edge in graph["edges"]})
        traversed_layers = sorted({node["layer_id"] for node in graph["nodes"] if node["depth"] == 0})
        learned_plan = plan.model_copy(update={
            "start_layer_ids": traversed_layers,
            "relation_types": traversed_relations,
            "max_depth": graph["depth_reached"],
        })
        signature = self._plan_signature(learned_plan)
        candidates = self.repository.candidates_by_signature(signature)
        matched: tuple[float, dict[str, Any]] | None = None
        for candidate in candidates:
            stored = self.vectors.fetch("shortcuts", [candidate["id"]], self.embedder.name)
            prototype = stored.get(candidate["id"])
            if prototype is None:
                trigger_vectors = self.embedder.encode(candidate["trigger_examples"])
                prototype = np.mean(trigger_vectors, axis=0)
                prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
                self.vectors.upsert("shortcuts", candidate["id"], prototype, {}, self.embedder.name)
            similarity = float(prototype @ query_vector)
            if matched is None or similarity > matched[0]:
                matched = (similarity, candidate)
        if matched and matched[0] >= similarity_threshold:
            existing = matched[1]
            example_vectors = self.embedder.encode(existing["trigger_examples"])
            max_example_similarity = max(float(vector @ query_vector) for vector in example_vectors)
            if max_example_similarity >= novelty_ceiling:
                return existing["id"]
            if self.repository.reinforce_shortcut_candidate(existing["id"], question):
                updated = self.repository.shortcut(existing["id"])
                trigger_vectors = self.embedder.encode(updated["trigger_examples"])
                prototype = np.mean(trigger_vectors, axis=0)
                prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
                self.vectors.upsert("shortcuts", existing["id"], prototype, {}, self.embedder.name)
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
        query_vector = self.embedder.encode([request.question])[0]
        shortcut = self.find_shortcut(
            request.question, request.shortcut_threshold, query_vector=query_vector)
        global_probe: list[dict[str, Any]] | None = None
        shortcut_attempt_ms = 0.0
        shortcut_route_succeeded = False
        if shortcut:
            plan = ShortcutPlan.model_validate(shortcut["plan"])
            if request.layer_ids:
                plan = plan.model_copy(update={"start_layer_ids": request.layer_ids})
            plan = plan.model_copy(update={"max_depth": min(request.max_depth, plan.max_depth)})
            mode = "shortcut_guided"
            attempt_started = time.perf_counter()
            global_probe = self.vectors.search(
                "nodes", query_vector, max(30, request.seed_nodes * 4), self.embedder.name)
            route_layers = set(plan.start_layer_ids)
            route_is_plausible = not route_layers or any(
                hit.get("layer_id") in route_layers for hit in global_probe)
            if route_is_plausible:
                graph = self.retrieve(
                    request.question, plan, query_vector=query_vector,
                    initial_hits=global_probe, as_of=request.as_of)
                shortcut_route_succeeded = bool(graph["nodes"])
            else:
                graph = {"question": request.question, "nodes": [], "edges": [],
                         "selected_layers": [], "depth_reached": 0,
                         "information_gain": [], "stop_reason": "shortcut_precheck_failed"}
            shortcut_attempt_ms = (time.perf_counter() - attempt_started) * 1000
        else:
            plan = ShortcutPlan(
                start_layer_ids=request.layer_ids or [], relation_types=request.relation_types or [],
                max_depth=request.max_depth, breadth=request.breadth, seed_nodes=request.seed_nodes,
            )
            mode = "free_exploration"
            graph = self.retrieve(
                request.question, plan, query_vector=query_vector, as_of=request.as_of)
        if shortcut and not shortcut_route_succeeded:
            plan = ShortcutPlan(
                start_layer_ids=request.layer_ids or [], relation_types=request.relation_types or [],
                max_depth=request.max_depth, breadth=request.breadth, seed_nodes=request.seed_nodes,
            )
            graph = self.retrieve(
                request.question, plan, query_vector=query_vector,
                initial_hits=global_probe, as_of=request.as_of)
            mode = "shortcut_fallback"
        try:
            answer = self._answer(request.question, graph)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            if shortcut:
                self.repository.record_shortcut_run(
                    shortcut["id"], request.question, shortcut["semantic_similarity"], False,
                    elapsed_ms, {"depth": graph["depth_reached"], "fallback": True,
                                 "false_route": True, "wasted_ms": shortcut_attempt_ms})
            self.repository.record_query(
                question=request.question, mode=mode,
                shortcut_id=shortcut["id"] if shortcut else None,
                requested_depth=request.max_depth, reached_depth=graph["depth_reached"],
                evidence_ids=[node["node_id"] for node in graph["nodes"]], answer=None,
                latency_ms=elapsed_ms, success=False,
            )
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        candidate_id = None
        if request.learn_shortcut and mode in {"free_exploration", "shortcut_fallback"} and graph["nodes"]:
            candidate_id = self._learn_candidate(
                request.question, graph, plan, query_vector, answer,
                request.shortcut_learning_similarity,
                request.shortcut_learning_novelty_ceiling)
        if shortcut:
            self.repository.record_shortcut_run(
                shortcut["id"], request.question, shortcut["semantic_similarity"],
                shortcut_route_succeeded, elapsed_ms,
                {"depth": graph["depth_reached"], "fallback": not shortcut_route_succeeded,
                 "false_route": not shortcut_route_succeeded,
                 "wasted_ms": shortcut_attempt_ms if not shortcut_route_succeeded else 0.0})
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
            "shortcut_diagnostics": {
                "attempted": bool(shortcut),
                "false_route": bool(shortcut and not shortcut_route_succeeded),
                "fallback": mode == "shortcut_fallback",
                "wasted_ms": round(shortcut_attempt_ms if mode == "shortcut_fallback" else 0.0, 3),
            },
            "graph": graph, "answer": answer_data, "latency_ms": round(elapsed_ms, 3),
        }
