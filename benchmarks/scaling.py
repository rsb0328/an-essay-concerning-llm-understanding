"""Generate a synthetic memory and measure retrieval modes at one scale point.

This intentionally excludes generation latency so retrieval architecture can be compared
independently of the chosen LLM. Run each scale in a fresh process.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
import time
from pathlib import Path

from essay_understanding.engine import MemoryEngine
from essay_understanding.models import LayerCreate, MappingCreate, NodeCreate, QueryRequest, RelationTypeCreate, ShortcutCreate, ShortcutPlan
from essay_understanding.providers.embeddings import HashingEmbedder
from essay_understanding.providers.generation import DisabledGenerationProvider
from essay_understanding.repository import Repository
from essay_understanding.vector_store import SQLiteVectorStore


TOPICS = [
    "identity and conditional mapping", "contradiction and revision", "provenance and testimony",
    "abstraction and transfer", "memory consolidation", "analogy and its limits",
    "evidence and uncertainty", "definition and qualification", "causation and explanation",
    "procedural memory and learned routes",
]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--mappings-per-node", type=int, default=3)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    temp = tempfile.TemporaryDirectory(prefix="aec-scaling-")
    repository = Repository(Path(temp.name) / "knowledge.db")
    embedder = HashingEmbedder(384)
    engine = MemoryEngine(repository, embedder, SQLiteVectorStore(repository), DisabledGenerationProvider())
    for relation in ("qualifies", "supports", "contradicts"):
        repository.register_relation_type(RelationTypeCreate(
            id=f"benchmark:{relation}", label=relation, namespace="benchmark",
            default_traversal_weight=0.9))
    layer_ids = [engine.create_layer(LayerCreate(
        name=f"Synthetic layer {index + 1}", origin_type="input" if index == 0 else "derived"))
        for index in range(args.layers)]

    started = time.perf_counter()
    node_ids = []
    for index in range(args.nodes):
        topic = TOPICS[index % len(TOPICS)]
        layer_id = layer_ids[index % len(layer_ids)]
        node_ids.append(engine.create_node(NodeCreate(
            layer_id=layer_id, title=f"Synthetic unit {index}",
            content=f"Unit {index} discusses {topic}. It preserves source {index % 97} and condition {index % 31}.")))
    for index, source in enumerate(node_ids):
        for offset in range(1, args.mappings_per_node + 1):
            target = node_ids[(index + offset * len(layer_ids)) % len(node_ids)]
            if source != target:
                engine.create_mapping(MappingCreate(
                    source_node_id=source, target_node_id=target,
                    relation_type=f"benchmark:{('qualifies', 'supports', 'contradicts')[offset % 3]}",
                    confidence=0.7 + 0.05 * (offset % 3)))
    build_ms = (time.perf_counter() - started) * 1000

    engine.create_shortcut(ShortcutCreate(
        name="Synthetic topic route", description="Restrict repeated topic queries to two layers.",
        trigger_examples=["How do contradiction and revision interact?"],
        plan=ShortcutPlan(start_layer_ids=layer_ids[:2], max_depth=2, breadth=5), status="active"))
    questions = [f"How does {TOPICS[index % len(TOPICS)]} work under condition {index % 31}?"
                 for index in range(args.queries)]
    modes = {
        "single": dict(max_depth=0, learn_shortcut=False, shortcut_threshold=1.0),
        "multi": dict(max_depth=2, learn_shortcut=False, shortcut_threshold=1.0),
        "shortcut_first": dict(max_depth=2, learn_shortcut=False, shortcut_threshold=0.45),
    }
    results = {}
    for mode, parameters in modes.items():
        timings, nodes, depths, hits = [], [], [], 0
        for question in questions:
            result = engine.query(QueryRequest(question=question, **parameters))
            timings.append(result["latency_ms"])
            nodes.append(len(result["graph"]["nodes"]))
            depths.append(result["graph"]["depth_reached"])
            hits += int(result["mode"] == "shortcut_guided")
        results[mode] = {
            "mean_ms": statistics.mean(timings), "p50_ms": percentile(timings, .50),
            "p95_ms": percentile(timings, .95), "mean_evidence_nodes": statistics.mean(nodes),
            "mean_depth": statistics.mean(depths), "shortcut_hits": hits,
        }
    output = {
        "schema": "aec-scaling-v1", "synthetic": True, "seed": args.seed,
        "nodes": args.nodes, "layers": args.layers,
        "mappings_per_node": args.mappings_per_node, "queries": args.queries,
        "embedding": embedder.name, "vector_store": "sqlite", "build_ms": build_ms,
        "results": results,
    }
    rendered = json.dumps(output, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
