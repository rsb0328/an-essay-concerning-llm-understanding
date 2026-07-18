from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from essay_understanding.engine import MemoryEngine
from essay_understanding.models import (
    LayerCreate, MappingCreate, NodeCreate, QueryRequest, ShortcutCreate, ShortcutPlan,
)
from essay_understanding.providers.embeddings import HashingEmbedder
from essay_understanding.providers.generation import DisabledGenerationProvider
from essay_understanding.providers.generation import GenerationProvider
from essay_understanding.processing import KnowledgeProcessor
from essay_understanding.repository import Repository
from essay_understanding.vector_store import SQLiteVectorStore


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aec-tests-")
        self.repo = Repository(Path(self.temp.name) / "knowledge.db")
        self.embedder = HashingEmbedder(384)
        self.engine = MemoryEngine(
            self.repo, self.embedder, SQLiteVectorStore(self.repo), DisabledGenerationProvider())

    def tearDown(self):
        self.temp.cleanup()

    def layer(self, name: str, origin: str = "source") -> str:
        return self.engine.create_layer(LayerCreate(name=name, origin_type=origin))

    def node(self, layer: str, text: str) -> str:
        return self.engine.create_node(NodeCreate(layer_id=layer, title=text[:30], content=text))

    def test_runs_without_generation_model_and_returns_evidence(self):
        layer = self.layer("Source")
        node = self.node(layer, "A mapping is a conditional correspondence, not identity.")
        result = self.engine.query(QueryRequest(question="Why is a mapping not identity?", learn_shortcut=False))
        self.assertEqual(result["generation_mode"], "evidence_only")
        self.assertEqual(result["graph"]["nodes"][0]["node_id"], node)
        self.assertIsNone(result["answer"])

    def test_layers_are_peers_and_cycle_traversal_stops(self):
        first, second, third = self.layer("Text"), self.layer("Interpretation", "human_interpretation"), self.layer("Critique", "derived")
        a, b, c = self.node(first, "Knowledge begins from a finite perspective."), self.node(second, "Finite perspective requires selection."), self.node(third, "Selection can later be revised.")
        self.engine.create_mapping(MappingCreate(source_node_id=a, target_node_id=b, relation_type="interprets", confidence=.9))
        self.engine.create_mapping(MappingCreate(source_node_id=b, target_node_id=c, relation_type="extends", confidence=.9))
        self.engine.create_mapping(MappingCreate(source_node_id=c, target_node_id=a, relation_type="refers_to", confidence=.9))
        result = self.engine.query(QueryRequest(question="How can finite knowledge be revised?", max_depth=8, learn_shortcut=False))
        ids = [item["node_id"] for item in result["graph"]["nodes"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(result["graph"]["depth_reached"], 2)

    def test_shortcut_is_queried_before_graph_and_constrains_route(self):
        relevant = self.layer("Definitions")
        distractor = self.layer("Unrelated")
        wanted = self.node(relevant, "Contradiction requires incompatible claims under the same conditions.")
        self.node(distractor, "Contradiction is merely a popular dramatic technique in fiction.")
        shortcut_id = self.engine.create_shortcut(ShortcutCreate(
            name="Contradiction route", description="Begin with formal definitions.",
            trigger_examples=["What is a contradiction?"],
            plan=ShortcutPlan(start_layer_ids=[relevant], max_depth=0), status="active"))
        result = self.engine.query(QueryRequest(
            question="What is a contradiction?", max_depth=4, learn_shortcut=False))
        self.assertEqual(result["mode"], "shortcut_guided")
        self.assertEqual(result["shortcut"]["id"], shortcut_id)
        self.assertEqual({node["node_id"] for node in result["graph"]["nodes"]}, {wanted})
        self.assertEqual(result["graph"]["depth_reached"], 0)

    def test_repeated_success_promotes_candidate_route(self):
        layer = self.layer("Memory")
        self.node(layer, "A shortcut stores a reusable retrieval route rather than an answer.")
        request = QueryRequest(question="What does a shortcut store?", max_depth=0)
        first = self.engine.query(request)
        second = self.engine.query(request)
        third = self.engine.query(request)
        self.assertEqual(first["mode"], "free_exploration")
        self.assertEqual(second["mode"], "free_exploration")
        promoted = self.repo.shortcut(third["candidate_shortcut_id"])
        self.assertEqual(promoted["status"], "active")
        fourth = self.engine.query(request)
        self.assertEqual(fourth["mode"], "shortcut_guided")

    def test_export_contains_canonical_memory_not_derived_vectors(self):
        layer = self.layer("Source")
        self.node(layer, "Canonical text survives vector index replacement.")
        exported = self.repo.export_all()
        self.assertIn("layers", exported)
        self.assertIn("nodes", exported)
        self.assertIn("shortcuts", exported)
        self.assertNotIn("vectors", exported)

    def test_model_assisted_abstraction_is_validated_and_source_linked(self):
        class FakeGeneration(GenerationProvider):
            name = "fake"

            def structured(self, *, task, **_):
                self.last_task = task
                return {
                    "units": [
                        {"title": "Valid", "content": "Mapping preserves difference.",
                         "node_type": "claim", "source_node_ids": [source_id]},
                        {"title": "Invalid", "content": "Invented source.",
                         "node_type": "claim", "source_node_ids": ["missing"]},
                    ],
                    "relations": [],
                }

        source_layer = self.layer("Source")
        source_id = self.node(source_layer, "A mapping is not an identity relation.")
        self.engine.generation = FakeGeneration()
        result = KnowledgeProcessor(self.engine).abstract_layer(source_layer, "Abstraction")
        self.assertEqual(len(result["node_ids"]), 1)
        links = self.repo.rows("SELECT * FROM mappings WHERE relation_type='abstracted_from'")
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["target_node_id"], source_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
