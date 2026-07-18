from __future__ import annotations

import tempfile
import unittest
import sqlite3
from pathlib import Path

from essay_understanding.engine import MemoryEngine
from essay_understanding.models import (
    LayerCreate, LayerTypeCreate, MappingCreate, NodeCreate, OntologyBundle, QueryRequest,
    RelationTypeCreate, ShortcutCreate, ShortcutPlan,
)
from essay_understanding.providers.embeddings import HashingEmbedder
from essay_understanding.providers.generation import DisabledGenerationProvider
from essay_understanding.providers.generation import GenerationProvider
from essay_understanding.processing import AbstractionReadinessPolicy, KnowledgeProcessor
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

    def layer(self, name: str, origin: str = "input") -> str:
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
        first, second, third = self.layer("Input"), self.layer("External analysis", "external"), self.layer("Derived", "derived")
        a, b, c = self.node(first, "Knowledge begins from a finite perspective."), self.node(second, "Finite perspective requires selection."), self.node(third, "Selection can later be revised.")
        for relation in ("interprets", "extends", "refers_to"):
            self.repo.register_relation_type(RelationTypeCreate(
                id=f"test:{relation}", label=relation, namespace="test", default_traversal_weight=.9))
        self.engine.create_mapping(MappingCreate(source_node_id=a, target_node_id=b, relation_type="test:interprets", confidence=.9))
        self.engine.create_mapping(MappingCreate(source_node_id=b, target_node_id=c, relation_type="test:extends", confidence=.9))
        self.engine.create_mapping(MappingCreate(source_node_id=c, target_node_id=a, relation_type="test:refers_to", confidence=.9))
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
        shortcut_vectors = self.repo.rows("SELECT * FROM vectors WHERE namespace='shortcuts'")
        self.assertEqual([row["item_id"] for row in shortcut_vectors], [shortcut_id])

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
                    "relations": [
                        {"source_unit": 0, "target_unit": 0,
                         "relation_type": "workspace:needs_review", "confidence": .7,
                         "evidence": "A proposed domain relation"}
                    ],
                }

        source_layer = self.layer("Source")
        source_id = self.node(source_layer, "A mapping is not an identity relation.")
        self.engine.generation = FakeGeneration()
        processor = KnowledgeProcessor(self.engine, AbstractionReadinessPolicy(
            min_nodes=1, min_chars=1, short_record_nodes=1, required_surveys=1))
        result = processor.abstract_layer(source_layer, "Abstraction")
        self.assertEqual(len(result["node_ids"]), 1)
        self.assertEqual(result["proposed_relation_types"], ["workspace:needs_review"])
        self.assertIsNone(self.repo.relation_type("workspace:needs_review"))
        links = self.repo.rows("SELECT * FROM mappings WHERE relation_type='core:derived_from'")
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["target_node_id"], source_id)

    def test_domain_ontology_is_open_but_governed(self):
        bundle = OntologyBundle(
            name="Company",
            layer_types=[
                LayerTypeCreate(id="company:person", label="Person", namespace="company"),
                LayerTypeCreate(id="company:organization", label="Organization", namespace="company"),
            ],
            relation_types=[RelationTypeCreate(
                id="company:reports_to", label="reports to", namespace="company",
                inverse_type="company:manages", default_traversal_weight=.92,
                allowed_source_types=["company:person"],
                allowed_target_types=["company:person", "company:organization"],
                validators=["distinct_endpoints"],
            )],
        )
        self.repo.import_ontology(bundle)
        person_layer = self.layer("People", "company:person")
        org_layer = self.layer("Organizations", "company:organization")
        person, organization = self.node(person_layer, "Ava is the finance lead."), self.node(org_layer, "North division")
        mapping = self.engine.create_mapping(MappingCreate(
            source_node_id=person, target_node_id=organization,
            relation_type="company:reports_to", confidence=.95,
            attributes={"scope": "finance"}, valid_from="2026-01-01"))
        self.assertTrue(mapping)
        self.assertEqual(self.repo.relation_weight("company:reports_to"), .92)
        stored = self.repo.rows("SELECT * FROM mappings WHERE id=?", (mapping,))[0]
        self.assertEqual(stored["attributes_json"], '{"scope": "finance"}')
        self.assertEqual(stored["valid_from"], "2026-01-01")
        with self.assertRaisesRegex(ValueError, "Unknown relation type"):
            self.engine.create_mapping(MappingCreate(
                source_node_id=person, target_node_id=organization,
                relation_type="company:invented_relation", confidence=.5))
        with self.assertRaisesRegex(ValueError, "Unknown endpoint layer types"):
            self.repo.register_relation_type(RelationTypeCreate(
                id="company:bad_endpoint", label="bad endpoint", namespace="company",
                allowed_target_types=["company:not_registered"]))
        with self.assertRaisesRegex(ValueError, "unknown relation types"):
            self.engine.create_shortcut(ShortcutCreate(
                name="Invalid route", description="Must fail closed.",
                trigger_examples=["invalid"],
                plan=ShortcutPlan(relation_types=["company:not_registered"])))

    def test_legacy_mapping_table_gains_property_and_validity_columns(self):
        path = Path(self.temp.name) / "legacy.db"
        db = sqlite3.connect(path)
        try:
            db.execute("""CREATE TABLE layers (
                id TEXT PRIMARY KEY, name TEXT, description TEXT,
                origin_type TEXT, created_at TEXT)""")
            db.execute("INSERT INTO layers VALUES('legacy-input','Legacy input','','input','2025-01-01')")
            db.execute("""CREATE TABLE nodes (
                id TEXT PRIMARY KEY, layer_id TEXT, title TEXT, content TEXT,
                node_type TEXT, provenance_json TEXT, maturity TEXT, created_at TEXT)""")
            db.execute("""CREATE TABLE mappings (
                id TEXT PRIMARY KEY, source_node_id TEXT, target_node_id TEXT,
                relation_type TEXT, confidence REAL, evidence TEXT, status TEXT,
                created_at TEXT)""")
            db.commit()
        finally:
            db.close()
        migrated = Repository(path)
        columns = {row["name"] for row in migrated.rows("PRAGMA table_info(mappings)")}
        self.assertTrue({"attributes_json", "valid_from", "valid_to"}.issubset(columns))
        node_columns = {row["name"] for row in migrated.rows("PRAGMA table_info(nodes)")}
        self.assertIn("attributes_json", node_columns)
        self.assertEqual(migrated.rows(
            "SELECT is_initial FROM layers WHERE id='legacy-input'")[0]["is_initial"], 1)

    def test_initial_layer_is_historical_provenance_not_a_second_root(self):
        first = self.layer("First admitted source")
        second = self.layer("Later source")
        rows = {row["id"]: row for row in self.repo.rows("SELECT * FROM layers")}
        self.assertEqual(rows[first]["is_initial"], 1)
        self.assertEqual(rows[second]["is_initial"], 0)
        with self.assertRaisesRegex(ValueError, "already has"):
            self.engine.create_layer(LayerCreate(
                name="False second beginning", origin_type="input", is_initial=True))
        with self.assertRaisesRegex(ValueError, "must use"):
            self.engine.create_layer(LayerCreate(
                name="External cannot be initial", origin_type="external", is_initial=True))

    def test_readiness_and_external_placement_are_auditable(self):
        class PlacementGeneration(GenerationProvider):
            name = "placement-fake"

            def structured(self, *, task, payload, **_):
                source_ids = [item["id"] for item in payload["source_nodes"]]
                return {
                    "summary": "Different author and viewpoint require a peer layer.",
                    "decisions": [{
                        "source_node_ids": source_ids,
                        "action": "create_peer_layer",
                        "target_layer_type": "external",
                        "target_layer_name": "External commentary",
                        "independence_signals": ["different_author", "independent_viewpoint"],
                        "rationale": "The material has independent provenance and authority.",
                    }],
                }

        raw = self.layer("Initial text")
        raw_node = self.node(raw, "Original source passage")
        processor = KnowledgeProcessor(self.engine, AbstractionReadinessPolicy(
            min_nodes=2, min_chars=1000, short_record_nodes=5, required_surveys=2))
        self.assertFalse(processor.abstraction_readiness([raw])["eligible"])
        hits = self.engine.vectors.search(
            "nodes", self.embedder.encode(["Original source passage"])[0], 1,
            self.embedder.name, {"layer_id": [raw]})
        self.assertEqual(hits[0]["item_id"], raw_node)
        self.engine.generation = PlacementGeneration()
        plan = processor.plan_material_placement([raw], "external_source", sample_limit=1)
        self.assertEqual(plan["status"], "pending")
        self.assertEqual(plan["plan"]["decisions"][0]["action"], "create_peer_layer")
        self.assertEqual(self.repo.decide_placement_plan(plan["id"], "approved")["status"], "approved")
        self.assertEqual(self.repo.export_all()["placement_plans"][0]["status"], "approved")

    def test_schema_discovery_proposes_compares_and_requires_approval(self):
        class DiscoveryGeneration(GenerationProvider):
            name = "discovery-fake"

            def structured(self, *, task, payload, **_):
                self.task = task
                self.payload = payload
                if task == "schema_guided_cleaning":
                    source_id = payload["source_nodes"][0]["id"]
                    return {
                        "units": [
                            {"target_layer_type": "company:people", "target_layer_name": "People",
                             "title": "Employee E0", "content": "Employee E0",
                             "node_type": "company:employee", "source_node_ids": [source_id]},
                            {"target_layer_type": "company:projects", "target_layer_name": "Projects",
                             "title": "Project P0", "content": "Project P0 is due on 2026-08-01.",
                             "node_type": "entity", "attributes": {"company:deadline": "2026-08-01"},
                             "source_node_ids": [source_id]},
                        ],
                        "relations": [
                            {"source_unit": 0, "target_unit": 1,
                             "relation_type": "company:owns_project", "confidence": .98,
                             "evidence": "The raw record explicitly states ownership."}
                        ],
                    }
                return {
                    "dataset_summary": "Company records contain people, projects, deadlines, and ownership.",
                    "candidates": [
                        {"kind": "layer_type", "id": "company:people", "label": "People",
                         "description": "Personnel records", "rationale": "Independent identity and access."},
                        {"kind": "layer_type", "id": "company:projects", "label": "Projects",
                         "description": "Project records", "rationale": "Independent lifecycle and retrieval."},
                        {"kind": "node_type", "id": "company:employee", "label": "Employee",
                         "description": "A person employed by the company", "rationale": "Stable entity identity."},
                        {"kind": "attribute", "id": "company:deadline", "label": "Deadline",
                         "description": "A project due date", "rationale": "A scalar value, not a layer.",
                         "value_type": "datetime"},
                        {"kind": "relation_type", "id": "company:owns_project", "label": "owns project",
                         "description": "A people record owns a project", "rationale": "Verifiable cross-layer link.",
                         "allowed_source_types": ["company:people"],
                         "allowed_target_types": ["company:projects"],
                         "default_traversal_weight": 0.9},
                    ],
                    "cleaning_guidance": ["Preserve employee identifiers and deadline timestamps."],
                }

        raw = self.layer("Raw company inbox")
        for index in range(8):
            self.node(raw, f"Record {index}: employee E{index} owns project P{index}; deadline 2026-08-{index + 1:02d}.")
        with self.assertRaisesRegex(RuntimeError, "generation provider"):
            KnowledgeProcessor(self.engine).discover_schema([raw], "company", sample_limit=4)
        generation = DiscoveryGeneration()
        self.engine.generation = generation
        processor = KnowledgeProcessor(self.engine, AbstractionReadinessPolicy(
            min_nodes=4, min_chars=1, short_record_nodes=4, required_surveys=2))
        discovery = processor.discover_schema(
            [raw], "company", sample_limit=4, max_chars_per_node=500)
        self.assertEqual(generation.task, "schema_discovery")
        self.assertEqual(len(generation.payload["representative_raw_sample"]), 4)
        self.assertEqual(discovery["status"], "pending")
        self.assertEqual(len(discovery["comparisons"]), 5)
        self.assertFalse(self.repo.rows("SELECT id FROM layer_types WHERE id='company:projects'"))
        with self.assertRaisesRegex(ValueError, "must be approved"):
            processor.clean_with_schema(discovery["id"], max_nodes=4)

        keys = [f"{item['kind']}:{item['id']}" for item in discovery["candidates"]]
        with self.assertRaisesRegex(ValueError, "enough independent surveys"):
            self.repo.approve_schema_discovery(discovery["id"], keys)
        confirmation = processor.discover_schema(
            [raw], "company", sample_limit=4, max_chars_per_node=500)
        self.assertEqual(confirmation["survey_round"], 2)
        self.assertNotEqual(discovery["sample_node_ids"], confirmation["sample_node_ids"])
        approved = self.repo.approve_schema_discovery(confirmation["id"], keys)
        self.assertEqual(len(approved["approved_candidates"]), 5)
        ontology = self.repo.ontology_snapshot()
        self.assertIn("company:projects", {item["id"] for item in ontology["layer_types"]})
        self.assertIn("company:owns_project", {item["id"] for item in ontology["relation_types"]})
        self.assertIn("company:deadline", {item["id"] for item in ontology["semantic_dimensions"]})
        self.assertEqual(self.repo.schema_discovery(confirmation["id"])["status"], "approved")

        cleaned = processor.clean_with_schema(confirmation["id"], max_nodes=4)
        self.assertEqual(len(cleaned["layer_ids"]), 2)
        self.assertEqual(len(cleaned["node_ids"]), 2)
        self.assertEqual(len(cleaned["mapping_ids"]), 3)
        project = self.repo.rows("SELECT * FROM nodes WHERE id=?", (cleaned["node_ids"][1],))[0]
        self.assertEqual(project["attributes_json"], '{"company:deadline": "2026-08-01"}')
        self.assertEqual(generation.task, "schema_guided_cleaning")

        selected_source = confirmation["sample_node_ids"][0]
        placement_id = self.repo.create_placement_plan(
            source_layer_ids=[raw], source_node_ids=[selected_source],
            material_origin="external_source", generation_provider="test",
            plan={"summary": "A mixed record routes to two governed peer layers.", "decisions": [
                {"source_node_ids": [selected_source], "action": "create_peer_layer",
                 "target_existing_layer_id": None, "target_layer_type": "company:people",
                 "target_layer_name": "People", "same_source_continuation": False,
                 "independence_signals": ["independent_lifecycle"], "rationale": "Person identity."},
                {"source_node_ids": [selected_source], "action": "create_peer_layer",
                 "target_existing_layer_id": None, "target_layer_type": "company:projects",
                 "target_layer_name": "Projects", "same_source_continuation": False,
                 "independence_signals": ["independent_lifecycle"], "rationale": "Project identity."},
            ]})
        self.repo.decide_placement_plan(placement_id, "approved")
        constrained = processor.clean_with_schema(
            confirmation["id"], [selected_source], max_nodes=1, placement_plan_id=placement_id)
        constrained_node = self.repo.rows(
            "SELECT provenance_json FROM nodes WHERE id=?", (constrained["node_ids"][0],))[0]
        self.assertIn(placement_id, constrained_node["provenance_json"])

        repeated = processor.discover_schema([raw], "company", sample_limit=4)
        self.assertTrue(all(item["recommendation"] == "reuse_existing"
                            for item in repeated["comparisons"]))
        default_approval = self.repo.approve_schema_discovery(repeated["id"])
        self.assertEqual(default_approval["approved_candidates"], [])
        rejected = processor.discover_schema([raw], "company", sample_limit=4)
        self.assertEqual(self.repo.reject_schema_discovery(rejected["id"])["status"], "rejected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
