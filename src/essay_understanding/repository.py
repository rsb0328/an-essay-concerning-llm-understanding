from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import (
    LayerCreate, LayerTypeCreate, MappingCreate, NodeCreate, OntologyBundle,
    RelationTypeCreate, SchemaCandidate, ShortcutCreate,
)


DEFAULT_LAYER_TYPES = (
    LayerTypeCreate(id="input", label="Admitted input", description="Canonical material admitted to a workspace", namespace="core"),
    LayerTypeCreate(id="external", label="Externally supplied", description="Information supplied by a person or external system", namespace="core"),
    LayerTypeCreate(id="machine_derived", label="Machine derived", description="Units derived by a configured processing provider", namespace="core"),
    LayerTypeCreate(id="derived", label="Derived knowledge", namespace="core"),
    LayerTypeCreate(id="procedural", label="Procedural memory", namespace="core"),
)

DEFAULT_RELATION_TYPES = (
    RelationTypeCreate(id="core:derived_from", label="derived from", namespace="core", default_traversal_weight=0.9),
    RelationTypeCreate(id="core:cites", label="cites", namespace="core", default_traversal_weight=1.0),
    RelationTypeCreate(id="core:contains", label="contains", namespace="core", default_traversal_weight=0.8),
    RelationTypeCreate(id="core:version_of", label="version of", namespace="core", default_traversal_weight=0.9),
    RelationTypeCreate(id="core:replaced_by", label="replaced by", namespace="core", temporal=True, default_traversal_weight=0.8),
    RelationTypeCreate(id="core:semantic_candidate", label="semantic candidate", namespace="core", default_traversal_weight=0.45),
)

DEFAULT_SEMANTIC_DIMENSIONS = (
    SchemaCandidate(kind="node_type", id="semantic_unit", label="Semantic unit",
                    rationale="Domain-neutral canonical unit"),
    SchemaCandidate(kind="node_type", id="passage", label="Passage",
                    rationale="Contiguous text segment"),
    SchemaCandidate(kind="node_type", id="record", label="Record",
                    rationale="Structured admitted record"),
    SchemaCandidate(kind="node_type", id="entity", label="Entity",
                    rationale="Stable identifiable object"),
    SchemaCandidate(kind="node_type", id="event", label="Event",
                    rationale="Time-bounded occurrence"),
    SchemaCandidate(kind="node_type", id="measurement", label="Measurement",
                    rationale="Observed quantitative value"),
    SchemaCandidate(kind="node_type", id="requirement", label="Requirement",
                    rationale="Normative or operational constraint"),
    SchemaCandidate(kind="node_type", id="claim", label="Claim",
                    rationale="Propositional assertion"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    """Canonical memory. Vector indexes are derived and may be rebuilt at any time."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def initialize(self) -> None:
        with self.connection() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS layer_types (
              id TEXT PRIMARY KEY, label TEXT NOT NULL, description TEXT NOT NULL,
              namespace TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS relation_types (
              id TEXT PRIMARY KEY, label TEXT NOT NULL, description TEXT NOT NULL,
              namespace TEXT NOT NULL, inverse_type TEXT, directional INTEGER NOT NULL,
              symmetric INTEGER NOT NULL, transitive INTEGER NOT NULL, temporal INTEGER NOT NULL,
              default_traversal_weight REAL NOT NULL,
              allowed_source_types_json TEXT NOT NULL, allowed_target_types_json TEXT NOT NULL,
              validators_json TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS semantic_dimensions (
              id TEXT NOT NULL, kind TEXT NOT NULL, label TEXT NOT NULL, description TEXT NOT NULL,
              namespace TEXT NOT NULL, configuration_json TEXT NOT NULL,
              status TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(id,kind)
            );
            CREATE TABLE IF NOT EXISTS schema_discoveries (
              id TEXT PRIMARY KEY, source_layer_ids_json TEXT NOT NULL,
              sample_node_ids_json TEXT NOT NULL, namespace TEXT NOT NULL,
              dataset_summary TEXT NOT NULL, candidates_json TEXT NOT NULL,
              comparisons_json TEXT NOT NULL, cleaning_guidance_json TEXT NOT NULL,
              status TEXT NOT NULL, generation_provider TEXT NOT NULL,
              readiness_json TEXT NOT NULL DEFAULT '{}', survey_round INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL, decided_at TEXT
            );
            CREATE TABLE IF NOT EXISTS placement_plans (
              id TEXT PRIMARY KEY, source_layer_ids_json TEXT NOT NULL,
              source_node_ids_json TEXT NOT NULL, material_origin TEXT NOT NULL,
              plan_json TEXT NOT NULL, status TEXT NOT NULL,
              generation_provider TEXT NOT NULL, created_at TEXT NOT NULL, decided_at TEXT
            );
            CREATE TABLE IF NOT EXISTS layers (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
              origin_type TEXT NOT NULL, is_initial INTEGER NOT NULL DEFAULT 0,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS nodes (
              id TEXT PRIMARY KEY, layer_id TEXT NOT NULL REFERENCES layers(id) ON DELETE CASCADE,
              title TEXT NOT NULL, content TEXT NOT NULL, node_type TEXT NOT NULL,
              attributes_json TEXT NOT NULL DEFAULT '{}', provenance_json TEXT NOT NULL,
              maturity TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mappings (
              id TEXT PRIMARY KEY,
              source_node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
              target_node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
              relation_type TEXT NOT NULL, confidence REAL NOT NULL, evidence TEXT NOT NULL,
              status TEXT NOT NULL, attributes_json TEXT NOT NULL DEFAULT '{}',
              valid_from TEXT, valid_to TEXT, created_at TEXT NOT NULL,
              UNIQUE(source_node_id, target_node_id, relation_type)
            );
            CREATE TABLE IF NOT EXISTS shortcuts (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
              trigger_examples_json TEXT NOT NULL, plan_json TEXT NOT NULL,
              preconditions_json TEXT NOT NULL, validators_json TEXT NOT NULL,
              failure_conditions_json TEXT NOT NULL, route_signature TEXT NOT NULL,
              status TEXT NOT NULL, confirmations INTEGER NOT NULL DEFAULT 1,
              use_count INTEGER NOT NULL DEFAULT 0, success_count INTEGER NOT NULL DEFAULT 0,
              total_latency_ms REAL NOT NULL DEFAULT 0, version INTEGER NOT NULL DEFAULT 1,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS shortcut_runs (
              id TEXT PRIMARY KEY, shortcut_id TEXT NOT NULL REFERENCES shortcuts(id) ON DELETE CASCADE,
              question TEXT NOT NULL, similarity REAL NOT NULL, success INTEGER NOT NULL,
              latency_ms REAL NOT NULL, details_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS query_runs (
              id TEXT PRIMARY KEY, question TEXT NOT NULL, mode TEXT NOT NULL,
              shortcut_id TEXT REFERENCES shortcuts(id) ON DELETE SET NULL,
              requested_depth INTEGER NOT NULL, reached_depth INTEGER NOT NULL,
              evidence_node_ids_json TEXT NOT NULL, answer_json TEXT,
              latency_ms REAL NOT NULL, success INTEGER NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS vectors (
              item_id TEXT NOT NULL, namespace TEXT NOT NULL, model TEXT NOT NULL,
              dimension INTEGER NOT NULL, vector_json TEXT NOT NULL, payload_json TEXT NOT NULL,
              updated_at TEXT NOT NULL, PRIMARY KEY(item_id, namespace, model)
            );
            """)
            mapping_columns = {row[1] for row in db.execute("PRAGMA table_info(mappings)").fetchall()}
            if "attributes_json" not in mapping_columns:
                db.execute("ALTER TABLE mappings ADD COLUMN attributes_json TEXT NOT NULL DEFAULT '{}'")
            if "valid_from" not in mapping_columns:
                db.execute("ALTER TABLE mappings ADD COLUMN valid_from TEXT")
            if "valid_to" not in mapping_columns:
                db.execute("ALTER TABLE mappings ADD COLUMN valid_to TEXT")
            node_columns = {row[1] for row in db.execute("PRAGMA table_info(nodes)").fetchall()}
            if "attributes_json" not in node_columns:
                db.execute("ALTER TABLE nodes ADD COLUMN attributes_json TEXT NOT NULL DEFAULT '{}'")
            layer_columns = {row[1] for row in db.execute("PRAGMA table_info(layers)").fetchall()}
            if "is_initial" not in layer_columns:
                db.execute("ALTER TABLE layers ADD COLUMN is_initial INTEGER NOT NULL DEFAULT 0")
            if not db.execute("SELECT 1 FROM layers WHERE is_initial=1 LIMIT 1").fetchone():
                earliest = db.execute("""SELECT id FROM layers
                  ORDER BY CASE WHEN origin_type='input' THEN 0 ELSE 1 END, created_at LIMIT 1""").fetchone()
                if earliest:
                    db.execute("UPDATE layers SET is_initial=1 WHERE id=?", (earliest[0],))
            db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS one_initial_layer
              ON layers(is_initial) WHERE is_initial=1""")
            discovery_columns = {row[1] for row in db.execute(
                "PRAGMA table_info(schema_discoveries)").fetchall()}
            if "readiness_json" not in discovery_columns:
                db.execute("ALTER TABLE schema_discoveries ADD COLUMN readiness_json TEXT NOT NULL DEFAULT '{}'")
            if "survey_round" not in discovery_columns:
                db.execute("ALTER TABLE schema_discoveries ADD COLUMN survey_round INTEGER NOT NULL DEFAULT 1")
            for item in DEFAULT_LAYER_TYPES:
                db.execute("INSERT OR IGNORE INTO layer_types VALUES(?,?,?,?,?,?)", (
                    item.id, item.label, item.description, item.namespace, "active", now()))
            for item in DEFAULT_RELATION_TYPES:
                db.execute("INSERT OR IGNORE INTO relation_types VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    item.id, item.label, item.description, item.namespace, item.inverse_type,
                    int(item.directional), int(item.symmetric), int(item.transitive), int(item.temporal),
                    item.default_traversal_weight, json.dumps(item.allowed_source_types),
                    json.dumps(item.allowed_target_types), json.dumps(item.validators), "active", now()))
            for item in DEFAULT_SEMANTIC_DIMENSIONS:
                db.execute("INSERT OR IGNORE INTO semantic_dimensions VALUES(?,?,?,?,?,?,?,?)", (
                    item.id, item.kind, item.label, item.description, "core",
                    item.model_dump_json(), "active", now()))
            for row in db.execute("SELECT DISTINCT node_type FROM nodes").fetchall():
                legacy_id = row[0]
                configuration = {
                    "kind": "node_type", "id": legacy_id, "label": legacy_id,
                    "description": "", "rationale": "Migrated pre-registry node type",
                }
                db.execute("INSERT OR IGNORE INTO semantic_dimensions VALUES(?,?,?,?,?,?,?,?)", (
                    legacy_id, "node_type", legacy_id, "", "legacy",
                    json.dumps(configuration, ensure_ascii=False), "active", now()))
            for row in db.execute("SELECT DISTINCT origin_type FROM layers").fetchall():
                db.execute("INSERT OR IGNORE INTO layer_types VALUES(?,?,?,?,?,?)", (
                    row[0], row[0], "Migrated pre-registry layer type", "legacy", "active", now()))
            for row in db.execute("SELECT DISTINCT relation_type FROM mappings").fetchall():
                db.execute("INSERT OR IGNORE INTO relation_types VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    row[0], row[0], "Migrated pre-registry relation type", "legacy", None,
                    1, 0, 0, 0, 0.65, "[]", "[]", "[]", "active", now()))

    def rows(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self.connection() as db:
            return [dict(row) for row in db.execute(sql, tuple(params)).fetchall()]

    def create_layer(self, item: LayerCreate) -> str:
        if not self.rows("SELECT id FROM layer_types WHERE id=? AND status='active'", (item.origin_type,)):
            raise ValueError(f"Unknown layer type: {item.origin_type}. Register it before use.")
        existing_initial = bool(self.rows("SELECT id FROM layers WHERE is_initial=1 LIMIT 1"))
        any_layer = bool(self.rows("SELECT id FROM layers LIMIT 1"))
        is_initial = item.is_initial or (not any_layer and item.origin_type == "input")
        if is_initial and item.origin_type != "input":
            raise ValueError("The historical initial layer must use the core input type")
        if is_initial and existing_initial:
            raise ValueError("This workspace already has a historical initial layer")
        layer_id = str(uuid.uuid4())
        with self.connection() as db:
            db.execute("""INSERT INTO layers
              (id,name,description,origin_type,is_initial,created_at) VALUES(?,?,?,?,?,?)""", (
                layer_id, item.name, item.description, item.origin_type, int(is_initial), now()))
        return layer_id

    def create_node(self, item: NodeCreate, node_id: str | None = None) -> str:
        if not self.rows("""SELECT id FROM semantic_dimensions
          WHERE id=? AND kind='node_type' AND status='active'""", (item.node_type,)):
            raise ValueError(f"Unknown node type: {item.node_type}. Register or approve it before use.")
        if item.attributes:
            marks = ",".join("?" for _ in item.attributes)
            known = {row["id"] for row in self.rows(
                f"""SELECT id FROM semantic_dimensions WHERE kind='attribute'
                AND status='active' AND id IN ({marks})""", list(item.attributes))}
            unknown = sorted(set(item.attributes) - known)
            if unknown:
                raise ValueError(f"Unknown node attributes: {', '.join(unknown)}")
        node_id = node_id or str(uuid.uuid4())
        with self.connection() as db:
            db.execute("""INSERT INTO nodes
              (id,layer_id,title,content,node_type,attributes_json,provenance_json,maturity,created_at)
              VALUES(?,?,?,?,?,?,?,?,?)""", (
                node_id, item.layer_id, item.title, item.content, item.node_type,
                json.dumps(item.attributes, ensure_ascii=False),
                json.dumps(item.provenance, ensure_ascii=False), item.maturity, now()))
        return node_id

    def create_mapping(self, item: MappingCreate) -> str:
        relation = self.relation_type(item.relation_type)
        if not relation or relation["status"] != "active":
            raise ValueError(f"Unknown relation type: {item.relation_type}. Register it before use.")
        endpoints = self.rows("""SELECT n.id,l.origin_type FROM nodes n JOIN layers l ON l.id=n.layer_id
          WHERE n.id IN (?,?)""", (item.source_node_id, item.target_node_id))
        endpoint_types = {row["id"]: row["origin_type"] for row in endpoints}
        if len(endpoint_types) != 2:
            raise ValueError("Mapping endpoints must both exist")
        if relation["allowed_source_types"] and endpoint_types[item.source_node_id] not in relation["allowed_source_types"]:
            raise ValueError(f"Relation {item.relation_type} does not allow this source layer type")
        if relation["allowed_target_types"] and endpoint_types[item.target_node_id] not in relation["allowed_target_types"]:
            raise ValueError(f"Relation {item.relation_type} does not allow this target layer type")
        if "distinct_endpoints" in relation["validators"] and item.source_node_id == item.target_node_id:
            raise ValueError(f"Relation {item.relation_type} requires distinct endpoints")
        mapping_id = str(uuid.uuid4())
        with self.connection() as db:
            db.execute(
                """INSERT OR IGNORE INTO mappings
                (id,source_node_id,target_node_id,relation_type,confidence,evidence,status,
                 attributes_json,valid_from,valid_to,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (mapping_id, item.source_node_id, item.target_node_id, item.relation_type,
                 item.confidence, item.evidence, item.status,
                 json.dumps(item.attributes, ensure_ascii=False), item.valid_from, item.valid_to, now()),
            )
        return mapping_id

    def register_layer_type(self, item: LayerTypeCreate) -> str:
        with self.connection() as db:
            db.execute("""INSERT OR REPLACE INTO layer_types
              (id,label,description,namespace,status,created_at) VALUES(?,?,?,?,?,?)""",
              (item.id, item.label, item.description, item.namespace, "active", now()))
        return item.id

    def register_relation_type(self, item: RelationTypeCreate) -> str:
        referenced_layer_types = set(item.allowed_source_types) | set(item.allowed_target_types)
        if referenced_layer_types:
            marks = ",".join("?" for _ in referenced_layer_types)
            known = {row["id"] for row in self.rows(
                f"SELECT id FROM layer_types WHERE status='active' AND id IN ({marks})",
                sorted(referenced_layer_types),
            )}
            unknown = sorted(referenced_layer_types - known)
            if unknown:
                raise ValueError(f"Unknown endpoint layer types: {', '.join(unknown)}")
        with self.connection() as db:
            db.execute("""INSERT OR REPLACE INTO relation_types VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                item.id, item.label, item.description, item.namespace, item.inverse_type,
                int(item.directional), int(item.symmetric), int(item.transitive), int(item.temporal),
                item.default_traversal_weight, json.dumps(item.allowed_source_types),
                json.dumps(item.allowed_target_types), json.dumps(item.validators), "active", now()))
        return item.id

    def register_semantic_dimension(self, item: SchemaCandidate, namespace: str) -> str:
        if item.kind not in {"node_type", "attribute"}:
            raise ValueError("Only node_type and attribute candidates use the semantic dimension registry")
        with self.connection() as db:
            db.execute("""INSERT OR REPLACE INTO semantic_dimensions
              (id,kind,label,description,namespace,configuration_json,status,created_at)
              VALUES(?,?,?,?,?,?,?,?)""", (
                item.id, item.kind, item.label, item.description, namespace,
                item.model_dump_json(), "active", now(),
            ))
        return item.id

    def import_ontology(self, bundle: OntologyBundle) -> dict[str, Any]:
        for item in bundle.layer_types:
            self.register_layer_type(item)
        for item in bundle.semantic_dimensions:
            self.register_semantic_dimension(item, item.id.split(":", 1)[0] if ":" in item.id else "workspace")
        for item in bundle.relation_types:
            self.register_relation_type(item)
        return {
            "name": bundle.name, "layer_types": len(bundle.layer_types),
            "relation_types": len(bundle.relation_types),
            "semantic_dimensions": len(bundle.semantic_dimensions),
        }

    def relation_type(self, type_id: str) -> dict[str, Any] | None:
        found = self.rows("SELECT * FROM relation_types WHERE id=?", (type_id,))
        if not found:
            return None
        item = found[0]
        for key in ("allowed_source_types", "allowed_target_types", "validators"):
            item[key] = json.loads(item.pop(f"{key}_json"))
        for key in ("directional", "symmetric", "transitive", "temporal"):
            item[key] = bool(item[key])
        return item

    def relation_weight(self, type_id: str) -> float:
        item = self.relation_type(type_id)
        return float(item["default_traversal_weight"]) if item else 0.0

    def ontology_snapshot(self) -> dict[str, Any]:
        relations = []
        for row in self.rows("SELECT * FROM relation_types WHERE status='active' ORDER BY id"):
            relations.append(self.relation_type(row["id"]))
        return {
            "layer_types": self.rows("SELECT * FROM layer_types WHERE status='active' ORDER BY id"),
            "relation_types": relations,
            "semantic_dimensions": [self._decode_dimension(row) for row in self.rows(
                "SELECT * FROM semantic_dimensions WHERE status='active' ORDER BY kind,id")],
        }

    @staticmethod
    def _decode_dimension(row: dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        item["configuration"] = json.loads(item.pop("configuration_json"))
        return item

    def create_schema_discovery(self, *, source_layer_ids: list[str], sample_node_ids: list[str],
                                namespace: str, dataset_summary: str,
                                candidates: list[dict[str, Any]], comparisons: list[dict[str, Any]],
                                cleaning_guidance: list[str], generation_provider: str,
                                readiness: dict[str, Any], survey_round: int) -> str:
        discovery_id = str(uuid.uuid4())
        with self.connection() as db:
            db.execute("""INSERT INTO schema_discoveries
              (id,source_layer_ids_json,sample_node_ids_json,namespace,dataset_summary,
               candidates_json,comparisons_json,cleaning_guidance_json,status,generation_provider,
               readiness_json,survey_round,created_at,decided_at)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                discovery_id, json.dumps(sorted(source_layer_ids)), json.dumps(sample_node_ids), namespace,
                dataset_summary, json.dumps(candidates, ensure_ascii=False),
                json.dumps(comparisons, ensure_ascii=False),
                json.dumps(cleaning_guidance, ensure_ascii=False), "pending",
                generation_provider, json.dumps(readiness), survey_round, now(), None,
            ))
        return discovery_id

    @staticmethod
    def _decode_schema_discovery(row: dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        for key in ("source_layer_ids", "sample_node_ids", "candidates", "comparisons",
                    "cleaning_guidance", "readiness"):
            item[key] = json.loads(item.pop(f"{key}_json"))
        return item

    def next_schema_survey_round(self, source_layer_ids: list[str], namespace: str) -> int:
        encoded = json.dumps(sorted(source_layer_ids))
        rows = self.rows("""SELECT MAX(survey_round) AS maximum FROM schema_discoveries
          WHERE source_layer_ids_json=? AND namespace=?""", (encoded, namespace))
        return int(rows[0]["maximum"] or 0) + 1

    def schema_candidate_support(self, discovery: dict[str, Any], candidate_key: str) -> int:
        support = 0
        for other in self.schema_discoveries():
            if (set(other["source_layer_ids"]) != set(discovery["source_layer_ids"])
                    or other["namespace"] != discovery["namespace"]
                    or other["status"] == "rejected"):
                continue
            keys = {f"{item['kind']}:{item['id']}" for item in other["candidates"]}
            support += int(candidate_key in keys)
        return support

    def schema_discovery(self, discovery_id: str) -> dict[str, Any] | None:
        found = self.rows("SELECT * FROM schema_discoveries WHERE id=?", (discovery_id,))
        return self._decode_schema_discovery(found[0]) if found else None

    def schema_discoveries(self, status: str | None = None) -> list[dict[str, Any]]:
        rows = self.rows(
            "SELECT * FROM schema_discoveries WHERE status=? ORDER BY created_at DESC", (status,)
        ) if status else self.rows("SELECT * FROM schema_discoveries ORDER BY created_at DESC")
        return [self._decode_schema_discovery(row) for row in rows]

    def approve_schema_discovery(self, discovery_id: str,
                                 candidate_keys: list[str] | None = None) -> dict[str, Any]:
        discovery = self.schema_discovery(discovery_id)
        if not discovery:
            raise ValueError("Schema discovery not found")
        if discovery["status"] != "pending":
            raise ValueError(f"Schema discovery is already {discovery['status']}")
        all_candidates = [SchemaCandidate.model_validate(item) for item in discovery["candidates"]]
        by_key = {f"{item.kind}:{item.id}": item for item in all_candidates}
        comparison_by_key = {item["candidate_key"]: item for item in discovery["comparisons"]}
        if candidate_keys is None:
            selected = {key for key, item in comparison_by_key.items()
                        if item["recommendation"] == "add"}
        else:
            selected = set(candidate_keys)
            unknown = sorted(selected - set(by_key))
            if unknown:
                raise ValueError(f"Unknown schema candidate keys: {', '.join(unknown)}")
        selected = {key for key in selected
                    if comparison_by_key.get(key, {}).get("recommendation") != "reuse_existing"}
        required_surveys = int(discovery.get("readiness", {}).get("thresholds", {}).get(
            "required_surveys", 1))
        unstable = []
        for key in sorted(selected):
            support = self.schema_candidate_support(discovery, key)
            if support < required_surveys:
                unstable.append(f"{key} ({support}/{required_surveys} surveys)")
        if unstable:
            raise ValueError(
                "Schema candidates have not repeated across enough independent surveys: "
                + ", ".join(unstable))
        candidates = [item for key, item in by_key.items() if key in selected]
        selected_layer_types = {item.id for item in candidates if item.kind == "layer_type"}
        existing_layer_types = {row["id"] for row in self.rows(
            "SELECT id FROM layer_types WHERE status='active'")}
        for item in candidates:
            if item.kind != "relation_type":
                continue
            endpoints = set(item.allowed_source_types) | set(item.allowed_target_types)
            missing = sorted(endpoints - existing_layer_types - selected_layer_types)
            if missing:
                raise ValueError(
                    f"Approve or register endpoint layer types before relation {item.id}: {', '.join(missing)}")
        approved: list[str] = []
        for item in candidates:
            if item.kind == "layer_type":
                self.register_layer_type(LayerTypeCreate(
                    id=item.id, label=item.label, description=item.description,
                    namespace=discovery["namespace"]))
                approved.append(f"layer_type:{item.id}")
        for item in candidates:
            if item.kind == "relation_type":
                self.register_relation_type(RelationTypeCreate(
                    id=item.id, label=item.label, description=item.description,
                    namespace=discovery["namespace"], inverse_type=item.inverse_type,
                    directional=item.directional, symmetric=item.symmetric,
                    transitive=item.transitive, temporal=item.temporal,
                    default_traversal_weight=item.default_traversal_weight,
                    allowed_source_types=item.allowed_source_types,
                    allowed_target_types=item.allowed_target_types,
                    validators=item.validators))
                approved.append(f"relation_type:{item.id}")
            elif item.kind in {"node_type", "attribute"}:
                self.register_semantic_dimension(item, discovery["namespace"])
                approved.append(f"{item.kind}:{item.id}")
        with self.connection() as db:
            db.execute("UPDATE schema_discoveries SET status='approved',decided_at=? WHERE id=?",
                       (now(), discovery_id))
        return {"id": discovery_id, "status": "approved", "approved_candidates": approved}

    def reject_schema_discovery(self, discovery_id: str) -> dict[str, Any]:
        discovery = self.schema_discovery(discovery_id)
        if not discovery:
            raise ValueError("Schema discovery not found")
        if discovery["status"] != "pending":
            raise ValueError(f"Schema discovery is already {discovery['status']}")
        with self.connection() as db:
            db.execute("UPDATE schema_discoveries SET status='rejected',decided_at=? WHERE id=?",
                       (now(), discovery_id))
        return {"id": discovery_id, "status": "rejected"}

    def create_placement_plan(self, *, source_layer_ids: list[str], source_node_ids: list[str],
                              material_origin: str, plan: dict[str, Any],
                              generation_provider: str) -> str:
        plan_id = str(uuid.uuid4())
        with self.connection() as db:
            db.execute("""INSERT INTO placement_plans
              (id,source_layer_ids_json,source_node_ids_json,material_origin,plan_json,status,
               generation_provider,created_at,decided_at) VALUES(?,?,?,?,?,?,?,?,?)""", (
                plan_id, json.dumps(sorted(source_layer_ids)), json.dumps(sorted(source_node_ids)),
                material_origin, json.dumps(plan, ensure_ascii=False), "pending",
                generation_provider, now(), None))
        return plan_id

    @staticmethod
    def _decode_placement_plan(row: dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        for key in ("source_layer_ids", "source_node_ids", "plan"):
            item[key] = json.loads(item.pop(f"{key}_json"))
        return item

    def placement_plan(self, plan_id: str) -> dict[str, Any] | None:
        rows = self.rows("SELECT * FROM placement_plans WHERE id=?", (plan_id,))
        return self._decode_placement_plan(rows[0]) if rows else None

    def placement_plans(self, status: str | None = None) -> list[dict[str, Any]]:
        rows = self.rows("SELECT * FROM placement_plans WHERE status=? ORDER BY created_at DESC", (status,)) \
            if status else self.rows("SELECT * FROM placement_plans ORDER BY created_at DESC")
        return [self._decode_placement_plan(row) for row in rows]

    def decide_placement_plan(self, plan_id: str, status: str) -> dict[str, Any]:
        if status not in {"approved", "rejected"}:
            raise ValueError("Placement plan status must be approved or rejected")
        plan = self.placement_plan(plan_id)
        if not plan:
            raise ValueError("Placement plan not found")
        if plan["status"] != "pending":
            raise ValueError(f"Placement plan is already {plan['status']}")
        with self.connection() as db:
            db.execute("UPDATE placement_plans SET status=?,decided_at=? WHERE id=?",
                       (status, now(), plan_id))
        return {"id": plan_id, "status": status}

    def get_nodes(self, node_ids: list[str]) -> list[dict[str, Any]]:
        if not node_ids:
            return []
        marks = ",".join("?" for _ in node_ids)
        return self.rows(
            f"""SELECT n.*,l.name layer_name,l.origin_type FROM nodes n
            JOIN layers l ON l.id=n.layer_id WHERE n.id IN ({marks})""", node_ids)

    def mappings_from(self, node_ids: list[str], relation_types: list[str] | None = None) -> list[dict[str, Any]]:
        if not node_ids:
            return []
        marks = ",".join("?" for _ in node_ids)
        params: list[Any] = [*node_ids, *node_ids]
        relation_clause = ""
        if relation_types:
            relation_marks = ",".join("?" for _ in relation_types)
            relation_clause = f" AND relation_type IN ({relation_marks})"
            params.extend(relation_types)
        return self.rows(
            f"""SELECT * FROM mappings WHERE status!='rejected' AND
            (source_node_id IN ({marks}) OR target_node_id IN ({marks})){relation_clause}""", params)

    def create_shortcut(self, item: ShortcutCreate, route_signature: str) -> str:
        shortcut_id = str(uuid.uuid4())
        stamp = now()
        with self.connection() as db:
            db.execute("INSERT INTO shortcuts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                shortcut_id, item.name, item.description,
                json.dumps(item.trigger_examples, ensure_ascii=False),
                item.plan.model_dump_json(),
                json.dumps(item.preconditions, ensure_ascii=False),
                json.dumps(item.validators, ensure_ascii=False),
                json.dumps(item.failure_conditions, ensure_ascii=False),
                route_signature, item.status, 1, 0, 0, 0.0, 1, stamp, stamp,
            ))
        return shortcut_id

    def reinforce_shortcut_candidate(self, shortcut_id: str, activation_count: int = 3) -> None:
        with self.connection() as db:
            db.execute("""UPDATE shortcuts SET confirmations=confirmations+1,
              status=CASE WHEN confirmations+1>=? THEN 'active' ELSE status END,
              updated_at=? WHERE id=?""", (activation_count, now(), shortcut_id))

    def decode_shortcut(self, row: dict[str, Any]) -> dict[str, Any]:
        item = dict(row)
        for key in ("trigger_examples", "plan", "preconditions", "validators", "failure_conditions"):
            item[key] = json.loads(item.pop(f"{key}_json"))
        reliability = (item["success_count"] + 1) / (item["use_count"] + 2)
        item["reliability"] = reliability
        return item

    def shortcut(self, shortcut_id: str) -> dict[str, Any] | None:
        found = self.rows("SELECT * FROM shortcuts WHERE id=?", (shortcut_id,))
        return self.decode_shortcut(found[0]) if found else None

    def active_shortcuts(self) -> list[dict[str, Any]]:
        return [self.decode_shortcut(row) for row in self.rows(
            "SELECT * FROM shortcuts WHERE status='active'")]

    def candidate_by_signature(self, signature: str) -> dict[str, Any] | None:
        found = self.rows(
            "SELECT * FROM shortcuts WHERE route_signature=? AND status='candidate' ORDER BY created_at LIMIT 1",
            (signature,),
        )
        return self.decode_shortcut(found[0]) if found else None

    def record_shortcut_run(self, shortcut_id: str, question: str, similarity: float,
                            success: bool, latency_ms: float, details: dict[str, Any]) -> None:
        with self.connection() as db:
            db.execute("INSERT INTO shortcut_runs VALUES(?,?,?,?,?,?,?,?)", (
                str(uuid.uuid4()), shortcut_id, question, similarity, int(success), latency_ms,
                json.dumps(details, ensure_ascii=False), now()))
            db.execute("""UPDATE shortcuts SET use_count=use_count+1,
              success_count=success_count+?, total_latency_ms=total_latency_ms+?, updated_at=? WHERE id=?""",
              (int(success), latency_ms, now(), shortcut_id))

    def record_query(self, *, question: str, mode: str, shortcut_id: str | None,
                     requested_depth: int, reached_depth: int, evidence_ids: list[str],
                     answer: dict[str, Any] | None, latency_ms: float, success: bool) -> str:
        run_id = str(uuid.uuid4())
        with self.connection() as db:
            db.execute("INSERT INTO query_runs VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                run_id, question, mode, shortcut_id, requested_depth, reached_depth,
                json.dumps(evidence_ids), json.dumps(answer, ensure_ascii=False) if answer else None,
                latency_ms, int(success), now()))
        return run_id

    def export_all(self) -> dict[str, Any]:
        return {
            "format": "essay-understanding-memory-v1",
            "exported_at": now(),
            "layers": self.rows("SELECT * FROM layers"),
            "nodes": self.rows("SELECT * FROM nodes"),
            "mappings": self.rows("SELECT * FROM mappings"),
            "shortcuts": self.rows("SELECT * FROM shortcuts"),
            "ontology": self.ontology_snapshot(),
            "schema_discoveries": self.schema_discoveries(),
            "placement_plans": self.placement_plans(),
        }
