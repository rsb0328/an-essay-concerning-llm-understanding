from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import LayerCreate, MappingCreate, NodeCreate, ShortcutCreate


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
            CREATE TABLE IF NOT EXISTS layers (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
              origin_type TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS nodes (
              id TEXT PRIMARY KEY, layer_id TEXT NOT NULL REFERENCES layers(id) ON DELETE CASCADE,
              title TEXT NOT NULL, content TEXT NOT NULL, node_type TEXT NOT NULL,
              provenance_json TEXT NOT NULL, maturity TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mappings (
              id TEXT PRIMARY KEY,
              source_node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
              target_node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
              relation_type TEXT NOT NULL, confidence REAL NOT NULL, evidence TEXT NOT NULL,
              status TEXT NOT NULL, created_at TEXT NOT NULL,
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

    def rows(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self.connection() as db:
            return [dict(row) for row in db.execute(sql, tuple(params)).fetchall()]

    def create_layer(self, item: LayerCreate) -> str:
        layer_id = str(uuid.uuid4())
        with self.connection() as db:
            db.execute("INSERT INTO layers VALUES(?,?,?,?,?)", (
                layer_id, item.name, item.description, item.origin_type, now()))
        return layer_id

    def create_node(self, item: NodeCreate, node_id: str | None = None) -> str:
        node_id = node_id or str(uuid.uuid4())
        with self.connection() as db:
            db.execute("INSERT INTO nodes VALUES(?,?,?,?,?,?,?,?)", (
                node_id, item.layer_id, item.title, item.content, item.node_type,
                json.dumps(item.provenance, ensure_ascii=False), item.maturity, now()))
        return node_id

    def create_mapping(self, item: MappingCreate) -> str:
        mapping_id = str(uuid.uuid4())
        with self.connection() as db:
            db.execute(
                "INSERT OR IGNORE INTO mappings VALUES(?,?,?,?,?,?,?,?)",
                (mapping_id, item.source_node_id, item.target_node_id, item.relation_type,
                 item.confidence, item.evidence, item.status, now()),
            )
        return mapping_id

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
        }
