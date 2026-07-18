from __future__ import annotations

import argparse
import json

import uvicorn

from .documents import parser_for
from .models import OntologyBundle, QueryRequest
from .runtime import engine


def main() -> None:
    parser = argparse.ArgumentParser(prog="essay-understanding")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Start the local API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    ask = sub.add_parser("ask", help="Query memory from the terminal")
    ask.add_argument("question")
    ask.add_argument("--depth", type=int, default=2)
    ingest = sub.add_parser("ingest-file", help="Parse and ingest a local document")
    ingest.add_argument("layer_id")
    ingest.add_argument("path")
    ontology = sub.add_parser("ontology-import", help="Import layer and relation types from JSON")
    ontology.add_argument("path")
    discover = sub.add_parser("schema-discover", help="Propose dimensions from representative raw input")
    discover.add_argument("namespace")
    discover.add_argument("layer_ids", nargs="+")
    discover.add_argument("--sample-limit", type=int, default=24)
    approve = sub.add_parser("schema-approve", help="Approve a pending schema discovery")
    approve.add_argument("discovery_id")
    approve.add_argument("candidate_keys", nargs="*")
    reject = sub.add_parser("schema-reject", help="Reject a pending schema discovery")
    reject.add_argument("discovery_id")
    clean = sub.add_parser("schema-clean", help="Clean and route a node batch with an approved discovery")
    clean.add_argument("discovery_id")
    clean.add_argument("node_ids", nargs="*")
    clean.add_argument("--max-nodes", type=int, default=24)
    sub.add_parser("status", help="Show configured providers")
    sub.add_parser("export", help="Export canonical memory as JSON")
    args = parser.parse_args()
    if args.command == "serve":
        uvicorn.run("essay_understanding.api:app", host=args.host, port=args.port)
    elif args.command == "ask":
        print(json.dumps(engine().query(QueryRequest(
            question=args.question, max_depth=args.depth)), ensure_ascii=False, indent=2))
    elif args.command == "ingest-file":
        from pathlib import Path
        path = Path(args.path).resolve()
        document = parser_for(path).parse(path)
        ids = engine().ingest_text(
            args.layer_id, document.title, document.text, provenance=document.provenance)
        print(json.dumps({"node_ids": ids, "count": len(ids)}, indent=2))
    elif args.command == "ontology-import":
        from pathlib import Path
        bundle = OntologyBundle.model_validate_json(Path(args.path).read_text(encoding="utf-8"))
        print(json.dumps(engine().repository.import_ontology(bundle), ensure_ascii=False, indent=2))
    elif args.command == "schema-discover":
        from .processing import KnowledgeProcessor
        print(json.dumps(KnowledgeProcessor(engine()).discover_schema(
            args.layer_ids, args.namespace, args.sample_limit), ensure_ascii=False, indent=2))
    elif args.command == "schema-approve":
        selected = args.candidate_keys or None
        print(json.dumps(engine().repository.approve_schema_discovery(
            args.discovery_id, selected), ensure_ascii=False, indent=2))
    elif args.command == "schema-reject":
        print(json.dumps(engine().repository.reject_schema_discovery(
            args.discovery_id), ensure_ascii=False, indent=2))
    elif args.command == "schema-clean":
        from .processing import KnowledgeProcessor
        selected = args.node_ids or None
        print(json.dumps(KnowledgeProcessor(engine()).clean_with_schema(
            args.discovery_id, selected, args.max_nodes), ensure_ascii=False, indent=2))
    elif args.command == "status":
        core = engine()
        print(json.dumps({
            "generation": core.generation.name,
            "embedding": core.embedder.name,
            "vector_store": type(core.vectors).__name__,
        }, indent=2))
    elif args.command == "export":
        print(json.dumps(engine().repository.export_all(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
