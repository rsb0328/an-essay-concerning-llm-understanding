from __future__ import annotations

import argparse
import json

import uvicorn

from .documents import parser_for
from .models import QueryRequest
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
