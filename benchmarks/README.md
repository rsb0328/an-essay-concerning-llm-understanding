# Scaling benchmark

Install the project, then run a fresh process for each point:

```bash
python benchmarks/scaling.py --nodes 1000 --layers 5 --queries 30 --output result-1k.json
python benchmarks/scaling.py --nodes 10000 --layers 20 --queries 100 --output result-10k.json
```

The built-in script uses the dependency-free hashing embedder and SQLite vector store so anyone can verify the protocol. Serious scaling contributions should add equivalent runs with a production embedder and Qdrant, record hardware and configuration, and follow `research/SCALING_STUDY.md`.

Do not run multiple scale points against the same database. The script creates and removes an isolated temporary database for every invocation.

