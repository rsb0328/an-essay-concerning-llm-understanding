# Data model

## Canonical objects

- **Layer**: a peer context such as source text, human interpretation, AI abstraction, critique, or procedural memory.
- **Node**: a provenance-bearing semantic unit belonging to one layer.
- **Mapping**: a typed, directed relation between two nodes, with confidence, evidence, and status.
- **Shortcut**: a procedural retrieval plan with triggers, limits, validators, failure conditions, maturity, reliability, and version.
- **Query run**: an auditable observation of mode, depth, evidence, shortcut use, latency, and output.

## Derived objects

Vectors and vector-store payloads are indexes. They may be deleted and rebuilt without changing canonical meaning. Every vector points back to a canonical node or shortcut ID.

## Shortcut lifecycle

```text
free exploration
  → candidate route
  → repeated confirmation
  → active shortcut
  → success/failure history
  → revision or retirement
```

The current reference threshold promotes a repeated route after three confirmations. This is an implementation default, not a universal cognitive claim.

## Portability

`GET /export` and `essay-understanding export` return layers, nodes, mappings, and shortcuts without vectors. The format identifier is `essay-understanding-memory-v1`.

