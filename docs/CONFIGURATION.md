# Configuration

All settings use the `AEC_` prefix. The application loads a local `.env` file when present; process-level environment variables take precedence.

Domain semantics are data, not environment configuration. Register a workspace ontology through
`POST /ontology/import` or `essay-understanding ontology-import PATH` before creating domain-specific layers and
mappings. See [Extensible domain ontologies](ONTOLOGIES.md).

## Generation

| Variable | Meaning |
|---|---|
| `AEC_LLM_BASE_URL` | OpenAI-compatible base URL ending in `/v1` |
| `AEC_LLM_MODEL` | Provider model identifier |
| `AEC_LLM_API_KEY` | Optional bearer token |

If URL or model is empty, generation is disabled. In that mode, imports, mappings, vector search, graph traversal, shortcut routing, shortcut learning, and evidence output remain available; model-assisted transformation, ontology-constrained classification, and prose answers do not.

Pre-cleaning schema discovery and schema-guided cleaning require a generation model because both inspect content
and return structured proposals or routed units. They are bounded batch operations; sampling and character limits
are request parameters rather than global settings.

The endpoint should support `/chat/completions` and ideally strict JSON Schema output. Some nominally compatible servers ignore `response_format`; adapter contributions may add provider-specific handling without moving task logic into the adapter.

## Embeddings

| Variable | Values |
|---|---|
| `AEC_EMBEDDING_PROVIDER` | `hashing`, `sentence_transformers`, `openai` |
| `AEC_EMBEDDING_MODEL` | Model name required by non-hashing providers |
| `AEC_EMBEDDING_DIMENSION` | Expected dimension; default 384 |
| `AEC_EMBEDDING_BASE_URL` | Optional separate embedding API base URL |
| `AEC_EMBEDDING_API_KEY` | Optional separate token |

The hashing provider is deterministic and dependency-free, but it is only a functional demonstration. Do not use its benchmark scores as evidence of semantic retrieval quality.

Never mix vectors from different embedding models in one index. The store namespaces collections by model; changing providers should be followed by a full index rebuild from canonical nodes and shortcut triggers.

## Vector storage

| Variable | Values |
|---|---|
| `AEC_VECTOR_STORE` | `sqlite`, `qdrant` |
| `AEC_QDRANT_URL` | Optional service URL |
| `AEC_QDRANT_PATH` | Optional embedded-storage directory |

SQLite vector search is exhaustive and intended for installation, tests, and small memories. Qdrant is optional for larger workloads. Qdrant data remains a derived index and is omitted from canonical exports.

## Data

`AEC_DATA_DIR` defaults to `./data`. It contains the canonical SQLite database and, when configured, embedded Qdrant data. The entire directory is ignored by Git.
