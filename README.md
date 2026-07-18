# An Essay Concerning LLM Understanding

*A small engineering essay on layered memory, interpretation, and learned routes through knowledge—its title being a deliberate nod to John Locke.*

[中文说明](README.zh-CN.md) · [Research data](research/) · [Scaling study](research/SCALING_STUDY.md) · [Configuration](docs/CONFIGURATION.md) · [Data model](docs/DATA_MODEL.md)

> **Alpha research software.** The architecture runs and its core invariants are tested, but the current evidence does not establish that layered retrieval is generally faster or more accurate than flat vector search. The repository is published to make that hypothesis testable, not to present it as settled.

## Contents

- [What this repository is](#what-this-repository-is)
- [Install and use it](#install-and-use-it)
- [A working theory of understanding](#a-working-theory-of-understanding)
- [How information moves through the system](#how-information-moves-through-the-system)
- [Architecture and replaceable dependencies](#architecture-and-replaceable-dependencies)
- [What we measured](#what-we-measured)
- [What the present benchmark does not settle](#what-the-present-benchmark-does-not-settle)
- [Open scaling study](#open-scaling-study)
- [Project status and roadmap](#project-status-and-roadmap)
- [Privacy, safety, and epistemic limits](#privacy-safety-and-epistemic-limits)
- [Acknowledgements and legal notices](#acknowledgements-and-legal-notices)

## What this repository is

Most retrieval-augmented systems flatten source passages, interpretations, abstractions, disagreements, and procedures into one searchable space. This project keeps them as peer knowledge layers connected by explicit mappings.

It is intended for questions where *how a claim was understood* matters alongside *which passage looks similar*:

- a source text and several competing interpretations;
- machine-derived abstractions and human-authored scholarship;
- claims that are semantically similar but logically contradictory;
- knowledge whose provenance and revision history must remain visible;
- repeated research tasks that may benefit from learning a reusable retrieval route.

The repository provides:

- a FastAPI backend and command-line interface;
- peer knowledge layers with no permanent root layer;
- provenance-preserving nodes and typed cross-layer mappings;
- bounded graph retrieval with depth, breadth, relation, cycle, and information-gain controls;
- a parallel procedural-memory layer of retrieval **shortcuts**;
- shortcut-first routing and candidate-shortcut learning after successful exploration;
- evidence-only operation when no generation model is configured;
- replaceable generation, embedding, vector-store, and document-parser boundaries;
- canonical JSON export independent of the vector index.

## Install and use it

### What is included

The repository contains the application, database schema, built-in SQLite vector index, a deterministic demo embedder, tests, synthetic research results, and benchmark protocol.

It contains **no LLM weights, embedding-model weights, Qdrant server, user documents, or prebuilt knowledge database**.

### Requirements

- Python 3.11 or newer.
- Optional: an OpenAI-compatible generation endpoint, local or online.
- Optional: Sentence Transformers and a model of your choice.
- Optional: Qdrant, either embedded through `qdrant-client` or as a service.
- Optional: Docling for document formats beyond plain text.

### Minimal installation

The minimal configuration needs no model and no external vector database. It uses a low-quality hashing embedder for demonstration and returns an evidence graph instead of a generated answer.

```bash
git clone https://github.com/rsb0328/an-essay-concerning-llm-understanding.git
cd an-essay-concerning-llm-understanding
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
essay-understanding serve
```

On macOS or Linux:

```bash
source .venv/bin/activate
pip install -e .
essay-understanding serve
```

Open `http://127.0.0.1:8765/docs` for the interactive API.

### Production-oriented optional components

```bash
pip install -e ".[sentence-transformers,qdrant,documents]"
cp .env.example .env
```

Choose your own providers:

```env
AEC_LLM_BASE_URL=http://127.0.0.1:1234/v1
AEC_LLM_MODEL=your-model
AEC_LLM_API_KEY=

AEC_EMBEDDING_PROVIDER=sentence_transformers
AEC_EMBEDDING_MODEL=your-embedding-model

AEC_VECTOR_STORE=qdrant
AEC_QDRANT_PATH=./data/qdrant
```

The application does not require Qwen, Ollama, BGE-M3, or a particular hosted API. A provider must reliably return structured JSON for model-assisted abstraction, relation classification, and grounded answering.

### A first evidence-only query

Create a layer, add text through `/ingest/text`, then query `/query`. The API examples in `/docs` expose the complete schemas. From the terminal:

```bash
essay-understanding ask "How does the text distinguish mapping from identity?" --depth 2
essay-understanding export > memory.json
```

See [configuration](docs/CONFIGURATION.md) for provider examples and [data model](docs/DATA_MODEL.md) for the persistent schema.

## A working theory of understanding

The project begins from a modest claim: learning often occurs by establishing a revisable correspondence between new material and structures already available to the learner.

That correspondence is not identity. A useful mapping preserves both resemblance and difference, including the conditions under which an analogy, interpretation, or inference ceases to hold. A contradiction is therefore not merely a retrieval failure. It may indicate a false mapping, an omitted condition, incompatible testimony, or a conceptual boundary that needs revision.

Several design consequences follow:

1. **There is no permanently privileged first layer.** A source can anchor a particular investigation without becoming the metaphysical root of all later understanding.
2. **Interpretations remain distinguishable.** Human scholarship, machine abstraction, source passages, and later critique may map to one another without being collapsed into one voice.
3. **Self-derived and externally taught knowledge are different events.** A machine-produced abstraction and a human-authored interpretation can occupy peer layers while retaining different provenance.
4. **Memory includes procedures as well as propositions.** Repeatedly discovering the same useful route should eventually create a reusable procedural memory rather than force the system to explore from zero.
5. **Understanding develops historically.** Tentative, reinforced, revised, and retired structures should remain auditable instead of being silently overwritten.

This is an implementable memory hypothesis, not a claim that vector mappings are a complete theory of human learning.

## How information moves through the system

The original ideas are implemented as ordinary application logic and data—not as a hidden prompt or a skill inside one particular LLM.

```mermaid
flowchart TD
    A["Source or interpretation"] --> B["Normalize, segment, preserve provenance"]
    B --> C["Create a peer knowledge layer"]
    C --> D["Build a replaceable vector index"]
    D --> E["Optional abstraction or cross-layer comparison"]
    E --> F["Validate and store explicit mappings"]
    Q["User question"] --> S["Search shortcut layer first"]
    S -->|"trusted match"| G["Guided bounded traversal"]
    S -->|"no match"| H["Free bounded exploration"]
    F --> G
    F --> H
    G --> I["Evidence graph"]
    H --> I
    I --> J["Evidence-only result or grounded generation"]
    J --> K["Record outcome and route"]
    K --> L["Candidate shortcut"]
    L -->|"repeated success"| S
```

### 1. Admission and provenance

Text enters as source material, human interpretation, machine abstraction, or another declared origin. Content hashes, locations, and user-supplied provenance remain attached to canonical nodes. Generated vectors are never treated as the sole copy of knowledge.

### 2. Peer-layer creation

Each admitted document or interpretation can become an independent layer. A query may select a temporary reference layer, but the data model has no permanent root. This allows a later critique to address a source, an abstraction, or another interpretation directly.

### 3. Normalization and segmentation

Text is normalized and split into overlapping passages while preserving passage boundaries and source pointers. Optional document parsers may extract richer structure, but the internal representation remains parser-independent.

### 4. Replaceable semantic indexing

An embedding provider creates search candidates. The model name and vector dimension are index metadata, not properties of the knowledge itself. Changing embedding models requires a new index, not a new memory: canonical text, mappings, and shortcut procedures remain available for re-embedding.

### 5. Two routes to a new layer

A human interpretation enters as externally taught knowledge with its own author and source. A model-generated abstraction enters as self-derived knowledge: propositions, definitions, premises, conclusions, qualifications, and tensions are proposed, validated, and linked back to the exact source nodes from which they were derived.

### 6. Candidate comparison

Vector similarity proposes which elements across selected layers are worth comparing. It does not decide that they agree. This stage reduces the comparison space while remaining deliberately agnostic about logical relation.

### 7. Explicit relation judgment

Candidate pairs may be classified as support, contradiction, qualification, interpretation, derivation, analogy, reference, or another typed relation. Negation, conditions, direction, confidence, evidence, and provenance are considered. Invalid node IDs and malformed model outputs fail closed rather than silently creating graph edges.

### 8. The shortcut layer is queried first

A shortcut is a peer procedural-memory object. It stores trigger examples, starting layers, preferred relation types, depth, breadth, stopping criteria, validators, failure conditions, history, and reliability. It does **not** store a canned answer.

Shortcut selection combines semantic trigger similarity with observed reliability and maturity. A high-confidence match constrains retrieval before graph expansion. A partial or untrusted match must fall back to free exploration.

### 9. Guided or free exploration

When a mature shortcut matches, retrieval begins where the learned route recommends. Otherwise, the system selects semantically relevant layers and seed nodes, then explores mappings. Both paths produce the same evidence-graph format and remain inspectable.

### 10. Association depth and stopping

The user sets a maximum association depth. Each traversal step may cross into another layer through an explicit mapping. Breadth limits the number of new nodes per step; relation filters limit which edges are legal; visited-node tracking prevents cycles; and low information gain can stop exploration before the maximum depth. Depth is therefore a research budget, not an instruction to wander indefinitely.

### 11. Evidence budgeting and grounded synthesis

The evidence graph is ranked and bounded before it reaches a generation model. With no model configured, the graph is the result. With a model configured, answers must cite existing evidence aliases, which are validated and translated back to canonical node IDs. Unsupported or unknown citations are rejected.

### 12. Route observation and shortcut learning

After a successful free exploration, the application summarizes the route as a **candidate** shortcut. Similar successful routes reinforce the candidate; the reference implementation promotes it after repeated confirmation. A mature route records later successes, failures, and latency, and can be retired when it stops helping.

This closes the intended loop:

```text
first encounter → explore → answer → retain the useful route
later similar encounter → retrieve route first → search selectively → update route history
```

### 13. Audit, export, and rebuilding

Queries record requested and reached depth, evidence IDs, shortcut use, latency, and outcome. Canonical layers, nodes, mappings, and shortcuts export as JSON. Vector stores are excluded from that export because they are derived indexes that should be reproducible from canonical memory.

## Architecture and replaceable dependencies

```text
Application-owned core
├── peer layer and provenance model
├── typed mapping model
├── graph traversal and depth controller
├── shortcut router, maturity, and history
├── evidence validation
└── canonical import/export

Replaceable adapters
├── GenerationProvider    (any compatible local or online model)
├── Embedder              (hashing demo, Sentence Transformers, API)
├── VectorStore           (built-in SQLite, Qdrant)
└── Document parser       (plain text, optional Docling)
```

SQLite is the default canonical store. The built-in SQLite vector index is useful for installation and small tests; Qdrant is the intended optional backend for serious vector workloads. Neither is allowed to own the conceptual data model.

## What we measured

These measurements come from the earlier local prototype, which used BGE-M3, local Qwen3 14B, SQLite, and embedded Qdrant. The ten questions were synthetic and covered ten different categories. Raw, non-private metrics are in [`research/results`](research/results/).

### Reliability observations

- Core unit tests in the local prototype: 12/12 passed.
- Early repeated full-chain observation: 16 successes in 17 attempts.
- The one failure was truncated answer JSON after the model reached its output cap; the run failed closed and later runs continued.
- Process crashes, timeouts, and cross-run data-pollution events observed: 0.
- Later architecture comparison: 30/30 mode queries completed.

These are observations, not proof of a zero or 5.9% population failure rate. Seventeen attempts are far too few for a tight reliability estimate.

The independent public package currently has 6/6 core tests passing. Its included 100-node scaling smoke test runs all three modes, but is explicitly too small to support an architecture claim.

### Flat, layered, and layered-plus-procedure comparison

Each of ten questions ran once in each mode, for 30 queries total.

| Metric | Flat single layer | Layered | Layered + procedural skill |
|---|---:|---:|---:|
| Success rate | 100% | 100% | 100% |
| Mean total latency | 5.908 s | 8.386 s | 6.757 s |
| P95 total latency | 9.274 s | 15.086 s | 12.503 s |
| Mean answer-generation latency | 5.835 s | 6.133 s | 4.330 s |
| Gold-answer embedding similarity | 0.805810 | 0.805606 | 0.806572 |
| Mean claim coverage | 0.642034 | 0.641771 | 0.650433 |
| Mean citation/evidence alignment | 0.770266 | 0.795861 | 0.844410 |
| Mean evidence nodes | 8.0 | 17.2 | 11.5 |
| Mean layer diversity | 1.0 | 4.2 | 3.4 |

The honest conclusion is limited:

- unrestricted layered retrieval was about 44% slower and did not measurably improve answer similarity or claim coverage;
- it did expose more layers and improved citation/evidence alignment modestly;
- the procedural skill reduced answer-generation time by about 25.8% and improved citation alignment, but end-to-end latency remained about 14.4% above the flat mean;
- the skill itself was cheap after vector reuse, but that prototype looked it up **after** full graph retrieval.

This open-source version changes the order to shortcut-first. It has unit tests for that ordering, but it has not yet been subjected to the same model-backed benchmark. The old results must not be presented as evidence that the new ordering is already faster.

## What the present benchmark does not settle

Ten synthetic questions on one workstation cannot determine whether the architecture wins at scale. It may remain slower for ordinary factual question answering. Its more plausible value may instead lie in dimensions a flat-answer benchmark only partially measures:

- keeping sources, interpretations, abstractions, and disagreements distinct;
- preserving contradictory evidence rather than averaging it away;
- tracking how a knowledge structure was revised;
- learning procedural routes across repeated related investigations;
- exposing a query's path for audit and correction;
- supporting long-term, personalized memory without declaring one permanent root.

These are hypotheses about useful memory organization. They do not demonstrate consciousness, subjective experience, or human-like understanding. The system may support experiments about memory, metacognition, self-revision, and procedural learning without settling the philosophical question of machine consciousness.

## Open scaling study

Hardware cost and local inference time prevented a large controlled study. We especially want to know how the three architectures behave as memory grows from thousands to millions of nodes, and whether a warmed-up shortcut layer eventually repays its learning and routing overhead.

The proposed study specifies logarithmic scale points, cold and warm runs, latency percentiles, throughput, recall, claim coverage, grounding, shortcut false-route rate, information depth, memory use, and hardware disclosure. See [the complete scaling protocol](research/SCALING_STUDY.md) and the [reproducible benchmark entry point](benchmarks/).

Contributions with negative or null results are welcome. A useful outcome is a curve showing *where the architecture stops helping*, not only a demonstration designed to make it win.

## Project status and roadmap

Implemented in this open-source package:

- model-free evidence mode;
- peer layers, nodes, provenance, and mappings;
- model-independent abstraction and relation protocols;
- depth/breadth/relation/cycle/information-gain traversal controls;
- shortcut-first routing;
- candidate shortcut creation, reinforcement, activation, and use history;
- built-in and optional provider adapters;
- canonical export and isolated tests.

Important next work:

- evaluate shortcut-first retrieval with the same controlled dataset;
- add candidate-shortcut editing, explicit rejection, retirement, and route version comparison;
- add model-assisted shortcut descriptions without giving the model authority to activate them;
- implement richer import/export and document parsing adapters;
- build a local user interface;
- add statistically useful real-world corpora and human evaluation;
- collect multi-scale community benchmarks.

## Privacy, safety, and epistemic limits

- Local databases, vector indexes, uploaded sources, model caches, and `.env` files are ignored by Git.
- Do not publish copyrighted source texts merely because their vectors were generated locally.
- Online generation or embedding providers receive the text sent to their APIs; local operation is required for material that must not leave the machine.
- A shortcut can reproduce a bad route. Candidate maturity, reliability, validators, failure conditions, and fallback exploration reduce this risk but do not eliminate it.
- Similarity is not truth, mapping is not identity, and graph depth is not understanding.

Security reports should follow [SECURITY.md](SECURITY.md).

## Acknowledgements and legal notices

The project benefited from the work of:

- [FastAPI](https://github.com/fastapi/fastapi) and [Pydantic](https://github.com/pydantic/pydantic) for the API and validation layer;
- [NumPy](https://github.com/numpy/numpy) for numerical operations;
- [Sentence Transformers](https://github.com/huggingface/sentence-transformers) for optional local embedding execution;
- [Qdrant](https://github.com/qdrant/qdrant) and its Python client for optional vector storage;
- [Docling](https://github.com/docling-project/docling) for optional document conversion;
- the creators of [BGE-M3](https://huggingface.co/BAAI/bge-m3) and [Qwen3](https://github.com/QwenLM/Qwen3), which were used in the local prototype experiments but are not bundled or required here.

This repository is licensed under Apache License 2.0. Third-party components and models retain their own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No model weights or third-party installers are redistributed.

## Citation

If you use the project or extend the scaling study, cite the repository using [`CITATION.cff`](CITATION.cff) and include your exact model, vector store, hardware, and configuration with reported results.
