# Schema discovery before semantic cleaning

Here “dimension” means a semantic storage dimension—layer type, node type, attribute, or relation type—not one of
the numeric coordinates emitted by an embedding model. Embedding dimensionality remains a provider-level index
choice.

The system can admit material into a neutral `input` layer before a domain schema is known. This admission step is
minimal normalization and provenance preservation, not final semantic cleaning.

The current engine also creates a provisional vector for each admitted raw node. That does not require knowing the
future semantic layers: it is a replaceable candidate index over the neutral input, while the schema survey itself
reads the sampled canonical content directly. Approved cleaning creates new canonical units and their final
layer-aware vectors; the provisional index may be rebuilt or discarded without losing the raw evidence.

## Why this stage exists

Requiring users to define every layer before inspecting unfamiliar data makes the architecture closed in practice.
Allowing a model to create types without review creates the opposite failure: synonym proliferation, unstable
schemas, and hallucinated relations. Schema discovery separates observation from activation.

```text
raw admission → immediate provisional raw-node embedding → initial layer is searchable
→ readiness gate → representative sample → LLM schema survey
→ compare with active ontology → rotating-sample recurrence gate → pending proposal → approve/reject
→ schema-guided cleaning and routing → embedding and mapping
```

The readiness gate does **not** gate raw admission, canonical storage, provisional embedding, or querying of the
historical initial layer. It gates only model-assisted schema discovery and abstraction into additional layers.

The survey distinguishes four kinds of semantic dimension:

- **layer type**: independently selected, governed, permissioned, or lifecycle-distinct context;
- **node type**: a stable kind of semantic unit with identity;
- **attribute**: a scalar or structured value describing a unit;
- **relation type**: a verifiable connection between units.

This prevents a deadline value from becoming a “deadline layer,” while still allowing projects and people to be
independently retrieved layers.

## Representative sampling

`POST /process/discover-schema` first applies a configurable evidence-size gate, then samples nodes at rotating
stratified offsets within each selected input layer. It limits
both node count and characters per node before one structured model call. This is a bounded survey, not proof that
rare structures have been found. A new candidate must recur by exact namespaced ID in at least two rotating-sample
rounds before approval. Those rounds share a model and protocol: recurrence is a variance guardrail, not an
independence assumption, significance test, or protection against shared model bias. Large or heterogeneous
datasets should use higher thresholds and more surveys. See
[Abstraction readiness and material placement](ABSTRACTION_READINESS.md).

## Comparison and approval

Each candidate is compared only with existing definitions of the same kind. Exact IDs are marked `existing`.
Otherwise label, description, and rationale embeddings are compared with active definitions. Similarity at or above
the reference threshold `0.82` is marked `possible_overlap`; lower similarity is marked `novel`.

The threshold is an advisory triage heuristic, not a mathematical proof of ontology identity. Discovery never
activates a type. Approval also rejects candidates that have not met the configured survey-recurrence requirement.
`POST /ontology/schema-discoveries/{id}/approve` activates selected candidates; an empty selection
means only candidates recommended as novel additions. Possible overlaps require explicit selection. Rejection keeps
an audit record without changing the ontology.

## Schema-guided cleaning

After approval, `POST /process/clean-with-schema` sends a bounded source-node batch, the cleaning guidance, and the
active ontology to the model. Output is validated before any writes:

- target layer types and node types must be active;
- attribute and relation IDs must be registered;
- every derived unit must cite a source node in the current batch;
- relation endpoints must refer to returned units.

Validated units are routed into peer layers, embedded through the configured provider, and linked to raw inputs with
`core:derived_from`. The raw input remains canonical evidence and is not deleted.

Before routing external or machine-derived material, `POST /process/plan-placement` can create an auditable proposal.
Different authors, sources, versions, viewpoints, access boundaries, or lifecycles default to peer layers; machine
abstraction defaults to a derived layer; only a genuine same-source continuation may append to the historical
initial layer. The LLM proposes while application validation and explicit approval govern.

## CLI

```bash
essay-understanding schema-discover company RAW_LAYER_ID --sample-limit 24
essay-understanding schema-approve DISCOVERY_ID
essay-understanding schema-clean DISCOVERY_ID
essay-understanding abstraction-readiness RAW_LAYER_ID
essay-understanding placement-plan external_source RAW_LAYER_ID
essay-understanding placement-approve PLACEMENT_PLAN_ID
```

For large inputs, pass explicit node IDs to repeated `schema-clean` calls. A generation provider is required for the
survey and cleaning stages; storage, manual ontology registration, retrieval, and evidence output remain usable
without one.

## Present limits

- Sampling can miss rare dimensions.
- Similarity does not determine whether two concepts are operationally equivalent.
- Approval is currently a whole discovery decision with an optional candidate subset, not a multi-user governance
  workflow.
- Mandatory per-proposal approval is not a scalable million-node governance strategy; risk-tiered batches and
  sampled audits remain future work.
- Approved schema changes do not automatically reprocess every historical node; re-cleaning is an explicit batch
  operation.
