# Abstraction readiness, historical origin, and material placement

## Historical origin is not a permanent semantic root

The first admitted `input` layer is marked `is_initial`. This is a chronological and provenance fact: it records
where the workspace began. It does **not** grant that layer permanent semantic authority, force every later layer to
derive from it, or give it automatic retrieval priority. Once admitted, all knowledge layers remain selectable peer
contexts connected by explicit mappings.

## When model-assisted abstraction is allowed

Schema discovery and direct machine abstraction are blocked until the selected source corpus satisfies:

$$
\operatorname{sizeReady}=N\ge 12\ \land\ (C\ge 24{,}000\ \lor\ N\ge 50),
$$

where $N$ is canonical source-node count and $C$ is total source characters. A proposed schema item can then be
activated only when the same namespaced candidate ID occurs in at least two differently sampled survey rounds:

$$
\operatorname{activationReady}(x)=\operatorname{sizeReady}\land
\operatorname{supportRounds}(x)\ge 2.
$$

The second survey rotates deterministic stratified offsets, so it does not simply reread the identical nodes. An
approval attempt before recurrence fails closed. `POST /process/abstraction-readiness` exposes the size decision;
discovery records preserve metrics, thresholds, source sample IDs, and survey round.

These values are **configurable Alpha engineering defaults**, not a theorem or universal sample-size claim:

```text
AEC_SCHEMA_MIN_NODES=12
AEC_SCHEMA_MIN_CHARS=24000
AEC_SCHEMA_SHORT_RECORD_NODES=50
AEC_SCHEMA_REQUIRED_SURVEYS=2
```

Operators should raise the thresholds and number of surveys for heterogeneous, high-risk, multilingual, or
long-lived corpora. The current recurrence check uses exact namespaced IDs; semantically equivalent renamings need
human reconciliation rather than silent merging.

## Self-derived and externally supplied knowledge

The LLM may propose a placement plan through `POST /process/plan-placement`, but it cannot create layers or alter
the ontology during that call. The application validates that every sampled source appears at least once (mixed
source units may legitimately route to more than one target), stores the
plan in `placement_plans`, and requires explicit approval or rejection.
An approved plan can be supplied to schema-guided cleaning; the validator then rejects output that targets a
different layer, merges incompatible decisions, creates units for `link_only`/`hold_for_review`, or omits material
that the plan requires routing.

| Material situation | Default placement |
|---|---|
| genuine continuation of the same source and lifecycle | append to the existing layer; initial layer allowed |
| different author, source, version, viewpoint, access boundary, or lifecycle | create a peer layer |
| model-generated abstraction or inference | create a derived layer; never append to the initial source |
| duplicate or citation without independent content | `link_only` |
| insufficient evidence | `hold_for_review` |

This keeps external commentary and machine abstraction distinct. Both may map to original material, while their
provenance, authority, and revision histories remain separate. If a target layer type does not exist, schema
discovery and approval must occur first.

## Research basis and limits

- Hennink and Kaiser report that narrow, homogeneous studies often reached saturation within 9–17 interviews,
  while broader studies needed more: <https://pubmed.ncbi.nlm.nih.gov/34785096/>.
- Code saturation can precede the deeper task of meaning saturation:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC9359070/>.
- A 2024 assessment found near saturation much earlier than true saturation and substantial variation by dataset:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC11267098/>.
- Simulations show that saturation depends on prevalence and subpopulation structure:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC5528901/>.
- Clustering work evaluates stability across resampled datasets rather than trusting one partition:
  <https://www.jmlr.org/papers/volume23/21-0052/21-0052.pdf>.
- Human-in-the-loop schema induction documents incomplete or unstable model schemas and the value of curation:
  <https://arxiv.org/abs/2302.13048>.

These sources motivate minimum evidence, recurrence, and human governance; they do not derive the exact
`12 / 24,000 / 50 / 2` defaults. Those values need recalibration with larger domain-specific experiments.
