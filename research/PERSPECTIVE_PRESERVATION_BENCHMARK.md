# Falsifiable benchmark for conflict, perspective, and revision preservation

The architecture is not supported merely because it can store several layers. Its distinctive claim must be tested
on outcomes that a strong flat baseline can also attempt. This protocol defines those outcomes before a large study
is run.

## Unit of evaluation

Each case contains canonical claim IDs, exact provenance, a perspective or authority ID, scope conditions, an
optional validity interval, and explicit gold relations such as contradiction, qualification, revision, or support.
Cases must include hard negatives: lexical contradiction without logical conflict, different conditions that make
two claims compatible, superseded claims, and two sources that happen to agree.

Queries cover five strata:

1. direct conflict under the same conditions;
2. apparent conflict resolved by scope or qualification;
3. temporal revision and an `as_of` question;
4. disagreement between independent authorities or perspectives;
5. directional relations whose inverse traversal would be invalid.

## Systems compared

Use identical documents, chunks, embedding model, generation model, prompts, and evidence/token budgets:

1. flat vector retrieval;
2. a strong flat baseline with the same provenance, perspective, time, and type metadata filters;
3. multi-layer retrieval without shortcuts;
4. multi-layer retrieval with a controlled shortcut warm-up.

The metadata-aware flat baseline is mandatory. Beating an intentionally weak vector-only baseline is not evidence
that layers are necessary.

## Primary metrics

- **conflict-pair recall**: fraction of cases where every gold incompatible claim is retrieved within the fixed
  evidence budget;
- **perspective attribution accuracy**: fraction of retrieved or cited claims assigned to the correct source and
  perspective;
- **false reconciliation rate**: fraction of genuinely incompatible cases that the answer incorrectly merges or
  presents as agreement;
- **scope-resolution accuracy**: fraction of apparent conflicts correctly recognized as conditionally compatible;
- **temporal validity accuracy**: fraction of `as_of` answers using only claims valid at the requested time;
- **revision-chain coverage**: fraction of required predecessor/successor claim IDs retrieved;
- **citation provenance accuracy**: fraction of answer citations resolving to the gold canonical source.

Report retrieval latency, end-to-end latency, P50/P95/P99, embedding calls, fetched vectors, traversed edges,
shortcut attempts, false-route rate, wasted shortcut milliseconds, and peak memory as secondary metrics.

## Scoring and falsification

Score retrieval before generation so a fluent model cannot conceal missing evidence. Use paired bootstrap confidence
intervals over cases and publish every case-level result. The multi-layer claim is unsupported if, under equal
budgets, it does not improve at least one preregistered primary preservation metric over the metadata-aware flat
baseline, or if any gain disappears when provenance and time filters are supplied to that baseline. Shortcut value
is unsupported if cumulative time saved after warm-up does not exceed learning, precheck, and false-route cost.

Negative and null results are valid outcomes. This benchmark is a protocol, not evidence that the current system
already wins it. A useful next contribution is a licensed, independently annotated dataset implementing these case
requirements at multiple scales.
