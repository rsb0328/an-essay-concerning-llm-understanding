# Extensible domain ontologies

The system deliberately does not treat `supports`, `contradicts`, or any other domain vocabulary as universal. It
provides a small structural core and lets each workspace register the layer and relation types its work needs.

Examples include:

- company memory: people, organizations, projects, decisions, risks, metrics, `reports_to`, `owns`, `blocks`;
- research: papers, methods, datasets, findings, `uses`, `replicates`, `challenges`;
- law: authorities, rules, matters, exceptions, `applies_to`, `overrules`, `distinguishes`;
- software operations: services, incidents, owners, deployments, `depends_on`, `caused_by`, `supersedes`;
- philosophy: passages, arguments, commentaries, `supports`, `contradicts`, `interprets`.

Philosophy is one optional ontology pack, not the system's hidden default.

## Open and governed

An unrestricted string field would be open but unreliable: spelling variants and hallucinated labels would
fragment the graph. Registration makes the system open **and governed**. A relation definition can declare a
namespaced ID, inverse, direction, symmetry, transitivity, temporality, traversal weight, allowed endpoint types,
and validators.

Mappings additionally carry confidence, evidence, open attributes, and optional `valid_from` / `valid_to` values.
The same mechanism can therefore represent a logical relation, an organizational reporting line, a time-bounded
ownership record, or a software dependency without forcing every domain into philosophical categories.

## Import a pack

```bash
essay-understanding ontology-import ontologies/company.example.json
```

Or post the same JSON to `POST /ontology/import`. `GET /ontology` returns the active registry. The repository's
company and philosophy files are examples, not mandatory schemas; a new domain should use its own namespace.

When the vocabulary is not known in advance, admit material to a neutral input layer and run the governed
[schema-discovery workflow](SCHEMA_DISCOVERY.md). Its LLM suggestions remain pending until approval.

## Model behavior

A configured generation model receives the active ontology. It may select an active relation, return `unrelated`,
or propose a namespaced relation for review. An unknown proposal is not persisted until registered. Vector
similarity only creates candidates and never decides a domain relation by itself.

## Shortcut layer

The shortcut layer is independent of every domain ontology. It is a peer procedural-memory vector namespace whose
items describe how to retrieve, not what answer to return. A query searches it first. A reliable match restricts
starting layers, relation types, depth, breadth, and stopping rules; a miss falls back to ordinary bounded
multi-layer exploration. A matched route that finds no seed evidence also falls back during the same query. Free or
fallback explorations with non-empty evidence create or reinforce candidate shortcuts; three observations promote
the same route in the Alpha implementation. This is an internal proxy for procedural usefulness, not correctness
feedback.
