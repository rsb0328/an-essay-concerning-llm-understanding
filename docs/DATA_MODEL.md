# Data model

The canonical model is domain-neutral. Philosophy, company memory, research, law, software operations, and
narrative analysis are ontology packs or workspaces built on the same storage and traversal machinery.

## Canonical objects

- **Layer type**: a registered, namespaced category such as `company:project`, `research:dataset`,
  `legal:rule`, or `philosophy:commentary`.
- **Layer**: a peer context containing one class or perspective of information. No layer is permanently privileged.
- **Node**: a provenance-bearing semantic unit in one layer. It may represent text, an entity, event, measurement,
  requirement, claim, or another domain-defined unit.
- **Relation type**: a registered ontology entry declaring direction, inverse, symmetry, transitivity, temporality,
  traversal weight, allowed endpoint layer types, and validators.
- **Mapping**: a typed relation between two nodes, with confidence, evidence, status, open attributes, and an
  optional validity interval. Similarity can nominate a pair; only a registered relation can become a mapping.
- **Shortcut layer**: an independent procedural-memory table and vector namespace parallel to domain knowledge
  layers. Its items store trigger examples and retrieval plans, never cached answers.
- **Shortcut**: a retrieval plan with starting layers, relation filters, depth, breadth, stopping rules, validators,
  failure conditions, maturity, reliability, history, and version.
- **Query run**: an auditable observation of mode, depth, evidence, shortcut use, latency, and output.

## Derived objects

Vectors and vector-store payloads are indexes. They may be deleted and rebuilt without changing canonical meaning.
Every vector points back to a canonical node or shortcut ID. Shortcut vectors live in the independent `shortcuts`
namespace, so shortcut-first routing is a real retrieval stage rather than hidden model state.

## Open but governed ontology

The core ships only structural types needed by the engine. A workspace imports or registers its vocabulary before
creating domain layers and mappings. Unknown types fail closed. Model-suggested unknown relations are returned as
proposals for review and are not written automatically. See [Extensible domain ontologies](ONTOLOGIES.md).

## Mathematical retrieval model

Let layer \(L_i\) contain nodes \(V_i\). An embedding function \(f_i\) maps a node into a search space, but the
canonical node and its provenance remain outside the vector index. The current implementation normally uses one
replaceable embedder across layers; it does not assume that an embedding alone represents the relation between them.

For a query \(q\), cosine similarity proposes seed or cross-layer candidates:

\[
s_0(v \mid q)=\cos(f(q),f(v)).
\]

An accepted cross-layer mapping is a property-graph edge
\(e=(u,v,r,c,a,[t_0,t_1])\): relation type \(r\), confidence \(c\), open attributes \(a\), and optional validity
interval. Its ontology supplies a traversal weight \(w_r\). A path \(p=(v_0,e_1,\ldots,e_k)\) is scored by the
implemented multiplicative propagation rule:

\[
S(p \mid q)=s_0(v_0 \mid q)\prod_{j=1}^{k}(c_{e_j}\,w_{r_j}\,\gamma),
\qquad \gamma=0.88.
\]

For a newly reached node, the reference implementation then combines its own semantic relevance with path
reliability:

\[
R(v\mid q)=0.6\cos(f(q),f(v))+0.4\max_{p\to v}S(p\mid q).
\]

Search is bounded by maximum depth \(D\), per-step breadth \(B\), allowed relation types, visited-node cycle
control, and a minimum information-gain stopping rule. These controls make “association depth” an explicit
research budget rather than an instruction for unlimited graph wandering.

This is currently **typed weighted graph mapping**, not a learned linear transformation \(W_{ij}x\) between every
pair of vector spaces. Future experiments may compare learned alignment matrices, contrastive cross-layer
projections, optimal transport, or graph neural message passing. Those methods are research directions, not claims
about what this release already implements.

## Shortcut lifecycle

```text
free exploration
  → candidate route in the shortcut layer
  → repeated confirmation
  → active shortcut
  → shortcut-first retrieval on a similar query
  → success/failure history
  → revision or retirement
```

The current reference threshold promotes a repeated route after three confirmations. This is an implementation
default, not a universal cognitive claim.

## Portability

`GET /export` and `essay-understanding export` return the ontology registry, layers, nodes, mappings, and shortcuts
without vectors. The format identifier is `essay-understanding-memory-v1`.
