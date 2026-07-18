# Open scaling study: from small memory to large memory

The present results are small because local hardware, model inference time, and the cost of repeated controlled runs limited the experiment. Ten question categories cannot tell us whether layered retrieval becomes more useful—or merely more expensive—as memory grows.

We invite independent replications. The central question is not only “does it work at one size?” but:

> How do single-layer retrieval, unrestricted layered traversal, and shortcut-first layered traversal scale as the number of nodes, layers, mappings, and accumulated shortcuts increases?

## Requested scale points

Where hardware permits, run at approximately:

| Scale | Nodes | Layers | Mappings |
|---|---:|---:|---:|
| S | 1,000 | 5 | 5,000 |
| M | 10,000 | 20 | 50,000 |
| L | 100,000 | 50 | 500,000 |
| XL | 1,000,000 | 100 | 5,000,000 |

Intermediate logarithmic points are welcome. Do not claim an XL result if the generated graph has materially different density or query difficulty.

## Compare three modes

1. Flat, single-layer vector retrieval.
2. Layered retrieval with the same maximum evidence budget.
3. Shortcut-first layered retrieval after a controlled warm-up period.

Report cold and warm runs separately. A shortcut system is specifically hypothesized to improve with repeated related tasks, so mixing first encounters with repeated encounters hides the effect under study.

## Required measurements

- Index construction time and peak memory.
- P50/P95/P99 retrieval and end-to-end latency.
- Queries per second under concurrency 1, 4, and 16.
- Retrieved evidence count, layer diversity, traversed edges, and reached depth.
- Recall@k against known relevant nodes.
- Claim coverage and citation grounding.
- Shortcut hit rate, false-route rate, and latency saved after warm-up.
- Failure, retry, timeout, and recovery counts.
- Hardware, operating system, model, quantization, embedding model, vector store, and all relevant settings.

## Curves we want to see

Plot each metric against node count on a logarithmic x-axis. Especially useful curves are:

- retrieval latency vs. node count;
- recall/claim coverage vs. node count;
- shortcut hit rate and time saved vs. accumulated related queries;
- false shortcut selection vs. shortcut-layer size;
- total cost of learning a route vs. cumulative cost saved by reuse;
- accuracy and latency vs. association depth.

Submit raw machine-readable results together with a short methodology note. Negative results are as valuable as positive ones. The purpose is to find the regime, if any, in which layered procedural memory earns its added complexity.

