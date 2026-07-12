# Graph Neural Networks

Graph Neural Networks (GNNs) extend deep learning to data that's naturally structured as a graph — nodes (entities) connected by edges (relationships) — rather than the grid-like structure CNNs assume (see [cnn-family.md](cnn-family.md)) or the linear sequence structure Transformers and RNNs assume. Examples of graph-structured data: molecules (atoms as nodes, chemical bonds as edges), social networks (people as nodes, friendships as edges), and knowledge graphs (entities as nodes, relationships as edges).

## The message passing framework

**Name & definition.** Message passing is the general computational pattern nearly all modern GNNs use: at each layer, every node gathers information ("messages") from its immediate graph neighbors, combines them, and uses the result to update its own representation.

**Core mechanism.**

```
message_i = AGGREGATE({ h_j : j ∈ neighbors(i) })
h_i' = UPDATE(h_i, message_i)
```

```
   Before one message-passing layer      After: node A has gathered from B, C, D
        B                                       B
        │                                       │
   C ── A ── D                             C ── A ── D
                                                ▲
   A's new representation = UPDATE(            A now "knows about" its
     A's old value,                            immediate neighbors B, C, D.
     AGGREGATE(B, C, D) )                      After a 2nd layer, it also knows
                                               about B/C/D's neighbors (2 hops), etc.
```

Walkthrough: for every node i, look at all of its directly-connected neighbor nodes j, combine their current representations h_j somehow (AGGREGATE — often a sum, mean, or max, since the number of neighbors varies per node and the aggregation needs to handle a variable-size input), and use that combined message, together with the node's own current representation, to compute its next-layer representation (UPDATE — typically a small neural network layer). Stacking k message-passing layers lets each node's final representation incorporate information from neighbors up to k graph-hops away — directly analogous to how stacking CNN layers grows the receptive field (see [cnn-family.md](cnn-family.md)), except the relevant "neighborhood" here is defined by graph connectivity rather than spatial adjacency. (In fact, a CNN can be seen as a special case of message passing on a very regular grid-shaped graph where each pixel's neighbors are the pixels around it, and self-attention as message passing on a fully-connected graph where every token is every other token's neighbor — a unifying view that recurs in the literature.)

**Why this framework matters.** It gives GNNs the same kind of built-in structural assumption (inductive bias — see the glossary) that made CNNs effective for images: rather than treating every node-pair as equally likely to be related (as a fully-connected network would, ignoring the graph structure entirely), message passing bakes in the assumption that directly-connected nodes are the most relevant source of information for updating a node's representation, which is usually a good real-world assumption for graph-structured data.

## Graph Convolutional Networks (GCN)

**Origin.** Kipf, Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (2017).

**Core mechanism.** GCN is a specific, simplified instance of message passing: it aggregates neighbor representations via a (degree-normalized) weighted average — normalized so that nodes with many neighbors don't automatically dominate the aggregation just by having a larger raw sum — followed by a shared linear transformation and nonlinearity, applied identically at every node (weight sharing, again analogous to how a CNN filter's weights are shared across spatial positions).

**Why it mattered.** GCN was one of the first GNN variants to demonstrate strong, simple, broadly-reproducible results on standard graph benchmarks (like node classification — predicting a label for each node using both its own features and its neighbors' features and connectivity), and remains a standard, well-understood baseline architecture in the field.

**Limitation.** The original GCN formulation is **transductive** — it's defined over a fixed, specific graph seen during training, and doesn't naturally generalize to predicting on new nodes or entirely new graphs not seen during training. This limitation directly motivated GraphSAGE (below).

## GraphSAGE

**Origin.** Hamilton, Ying, Leskovec, "Inductive Representation Learning on Large Graphs" (2017).

**Core contribution.** GraphSAGE ("SAmple and aggreGatE") makes message passing **inductive** — able to generalize to nodes and graphs never seen during training — by learning general aggregation *functions* (rather than fixed weights tied to specific nodes in a specific graph) and, for scalability on large graphs, by randomly *sampling* a fixed-size subset of each node's neighbors at each layer rather than aggregating over all of them (important for real-world graphs where some nodes — a very popular social media account, for instance — might have millions of neighbors, making full aggregation prohibitively expensive).

**Why it mattered.** GraphSAGE's inductive capability made GNNs practical for real-world, large, and constantly-changing graphs (like a social network gaining new users and connections every second), where you fundamentally cannot retrain the whole model from scratch every time the graph changes, and where you need to make predictions for nodes that didn't exist at training time.

## Graph Attention Networks (GAT)

**Origin.** Veličković et al., "Graph Attention Networks" (2018).

**Core contribution.** GAT replaces GCN's fixed, degree-based neighbor-averaging with a learned, content-based attention mechanism (directly analogous to the attention covered in [transformer-architecture.md](transformer-architecture.md)) — each node computes an attention weight for each of its neighbors, based on both nodes' current representations, and aggregates neighbor information using those learned, content-dependent weights rather than a fixed structural formula.

**Why it mattered.** This let the model learn that some neighbors matter more than others for a given node's representation (rather than treating all neighbors as equally important modulo their degree), the same qualitative benefit that content-based attention provides over fixed positional/structural weighting elsewhere in this repository — bringing GNNs conceptually in line with the broader attention-based direction the field moved in around the same period.

## Use cases

- **Molecular modeling.** Representing a molecule as a graph (atoms as nodes, bonds as edges) and using a GNN to predict molecular properties (solubility, toxicity, binding affinity) or to generate novel candidate molecules — a major and still-growing application area in computational chemistry and drug discovery, where GNNs' natural fit to graph-structured chemical data gives them a real structural advantage over architectures that would need molecules awkwardly reformatted into grids or sequences.
- **Recommendation systems.** Modeling users and items as nodes in a bipartite graph (edges representing interactions like purchases or clicks), where GNN-based approaches can propagate information across the graph (e.g., "users similar to you liked this item") more naturally than traditional matrix-factorization-based recommendation approaches.
- **Knowledge graphs.** Representing structured facts (entity-relationship-entity triples) as a graph and using GNNs for tasks like link prediction (will this relationship likely exist, even though it isn't explicitly recorded?) or entity classification.

## Current status

GNNs remain the standard, most natural architectural choice specifically for genuinely graph-structured data (molecules, social/citation networks, knowledge graphs) — this is a domain where their structural fit gives them a real, durable advantage over generic architectures, similar to how CNNs retain an advantage for genuinely grid-structured image data even in a Transformer-dominated era. GNNs are a comparatively specialized corner of the field relative to the enormous investment in general-purpose Transformer-based LLMs, but within their applicable domains (especially computational chemistry/drug discovery and large-scale industrial recommendation systems) they remain in substantial, non-declining production use as of 2026.

## Comparison table

| Architecture | Year | Key Innovation | Still Relevant (2026)? |
|---|---|---|---|
| GCN | 2017 | Degree-normalized neighbor averaging | Yes — standard baseline |
| GraphSAGE | 2017 | Inductive learning + neighbor sampling for scalability | Yes — standard for large/dynamic graphs |
| GAT | 2018 | Learned, content-based attention over neighbors | Yes — standard when neighbor importance varies |

## Relationship to other algorithms

- Message passing's neighbor-aggregation step is conceptually the graph-structured analogue of a CNN's convolution over spatial neighbors — see [cnn-family.md](cnn-family.md).
- GAT's attention mechanism is a direct structural cousin of the self-attention mechanism in [transformer-architecture.md](transformer-architecture.md), applied over graph neighbors instead of sequence positions.
- GNN-based molecular modeling is a domain-specific alternative/complement to the generative models in [04-generative-models](../04-generative-models/) when the generation target (a molecule) is naturally graph-structured rather than a sequence or image.

## Sources

- Kipf, Welling, "Semi-Supervised Classification with Graph Convolutional Networks" (2017)
- Hamilton, Ying, Leskovec, "Inductive Representation Learning on Large Graphs" (2017)
- Veličković, Cucurull, Casanova, Romero, Liò, Bengio, "Graph Attention Networks" (2018)
