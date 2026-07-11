# Distributed Training

Training a frontier-scale model requires spreading the work across hundreds to tens of thousands of accelerators (GPUs/TPUs) simultaneously — a single accelerator, however powerful, has neither enough memory to hold a modern large model plus its training state, nor enough raw compute to train it in a reasonable amount of time. This file covers the major strategies for splitting a training job across many devices, and the communication costs that come with doing so.

## Data parallelism

**Name & definition.** Data parallelism replicates the *entire* model on every accelerator, and splits the training data (each mini-batch) across them — every device processes a different slice of the batch using an identical copy of the model, and the resulting gradients are averaged across all devices before each parameter update.

**Core mechanism.** Each device computes a forward and backward pass (see [optimization-algorithms.md](../01-foundations/optimization-algorithms.md)) on its own slice of the mini-batch, producing its own local gradient estimate. An **all-reduce** operation (a collective communication pattern where every device ends up with the sum, or average, of a value across all devices) combines these local gradients into a single averaged gradient, which every device then uses to update its own (identical) copy of the model — keeping all copies synchronized after every step.

**Why it mattered / limitation.** Data parallelism is the simplest and most communication-efficient parallelism strategy (only gradients need to be communicated, not activations or model weights) and scales training throughput well when the model fits comfortably in a single device's memory. Its fundamental limitation: it does nothing to reduce the memory required to hold the model itself, its activations, and its optimizer state — every device still needs to hold a full copy of everything, which becomes impossible once the model is large enough (and at frontier LLM scale, it always is).

## Model/tensor parallelism

**Name & definition.** Tensor parallelism splits individual weight matrices/layers *within* the model across multiple devices, so that no single device needs to hold an entire layer's full set of weights.

**Core mechanism (conceptual).** For a large matrix multiplication (a core operation throughout a Transformer — see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)), the weight matrix can be split (e.g., by columns) across several devices, each device computing its own partial result using its slice of the weights, with the partial results combined (via communication between devices) to produce the full, correct output. This requires devices to communicate intermediate activations (not just gradients, as in data parallelism) at multiple points within a single layer's computation, which demands very fast interconnects between the devices involved (this is why tensor parallelism is typically applied only *within* a single physical server/node, where devices are connected via very high-bandwidth links, rather than across an entire cluster).

**Why it mattered.** Tensor parallelism directly addresses data parallelism's core limitation — it lets a model too large to fit on a single device be split across several, at the cost of requiring much more frequent, latency-sensitive inter-device communication than data parallelism does.

## Pipeline parallelism

**Name & definition.** Pipeline parallelism splits a model *by layer* across devices — e.g., the first quarter of the model's layers live on device 1, the next quarter on device 2, and so on — with data flowing through the devices in sequence, like an assembly line.

**Core mechanism.** Because each device only needs the output of the previous device's layers (not fine-grained intermediate communication within a single layer, unlike tensor parallelism), pipeline parallelism requires much less frequent, and much less latency-sensitive, communication — a real advantage for splitting a model across devices that aren't as tightly interconnected (e.g., across different nodes/servers in a cluster). The naive version of this has an obvious inefficiency, though: while device 1 is idle waiting for its output to work its way through devices 2, 3, and 4 and come back around for the next batch, it's doing nothing — a "pipeline bubble." Practical pipeline parallelism implementations reduce this idle time by splitting each mini-batch into smaller **micro-batches** and feeding them through the pipeline in an overlapping, staggered fashion, so that while device 1 works on micro-batch 2, device 2 is simultaneously working on micro-batch 1 — keeping more devices busy more of the time, analogous to instruction pipelining in a CPU.

**Why it mattered.** Pipeline parallelism is a natural complement to tensor parallelism: tensor parallelism handles splitting within a tightly-interconnected group of devices (a single node), while pipeline parallelism handles splitting across less-tightly-interconnected groups (across nodes) — and large-scale training runs typically combine both, along with data parallelism, in what's often called **3D parallelism**.

## ZeRO (Zero Redundancy Optimizer) — stages 1, 2, 3

**Name & definition.** ZeRO is a family of techniques that eliminates the memory redundancy in standard data parallelism (where every device holds a full, identical copy of the optimizer state, gradients, and parameters) by sharding (splitting) these across the data-parallel devices instead, while preserving data parallelism's relatively simple communication pattern.

**Origin.** Rajbhandari, Rasley, Ruwase, He (Microsoft), "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020).

**What each stage actually shards:**

- **ZeRO Stage 1** shards the **optimizer state** (recall from [optimization-algorithms.md](../01-foundations/optimization-algorithms.md) that Adam/AdamW keep two extra values per parameter — this state is often the single largest memory consumer in training, larger than the parameters or gradients themselves) across data-parallel devices — each device holds only its assigned shard of the optimizer state, rather than the full thing.
- **ZeRO Stage 2** additionally shards the **gradients** — each device only needs to hold the gradient values relevant to the optimizer-state shard it owns, further cutting memory.
- **ZeRO Stage 3** additionally shards the **model parameters themselves** — no single device holds a full copy of the model's weights at rest; instead, the specific parameter shards needed for a given computation are temporarily gathered from other devices just before they're used, then released again afterward.

Walkthrough: each stage trades a bit more communication overhead (to reconstruct the full state/gradients/parameters when actually needed for computation) for a large reduction in the memory required per device — Stage 3, the most aggressive, lets you train a model whose total size (across all its shards, combined) is far larger than what any single device could hold, using ordinary data-parallel-style training logic underneath, without needing the more invasive tensor/pipeline parallelism restructuring described above (though ZeRO is frequently combined with them anyway at the largest scales).

**Why it mattered.** ZeRO made it possible to train very large models using primarily data-parallel-style infrastructure (simpler to reason about and implement than tensor/pipeline parallelism) by attacking the actual source of the memory problem (redundant copies of state) directly, rather than requiring the model's computation itself to be restructured.

## Fully Sharded Data Parallel (FSDP)

**Name & definition.** FSDP is (in PyTorch's ecosystem specifically) the widely-used implementation of the ZeRO Stage 3 idea — full parameter, gradient, and optimizer-state sharding — integrated as a relatively easy-to-use training wrapper rather than a from-scratch custom implementation.

**Why it mattered.** FSDP made ZeRO-Stage-3-style full sharding broadly accessible to the wider research and engineering community (rather than requiring bespoke, in-house distributed training infrastructure), and is, alongside similar tools in other frameworks, one of the most commonly used tools for large-scale model training as of 2026.

## Communication bottlenecks and hardware/interconnect design

**Why this matters beyond software.** Every parallelism strategy above has a communication cost — all-reduce for data parallelism's gradient averaging, activation exchange for tensor parallelism, and activation hand-offs for pipeline parallelism — and as models and clusters have grown, this communication has become a first-order bottleneck, not an afterthought. This is the direct reason hardware and datacenter design for AI training has become its own significant engineering discipline: extremely high-bandwidth, low-latency interconnects between accelerators within a single server (needed for tensor parallelism's frequent, latency-sensitive communication), and high-bandwidth networking between servers within a cluster (needed for data-parallel gradient averaging and pipeline-parallel hand-offs across nodes) directly determine how efficiently a given amount of raw accelerator compute can actually be turned into training progress. A cluster with excellent individual accelerators but a poor interconnect can spend more time waiting on communication than doing useful computation — which is why frontier AI labs' infrastructure investment is as much about networking and interconnect topology as it is about the accelerators themselves.

## Comparison table

| Strategy | What's split across devices | Communication pattern | Typical scope |
|---|---|---|---|
| Data parallelism | Training data (batch); model fully replicated | All-reduce gradients each step | Any scale where model fits per-device |
| Tensor parallelism | Individual weight matrices/layers | Frequent, latency-sensitive activation exchange | Within a single tightly-interconnected node |
| Pipeline parallelism | Groups of layers (model depth) | Less frequent, activation hand-offs between stages | Across nodes in a cluster |
| ZeRO / FSDP | Optimizer state (Stage 1), + gradients (2), + parameters (3) | Gather/release shards as needed, layered on data-parallel communication | Any scale; commonly combined with tensor/pipeline parallelism at the largest scales |

## Relationship to other algorithms

- Optimizer state, the primary target of ZeRO sharding, is explained in its original context in [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).
- Mixture-of-Experts models add another dimension of distributed complexity — expert sharding across devices — building directly on the parallelism concepts in this file; see [mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md) and its discussion of GShard.
- Distributed training infrastructure choices interact with the data/curriculum strategies in [curriculum-and-data-strategies.md](curriculum-and-data-strategies.md) (e.g., how data is sharded and streamed to data-parallel workers).

## Sources

- Rajbhandari, Rasley, Ruwase, He, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020)
- Shoeybi et al. (NVIDIA), "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019) [tensor parallelism]
- Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism" (2019)
- Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (2021) [3D parallelism]
