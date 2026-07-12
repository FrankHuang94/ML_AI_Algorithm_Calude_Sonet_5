# KV Cache and Attention Optimization

Attention's quadratic cost in sequence length (see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md) and [state-space-models.md](../03-deep-learning-architectures/state-space-models.md)) is a training-time concern, but it also creates a distinct, arguably more operationally important, inference-time cost problem, centered on something called the KV cache. This file explains that problem from scratch, then covers the four major techniques used to manage it: FlashAttention, PagedAttention, and the MQA/GQA family of architectural changes.

## The KV cache, explained from scratch

**Why autoregressive generation needs it.** Recall from [autoregressive-generation.md](../04-generative-models/autoregressive-generation.md) that generating text one token at a time means running the model repeatedly, each time processing one additional token appended to the growing sequence. Recall from [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md) that computing self-attention at a given position requires that position's **query** vector, and the **key** and **value** vectors of *every* position up to and including it.

**The naive, wasteful approach.** Without any caching, generating each new token would require recomputing the key and value vectors for *every* previous position in the sequence all over again, from scratch, every single step — even though those earlier tokens' keys and values never change once computed (they only depend on the earlier tokens themselves, which are fixed once generated). For a sequence of length n, this means the total work across n generation steps grows roughly with n² — recomputing, redundantly, work that was already done in a previous step.

**The KV cache fix.** Instead, store (cache) each position's key and value vectors the first time they're computed, and simply reuse them for every subsequent generation step, computing only the new query, key, and value for the newest token at each step. This turns the redundant, ever-growing recomputation into a linear amount of new work per step (compute K/V once for the new token, attend the new query against all cached K/V) — a large practical speedup for autoregressive generation.

The waste (and the fix) are easiest to see laid out step by step — capitals = freshly computed this step, lowercase = reused from cache:

```
   Generating "The cat sat on":

   WITHOUT cache (recomputes everything, every step):    WITH KV cache (compute only the new one):
   step 1  "The"            → K/V for: [THE]              → compute & store K/V[THE]
   step 2  "The cat"        → K/V for: [THE  CAT]         → reuse the, compute & store K/V[CAT]
   step 3  "The cat sat"    → K/V for: [THE  CAT  SAT]    → reuse the,cat, compute K/V[SAT]
   step 4  "...sat on"      → K/V for: [THE CAT SAT ON]   → reuse the,cat,sat, compute K/V[ON]
                              ^^^ recomputes the same           ^^^ each token's K/V computed
                              early tokens over and over        exactly ONCE, then reused
                              (work ∝ n² total)                 (work ∝ n total)
```

The cache trades memory for compute: you never recompute a token's keys/values, at the cost of storing them all. For a long conversation that trade is overwhelmingly worth it on compute — but the stored cache itself becomes the new bottleneck, which is what the rest of this file is about.

**What it costs in memory.** The KV cache isn't free — it has to be stored somewhere, and its size grows linearly with sequence length, and also scales with the number of attention heads, the size of each head, the number of layers in the model, and the batch size (how many sequences are being generated simultaneously). For long contexts and/or large batches, the KV cache can become a very large fraction of total GPU memory usage — in some regimes, larger than the model's own weights — which makes managing it efficiently a first-order concern for serving cost and throughput, directly connected to the batching decisions covered in [serving-and-batching.md](serving-and-batching.md).

## FlashAttention — I/O-aware attention

**Origin.** Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022), with subsequent versions (FlashAttention-2, FlashAttention-3) improving hardware utilization further.

**The problem it solves.** On modern accelerators, moving data between the GPU's slower main memory (HBM — high-bandwidth memory, still much slower than the chip's on-board fast memory) and its much faster, but much smaller, on-chip memory is frequently the actual bottleneck in attention computation — not the raw arithmetic itself. A naive implementation of the attention formula (see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)) computes and stores the full n×n attention score matrix in slow memory at an intermediate step, which involves a large amount of otherwise-avoidable data movement back and forth.

**Core mechanism (conceptual — "I/O-aware," not a different math result).** FlashAttention computes the *exact same* attention output as the standard formula — this is a crucial, easily-missed point: it is not an approximation — but restructures the computation to process the input in small blocks that fit in the GPU's fast on-chip memory, computing partial attention results block by block and combining them incrementally, so that the full n×n score matrix is never fully materialized in slow memory at all. This is a case where a purely engineering-level reorganization of *how* a computation is scheduled and where its intermediate results live — with zero change to the mathematical result — produced a substantial real-world speedup, simply by better matching the computation's memory access pattern to the hardware's actual performance characteristics.

**Why it mattered.** FlashAttention became close to a default, drop-in replacement for standard attention implementations across essentially all serious Transformer training and inference stacks, since it's strictly faster and more memory-efficient for the exact same mathematical result — a rare "no real tradeoff" optimization in this repository's coverage.

## PagedAttention — an OS-paging analogy for KV cache memory

**Origin.** Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) — the algorithm underlying the widely-used vLLM serving system (see [serving-and-batching.md](serving-and-batching.md)).

**The problem it solves.** Naively, a serving system might reserve one large, contiguous block of memory for each request's KV cache, sized for the maximum sequence length that request might reach. This wastes a great deal of memory — most requests don't reach the maximum length, and reserving for the worst case up front means that memory sits unused (but unavailable to other requests) for the duration of the request, badly limiting how many requests can be served concurrently within a fixed memory budget.

**Core mechanism — the OS-paging analogy.** PagedAttention manages KV cache memory the way an operating system manages virtual memory for running programs: instead of one large contiguous allocation per request, the KV cache is broken into small, fixed-size **blocks** ("pages," directly analogous to OS memory pages), allocated to a request only as it actually needs them (as the sequence grows), and a lightweight lookup table (analogous to an OS page table) tracks which physical blocks belong to which logical position in each request's sequence. Blocks don't need to be physically contiguous in memory — the lookup table handles the indirection — which eliminates the waste of reserving unused worst-case memory up front and lets the serving system pack many more concurrent requests into the same physical memory.

**Why it mattered.** This memory-management technique, more than any single algorithmic change to attention itself, was widely reported to substantially increase achievable serving throughput (more concurrent requests served per unit of GPU memory) for LLM inference, and its underlying block/page-table design has become a standard architectural pattern in modern high-throughput LLM serving systems.

## Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

**The problem these address.** Standard multi-head attention (see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)) gives every attention head its own separate key and value projections — which means the KV cache must store a separate set of keys and values *per head*, multiplying the KV cache's memory cost by the number of heads.

**Multi-Query Attention (MQA).** Origin: Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019). MQA has every attention head share a *single* set of key and value projections (only the query projections remain separate per head), which shrinks the KV cache by roughly a factor of the number of heads — a large reduction — since keys and values, the things actually being cached, are no longer duplicated per head.

**Grouped-Query Attention (GQA).** Origin: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023). GQA is a middle ground between standard multi-head attention (every head has its own K/V) and MQA (all heads share one K/V): heads are split into a smaller number of groups, and all heads within a group share one set of key/value projections. This gives most of MQA's KV-cache memory savings while empirically preserving more of standard multi-head attention's quality than pure MQA does (having a moderate number of independent K/V projections, rather than just one, appears to preserve more of the representational benefit of having multiple heads).

**Why this mattered.** As KV cache memory became an increasingly dominant cost at long context lengths and high concurrent-request volumes, MQA/GQA became one of the most impactful *architectural* changes (decided at pretraining time, unlike FlashAttention/PagedAttention, which are inference-serving-level optimizations that don't require changing the trained model) for controlling long-term serving cost — which is why GQA specifically has become a standard architectural choice in many modern LLMs as of 2026, offering most of the serving-cost benefit of MQA with less of a quality tradeoff.

## Comparison table

| Technique | What it optimizes | Changes model architecture? | Changes attention's output? |
|---|---|---|---|
| KV cache (baseline concept) | Avoids redundant recomputation across generation steps | No | No |
| FlashAttention | Memory movement / speed of computing attention | No | No (exact, same result) |
| PagedAttention | KV cache memory fragmentation/waste across concurrent requests | No (serving-level) | No |
| MQA | KV cache memory size (shared K/V across all heads) | Yes (must be trained/converted this way) | Approximately (some quality tradeoff) |
| GQA | KV cache memory size (shared K/V within head groups) | Yes (must be trained/converted this way) | Approximately (smaller quality tradeoff than MQA) |

## Relationship to other algorithms

- The KV cache and its memory cost are the direct motivating problem behind quantization's relevance to serving (see [quantization.md](quantization.md)) and behind the batching decisions in [serving-and-batching.md](serving-and-batching.md).
- MQA/GQA are architectural decisions made at the same design stage as the positional encoding and normalization choices in [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md).
- Attention's underlying quadratic cost, which all of this section addresses from different angles, is the same problem state space models (see [state-space-models.md](../03-deep-learning-architectures/state-space-models.md)) attempt to sidestep architecturally rather than optimize around.
- Speculative decoding (see [speculative-decoding.md](speculative-decoding.md)) is a complementary technique that addresses generation speed from a different angle (reducing the number of sequential model calls) rather than the cost of each individual attention computation.

## Sources

- Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019) [MQA]
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023)
