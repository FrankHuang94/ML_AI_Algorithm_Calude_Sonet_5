# KV Cache and Attention Optimization

Attention's quadratic cost in sequence length (see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md) and [state-space-models.md](../03-deep-learning-architectures/state-space-models.md)) is a training-time concern, but it also creates a distinct, arguably more operationally important, inference-time cost problem, centered on something called the KV cache. This file explains that problem from scratch, then covers the techniques used to manage it in two groups: **serving-level techniques** that change how attention is computed or how its cache is stored without changing the trained model (FlashAttention, PagedAttention), and **architectural techniques**, decided at pretraining time, that shrink, window, sparsify, or compress what has to be cached (MQA/GQA, Multi-head Latent Attention, sliding-window attention, DeepSeek Sparse Attention, and DeepSeek-V4's compressed attention).

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

**What it costs in memory.** The KV cache isn't free — it has to be stored somewhere, and its size grows linearly with sequence length, and also scales with the number of attention heads, the size of each head, the number of layers in the model, and the batch size (how many sequences are being generated simultaneously). Written out for standard multi-head attention:

```
KV cache bytes = 2 × n_layers × n_kv_heads × d_head × seq_len × batch_size × bytes_per_number
                 ↑ one K and one V
```

Walkthrough: every layer stores, for every token of every sequence in the batch, one key vector and one value vector per KV head, each `d_head` numbers long. Every factor is a lever: the techniques below attack `n_kv_heads × d_head` (MQA, GQA, MLA), `seq_len` (sliding windows, sequence compression), the number of layers that keep a cache at all (linear-attention hybrids), and `bytes_per_number` (KV cache quantization — see [quantization.md](quantization.md)). For long contexts and/or large batches, the KV cache can become a very large fraction of total GPU memory usage — in some regimes, larger than the model's own weights — which makes managing it efficiently a first-order concern for serving cost and throughput, directly connected to the batching decisions covered in [serving-and-batching.md](serving-and-batching.md).

A second, subtler cost: during generation, every new token's attention has to *read* the entire cache from GPU memory. So a big cache costs not only storage but also memory bandwidth on every decoding step — which is why reading *less* of the cache (sparse attention, below) helps speed even when it doesn't reduce storage.

## FlashAttention — I/O-aware attention

**Origin.** Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022), with subsequent versions (FlashAttention-2, FlashAttention-3) improving hardware utilization further.

**The problem it solves.** On modern accelerators, moving data between the GPU's slower main memory (HBM — high-bandwidth memory, still much slower than the chip's on-board fast memory) and its much faster, but much smaller, on-chip memory is frequently the actual bottleneck in attention computation — not the raw arithmetic itself. A naive implementation of the attention formula (see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)) computes and stores the full n×n attention score matrix in slow memory at an intermediate step, which involves a large amount of otherwise-avoidable data movement back and forth.

**Core mechanism (conceptual — "I/O-aware," not a different math result).** FlashAttention computes the *exact same* attention output as the standard formula — this is a crucial, easily-missed point: it is not an approximation — but restructures the computation to process the input in small blocks that fit in the GPU's fast on-chip memory, computing partial attention results block by block and combining them incrementally, so that the full n×n score matrix is never fully materialized in slow memory at all. This is a case where a purely engineering-level reorganization of *how* a computation is scheduled and where its intermediate results live — with zero change to the mathematical result — produced a substantial real-world speedup, simply by better matching the computation's memory access pattern to the hardware's actual performance characteristics.

**Why it mattered.** FlashAttention became close to a default, drop-in replacement for standard attention implementations across essentially all serious Transformer training and inference stacks, since it's strictly faster and more memory-efficient for the exact same mathematical result — a rare "no real tradeoff" optimization in this repository's coverage.

## PagedAttention — an OS-paging analogy for KV cache memory

**Origin.** Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) — the algorithm underlying the widely-used vLLM serving system (see [serving-and-batching.md](serving-and-batching.md)).

**The problem it solves.** Naively, a serving system might reserve one large, contiguous block of memory for each request's KV cache, sized for the maximum sequence length that request might reach. This wastes a great deal of memory — most requests don't reach the maximum length, and reserving for the worst case up front means that memory sits unused (but unavailable to other requests) for the duration of the request, badly limiting how many requests can be served concurrently within a fixed memory budget.

**Core mechanism — the OS-paging analogy.** PagedAttention manages KV cache memory the way an operating system manages virtual memory for running programs: instead of one large contiguous allocation per request, the KV cache is broken into small, fixed-size **blocks** ("pages," directly analogous to OS memory pages), allocated to a request only as it actually needs them (as the sequence grows), and a lightweight lookup table (analogous to an OS page table) tracks which physical blocks belong to which logical position in each request's sequence. Blocks don't need to be physically contiguous in memory — the lookup table handles the indirection — which eliminates the waste of reserving unused worst-case memory up front and lets the serving system pack many more concurrent requests into the same physical memory.

**Why it mattered.** This memory-management technique, more than any single algorithmic change to attention itself, was widely reported to substantially increase achievable serving throughput (more concurrent requests served per unit of GPU memory) for LLM inference, and its underlying block/page-table design has become a standard architectural pattern in modern high-throughput LLM serving systems. The same block structure also makes it cheap to *share* cache blocks between requests that start with the same prompt prefix — the basis of prefix caching, covered in [serving-and-batching.md](serving-and-batching.md).

## Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

**The problem these address.** Standard multi-head attention (see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)) gives every attention head its own separate key and value projections — which means the KV cache must store a separate set of keys and values *per head*, multiplying the KV cache's memory cost by the number of heads.

**Multi-Query Attention (MQA).** Origin: Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019). MQA has every attention head share a *single* set of key and value projections (only the query projections remain separate per head), which shrinks the KV cache by roughly a factor of the number of heads — a large reduction — since keys and values, the things actually being cached, are no longer duplicated per head.

**Grouped-Query Attention (GQA).** Origin: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023). GQA is a middle ground between standard multi-head attention (every head has its own K/V) and MQA (all heads share one K/V): heads are split into a smaller number of groups, and all heads within a group share one set of key/value projections. This gives most of MQA's KV-cache memory savings while empirically preserving more of standard multi-head attention's quality than pure MQA does (having a moderate number of independent K/V projections, rather than just one, appears to preserve more of the representational benefit of having multiple heads).

**Why this mattered.** As KV cache memory became an increasingly dominant cost at long context lengths and high concurrent-request volumes, MQA/GQA became one of the most impactful *architectural* changes (decided at pretraining time, unlike FlashAttention/PagedAttention, which are inference-serving-level optimizations that don't require changing the trained model) for controlling long-term serving cost — which is why GQA specifically has become a standard architectural choice in many modern LLMs as of late 2026, offering most of the serving-cost benefit of MQA with less of a quality tradeoff.

## Multi-head Latent Attention (MLA)

**Name & definition.** Multi-head Latent Attention compresses each token's keys and values into one small shared **latent vector** (a compact learned summary — "latent" meaning an internal representation the model learns rather than one with a predefined meaning), caches only that latent, and reconstructs per-head keys and values from it when attention is computed.

**Origin.** DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (May 2024); carried forward in DeepSeek-V3 and DeepSeek-R1.

**Core mechanism — the intuition.** MQA and GQA save memory by making heads *share* keys and values, which throws away some of what makes multiple heads useful. MLA takes a different bet: the full set of per-head keys and values for a token is highly redundant, so it can be squeezed through a narrow bottleneck *without* forcing heads to share. Think of it like storing a compressed file instead of the original: you keep a small archive in the cache and "unzip" it into full per-head K and V only when needed — and because the unzipping is a fixed linear map, it can largely be folded into other matrix multiplications rather than actually performed.

```
c_t        = W_down · h_t           (h_t: token t's hidden state; c_t: small latent, THIS is cached)
k_t^(i)    = W_upK^(i) · c_t        (key for head i, rebuilt from the latent)
v_t^(i)    = W_upV^(i) · c_t        (value for head i, rebuilt from the latent)
```

Walkthrough: `W_down` is a learned "down-projection" matrix that maps the token's full hidden state to a latent `c_t` that is much shorter than all heads' keys and values put together. Each head `i` has its own learned "up-projection" matrices `W_upK^(i)` and `W_upV^(i)` that turn the shared latent back into that head's own key and value — so, unlike MQA/GQA, every head still gets distinct keys and values. The trick that makes this fast at inference: the attention score for head `i` is `q^(i) · k_t^(i) = q^(i) · (W_upK^(i) · c_t)`, and matrix multiplication is associative, so `W_upK^(i)` can be pre-multiplied into the query side ("absorbed"), letting the query be compared directly against the cached latent without ever materializing full keys. The value up-projection can similarly be absorbed into the attention output projection.

**One complication — positional information.** RoPE (rotary position embedding, see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)) rotates keys by a position-dependent angle, which breaks the "absorb the up-projection into the query" trick (the rotation sits between the two matrices and changes with position). MLA's fix is a **decoupled RoPE key**: a small extra per-token key component that carries the positional rotation, shared across heads and cached alongside the latent. So the cache per token per layer is "latent + small positional key."

**What gets cached per token — a visual.** Illustrative numbers for a hypothetical layer with 32 heads of dimension 128 (these are *not* any specific model's configuration; they are chosen only to show the shape of the comparison):

```
 Numbers cached per token, per layer           (bar length ∝ cache size)

 MHA    32 heads × (K 128 + V 128)       ████████████████████████████████  8,192
 GQA-8   8 groups × (K 128 + V 128)      ████████                          2,048
 MLA    1 latent (512) + RoPE key (64)   ██▎                                 576
 MQA     1 shared × (K 128 + V 128)      █                                   256

 MHA / GQA / MQA: cache the actual keys and values (fewer distinct heads → smaller, but heads must share)
 MLA:             caches a compressed latent; every head still gets its own K/V, rebuilt on the fly
```

The comparison that matters is not raw size alone — MQA is smaller still — but size *at a given quality*. DeepSeek reported that MLA's cache is much smaller than GQA's while matching or exceeding standard multi-head attention's quality, which MQA and GQA generally do not.

**Why it mattered.** MLA showed that KV cache size and attention quality are not a simple one-dimensional tradeoff controlled by "how many heads share K/V." It was a central ingredient in the DeepSeek models' reportedly low serving cost, and it was adopted beyond DeepSeek — for example, Moonshot's Kimi Linear (Oct 2025) uses MLA for its full-attention layers.

**Current status (late 2026).** Established in production in the DeepSeek lineage and some other open-weight models; GQA remains the more widespread default overall. MLA is the main alternative wherever KV cache size is a first-order concern.

**Strengths & limitations.**
- Strengths: large KV cache reduction without forcing heads to share keys/values; quality reported on par with or better than multi-head attention; composes with paging, quantization, and sparse attention.
- Limitations: more complex to implement efficiently (custom kernels, the RoPE decoupling); the latent is a low-rank bottleneck whose size is a new hyperparameter; like MQA/GQA, it must be built in at training time (converting an existing model is possible in principle but is not a free drop-in).

## Sliding-window attention and local/global interleaving

**Name & definition.** In **sliding-window (local) attention**, each token attends only to the most recent W tokens rather than the whole history. **Local/global interleaving** stacks many local layers with occasional **global** (full-attention) layers.

**Origin.** Fixed sparse attention patterns appear in Child et al., "Generating Long Sequences with Sparse Transformers" (2019) and Beltagy et al., "Longformer" (2020); Mistral 7B (Jiang et al., 2023) brought sliding-window attention into a widely-used open LLM.

**Core mechanism.** A sliding-window layer only ever needs the last W tokens' keys and values, so its KV cache stops growing at W tokens (it can be kept as a ring buffer — a fixed-size array where the newest entry overwrites the oldest). Information from further back is not lost entirely: each stacked local layer lets information hop another W positions, so the effective reach grows with depth. Interleaving a few global layers guarantees that any token can still be reached directly.

```
 Layer pattern (example, L = local window, G = global/full):   L L L G L L L G ...
 KV cache per layer:  L layers → capped at W tokens     G layers → grows with the full context
```

**Where it's used today.** Gemma 2 and Gemma 3 interleave local sliding-window layers with global layers; OpenAI's gpt-oss (Aug 2025) alternates banded (windowed) and dense attention layers.

**A pitfall: attention sinks.** Trained models often dump a large share of attention onto the first few tokens of the sequence, using them as a "sink" for attention mass that has nowhere useful to go. If a window slides past those tokens, quality can collapse. Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (StreamingLLM, 2023) showed that simply keeping the first few tokens' KV entries permanently fixes much of this; some newer models instead learn an explicit sink, and the Qwen team's gated attention (2025) — a learned elementwise sigmoid gate on the attention output — was reported to largely remove sinks in the first place.

**Strengths & limitations.** Cheap and simple; cache for local layers is bounded. But local layers genuinely cannot see far-back tokens directly, so long-range retrieval rests on the global layers, whose caches still grow with context.

## Sparse attention: DeepSeek Sparse Attention (DSA)

**Name & definition.** Sparse attention lets each query attend to a *selected subset* of past tokens instead of all of them. DeepSeek Sparse Attention chooses that subset *per query, based on content*, using a cheap scoring module called the **lightning indexer**.

**Origin.** DeepSeek-V3.2-Exp (Sep 2025) and DeepSeek-V3.2 (Dec 2025). Earlier sparse-attention work (Sparse Transformer, Longformer, BigBird, 2019–2020) mostly used *fixed* patterns (windows, strides, a few global tokens) rather than content-based selection.

**Core mechanism.** A two-stage "skim, then read carefully" procedure for each new query token:

```
for the current query token q:
    scores  = lightning_indexer(q, all previous tokens)   # cheap, small scoring network
    chosen  = top_k(scores, k)                            # keep the k most relevant positions
    output  = attention(q, K[chosen], V[chosen])          # expensive attention over only k tokens
```

Walkthrough: the indexer still looks at every previous token, so it is technically still quadratic over a whole sequence — but it is designed to be far cheaper per token than real attention. The expensive main attention then touches only `k` tokens per query instead of all `n`, so for long contexts most of the cost moves from "expensive × n" to "cheap × n + expensive × k." Unlike a fixed window, the selected tokens can be anywhere in the history — the indexer can pick out one relevant paragraph from far back.

**An important nuance.** DSA reduces compute and the amount of cache *read* per step, but by itself it does not reduce how much cache is *stored* — any past token might be selected later, so all of them must be kept. (Storage reduction comes from other techniques in this file, such as MLA or sequence compression, below.)

**Status.** In production in DeepSeek's V3.2 line as of late 2026; content-based sparse attention as a general technique is still young, and how well it holds up across all long-context tasks is an active question.

## Compressed attention in DeepSeek-V4: CSA and HCA (reported)

**What it is.** DeepSeek-V4's technical report (Apr 2026; models V4-Pro, 1.6T total / 49B active parameters, and V4-Flash, 284B / 13B active, both Mixture-of-Experts with a 1M-token context) describes a hybrid attention design that interleaves two kinds of layers. Details below are as reported by DeepSeek; independent analysis is still limited.

- **Compressed Sparse Attention (CSA).** Every few consecutive tokens are compressed into a single KV entry. A lightning indexer (as in DSA) picks the top-k most relevant *compressed* entries, and the query attends to those plus a **sliding window** of recent, uncompressed tokens.
- **Heavily Compressed Attention (HCA).** A much more aggressive compression (reported ~128× fewer KV entries), with ordinary dense attention over all the compressed blocks, again plus a sliding window of recent tokens.

```
 Past context:   [t1 t2 t3 t4][t5 t6 t7 t8][ ... ][ ... ]  | recent tokens (uncompressed window)

 CSA layer:  [c1][c2][c3][c4] ...  ─ indexer picks top-k ─►  attend to {top-k compressed} ∪ {window}
 HCA layer:  [====== C1 ======][====== C2 ======] ...     ►  attend to {all heavy blocks} ∪ {window}
```

**The intuition — compress along a different axis.** MLA compresses each token's *features* (many numbers per token → fewer numbers per token). CSA and HCA compress along the *sequence* (many tokens → fewer cache entries). The design then combines three ideas from this file: sequence compression to shrink storage, top-k sparse selection (CSA) to read only what's relevant, and a sliding window so recent context is always seen exactly. Pairing a finer-grained, selective layer type (CSA) with a coarser, everything-at-once type (HCA) is a way to get both precise retrieval and a cheap global overview.

**Reported effect.** At the same long context length, DeepSeek reports V4 needs about 27% of V3.2's per-token FLOPs and about 10% of its KV cache. These are the vendor's own figures; the exact compression ratios for CSA beyond "every few tokens" and the quality tradeoffs on diverse long-context tasks should be treated as not yet independently verified.

## Linear-attention hybrids (pointer)

A different way to shrink the cache is to replace most attention layers with **linear-attention** or state-space layers, which keep a *fixed-size* recurrent state instead of a growing KV cache — only the remaining full-attention layers keep a cache. With a 3:1 ratio of linear to full-attention layers, for example, only about a quarter of the layers store per-token K/V. Production examples include Qwen3-Next (Sep 2025) and Qwen3.5 (early 2026) with Gated DeltaNet layers at ~3:1, Moonshot's Kimi Linear (Oct 2025) with Kimi Delta Attention and MLA layers at ~3:1, and NVIDIA's Nemotron-H (Mamba-2 + attention). The mechanisms are covered in [state-space-models.md](../03-deep-learning-architectures/state-space-models.md).

## Where things stand (late 2026)

FlashAttention-style kernels and paged KV cache management are universal in serious serving stacks. On the architectural side there is no single winner; instead, long-context frontier models increasingly *combine* several of the levers above — GQA or MLA to shrink per-token cache, sliding windows or linear-attention layers to cap cache growth in most layers, and content-based sparse or compressed attention (DSA, CSA/HCA) to make the remaining full-context layers cheaper. Which combination generalizes best is an open question, and published comparisons are mostly from the labs that designed each approach.

## Comparison table

| Technique | What it optimizes | Changes model architecture? | Changes attention's output? |
|---|---|---|---|
| KV cache (baseline concept) | Avoids redundant recomputation across generation steps | No | No |
| FlashAttention | Memory movement / speed of computing attention | No | No (exact, same result) |
| PagedAttention | KV cache memory fragmentation/waste across concurrent requests | No (serving-level) | No |
| MQA | KV cache memory size (shared K/V across all heads) | Yes (must be trained/converted this way) | Approximately (some quality tradeoff) |
| GQA | KV cache memory size (shared K/V within head groups) | Yes (must be trained/converted this way) | Approximately (smaller quality tradeoff than MQA) |
| MLA | KV cache size (cache a compressed latent; per-head K/V rebuilt) | Yes (trained this way) | Different mechanism; quality reported on par with or above MHA |
| Sliding-window / local-global | Caps cache growth in local layers | Yes (layer pattern set at training) | Yes (local layers see only the last W tokens) |
| DeepSeek Sparse Attention | Attention compute and cache reads (top-k tokens per query) | Yes (indexer trained with model) | Yes (attends to a selected subset) |
| CSA / HCA (DeepSeek-V4, reported) | Cache size and compute via sequence compression + sparse selection + window | Yes | Yes (attends to compressed entries) |
| Linear-attention hybrid layers | Replaces most layers' growing cache with a fixed-size state | Yes | Yes (different layer type) |

## Relationship to other algorithms

- The KV cache and its memory cost are the direct motivating problem behind quantization's relevance to serving (see [quantization.md](quantization.md)) and behind the batching, prefix-caching, and prefill/decode-disaggregation decisions in [serving-and-batching.md](serving-and-batching.md).
- MQA/GQA/MLA and local/global layer patterns are architectural decisions made at the same design stage as the positional encoding and normalization choices in [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md).
- Attention's underlying quadratic cost, which all of this section addresses from different angles, is the same problem state space models and linear-attention hybrids (see [state-space-models.md](../03-deep-learning-architectures/state-space-models.md)) attempt to sidestep architecturally rather than optimize around.
- DeepSeek's MLA, DSA, and CSA/HCA ship inside Mixture-of-Experts models (see [mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md)); MoE cuts per-token compute in the feedforward layers, while these techniques cut the attention/cache side.
- Speculative decoding (see [speculative-decoding.md](speculative-decoding.md)) is a complementary technique that addresses generation speed from a different angle (reducing the number of sequential model calls) rather than the cost of each individual attention computation.

## Sources

- Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019) [MQA]
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023)
- DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (2024) [MLA]
- Child, Gray, Radford, Sutskever, "Generating Long Sequences with Sparse Transformers" (2019)
- Beltagy, Peters, Cohan, "Longformer: The Long-Document Transformer" (2020)
- Jiang et al., "Mistral 7B" (2023) [sliding-window attention]
- Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (2023)
- Gemma Team, Gemma 2 (2024) and Gemma 3 (2025) technical reports [local/global interleaving]
- OpenAI, gpt-oss-120b / gpt-oss-20b model card (2025) [alternating banded-window and dense attention]
- Qwen Team, "Gated Attention for Large Language Models" (2025) [gated attention; reported to remove attention sinks]
- DeepSeek-AI, DeepSeek-V3.2-Exp (Sep 2025) and DeepSeek-V3.2 (Dec 2025) [DeepSeek Sparse Attention, lightning indexer]
- DeepSeek-AI, DeepSeek-V4 technical report (Apr 2026) [CSA/HCA; FLOPs and KV cache figures are as reported by the vendor]
- Moonshot AI, Kimi Linear (2025); Qwen Team, Qwen3-Next (2025) and Qwen3.5 (2026) [linear-attention hybrids — see state-space-models.md]
