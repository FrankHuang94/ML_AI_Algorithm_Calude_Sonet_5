# State Space Models

State Space Models (SSMs) are a family of sequence architectures proposed as an alternative to attention, aiming to keep the long-range modeling ability of Transformers (see [transformer-architecture.md](transformer-architecture.md)) while avoiding attention's quadratic compute cost. This file also covers the closely related family of **linear attention** models (including **Gated DeltaNet**), because by 2024-2025 the two lines of work had converged into essentially one design space: a recurrent layer with a fixed-size state. Per this repository's quality bar, the honest headline as of late 2026 is two-sided: **pure** SSM/linear-attention models remain rare at the frontier, but **hybrid** architectures — mostly linear-attention layers with a minority of full-attention layers mixed in — have moved from research into production flagship models.

## Motivation: the quadratic attention cost problem

**The problem.** Self-attention computes a similarity score between every pair of positions in a sequence (the Q·Kᵀ term covered in [transformer-architecture.md](transformer-architecture.md)). For a sequence of length n, this means both the compute and the memory required grow proportional to n² — doubling the sequence length quadruples the cost. For short sequences this is fine, but as context lengths have grown into the hundreds of thousands of tokens for frontier LLMs, the quadratic term becomes a dominant cost, both during training and — especially — during inference, where it interacts with the KV cache (see [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md)).

**The SSM pitch.** What if a sequence architecture could process each new token in *constant* time and memory relative to how much sequence has come before, the way an RNN's recurrence does (see [rnn-lstm-gru.md](rnn-lstm-gru.md)), but without RNNs' vanishing-gradient problems and slow, strictly sequential training? State space models are, in a real sense, an attempt to bring recurrence back — with a much more carefully engineered internal mechanism — specifically to get linear (rather than quadratic) scaling with sequence length.

## S4

**Name & definition.** S4 (Structured State Space Sequence model) reformulates a classical continuous-time control-theory concept — a linear state space model, where a hidden state evolves continuously over time according to a linear differential equation driven by the input — into a form usable as a deep learning sequence layer.

**Origin.** Gu, Goel, Ré, "Efficiently Modeling Long Sequences with Structured State Spaces" (2021/2022).

**Core mechanism (conceptual).** A continuous-time state space model is defined by two update rules (in discretized form, since neural networks operate on discrete sequences of tokens/timesteps):

```
hₜ = A · h_{t-1} + B · xₜ
yₜ = C · hₜ
```

Walkthrough: this looks structurally similar to a vanilla RNN's recurrence (see [rnn-lstm-gru.md](rnn-lstm-gru.md)) — a hidden state hₜ updated from the previous hidden state and the current input. The crucial difference is that S4 places careful mathematical structure on the matrix A (specifically, structured so that it corresponds to a good "memory" — a HiPPO matrix, from earlier related work by the same research group, designed so that the hidden state can represent a compressed history of the input reasonably well over long ranges) and — critically for practicality — because this recurrence is *linear* (unlike an RNN's recurrence, which passes through a nonlinearity like tanh at every step), the entire sequence's output can be computed as a **convolution** over the whole input at once during training (mathematically equivalent to unrolling the recurrence, but computable in parallel, avoiding the RNN's strictly sequential training bottleneck), while still being usable as a genuine step-by-step recurrence at inference time, giving constant per-token cost and memory.

**Why it mattered.** S4 demonstrated, on long-range sequence benchmarks specifically designed to stress long-dependency modeling, that a properly-structured linear recurrence could substantially outperform both Transformers and vanilla RNNs on very long sequences, while scaling linearly (not quadratically) with sequence length — the first strong empirical evidence that an attention-free architecture might be competitive for long-context modeling.

## Mamba — selective state spaces

**Name & definition.** Mamba extends S4 by making the state space model's core matrices (A, B, C, and the discretization step size) **input-dependent** — computed dynamically from the current input, rather than fixed (the same for every position, as in S4) — a property the authors call **selectivity**.

**Origin.** Gu, Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023).

**Why selectivity mattered.** S4's fixed A/B/C matrices mean the model treats every position's "how much to remember vs. forget" behavior identically, regardless of content — a real limitation, since ideally a model would want to remember content-critical tokens (like a name mentioned once) much longer than filler tokens, content-dependently, the way attention's content-based weighting allows (see [transformer-architecture.md](transformer-architecture.md)). Making the recurrence's parameters a function of the current input lets Mamba selectively decide, per token, how much to write into and retain in its hidden state — closing much of the expressiveness gap between SSMs and attention's content-based weighting. The tradeoff is that this selectivity breaks the clean "equivalent to a convolution" trick that made S4 easy to parallelize during training, so Mamba instead relies on a custom hardware-aware parallel scan algorithm (an efficient parallel implementation of the recurrence, engineered specifically to run well on GPUs) to keep training efficient despite the input-dependent recurrence.

**Why it mattered.** Mamba showed competitive results against similarly-sized Transformers on language modeling benchmarks while retaining linear-time, constant-memory-per-token inference — a genuinely compelling result that renewed serious interest in attention-free architectures within the research community.

**Mamba-2 (2024).** Dao and Gu's follow-up, "Transformers are SSMs" (2024), simplified Mamba's state matrix (to a scalar-times-identity per step) and showed that the resulting selective SSM is mathematically a form of *linear attention with a decay* — the "state space duality." This is the bridge to the next section: from 2024 on, "SSM" and "linear attention" are best thought of as two vocabularies for one family of fixed-state recurrent layers. Mamba-2 is the SSM layer used in several of the production hybrids described below.

## RWKV

**Name & definition.** RWKV (an acronym derived from the four main parameter matrices in its recurrence: Receptance, Weight, Key, Value) is another linear-recurrence-based architecture, developed with an explicit design goal of combining Transformer-like parallelizable training with RNN-like constant-memory inference.

**Origin.** Peng et al., "RWKV: Reinventing RNNs for the Transformer Era" (2023), developed initially as an open-source community project.

**Core mechanism (brief).** RWKV uses a linear attention-like mechanism with an explicit decay term (older information is down-weighted over time in a structured way) that, similarly to S4, can be computed either as a parallelizable form during training or as an efficient recurrence during inference.

**Current status.** A smaller but active open-source lineage, with continued iteration; used in some open-weight model releases valuing its inference efficiency, though it has not reached the mainstream adoption of Mamba-style layers or Gated DeltaNet in the production hybrids described below.

## Linear attention and Gated DeltaNet

**Name & definition.** **Linear attention** is a family of attention replacements that drop the softmax so that attention can be rewritten as a recurrence over a fixed-size matrix "memory." **Gated DeltaNet** (GDN) is the 2024 member of this family that, by late 2026, has become one of the most prominent linear-attention layers in production LLMs: it combines a **delta-rule** write (overwrite what the memory currently stores for a key, rather than just adding to it) with a **gated decay** (a learned, input-dependent forgetting factor).

**Origin.** Linear attention: Katharopoulos et al., "Transformers are RNNs" (2020). The delta-rule view: Schlag, Irie, Schmidhuber, "Linear Transformers Are Secretly Fast Weight Programmers" (2021), made practical to train at scale by Yang et al. (2024, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length"). Gating/decay: Gated Linear Attention (Yang et al., 2023) and Mamba-2 (2024). Gated DeltaNet itself: Yang, Kautz, Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule" (2024; ICLR 2025).

### Step 1 — removing the softmax turns attention into a recurrence

In ordinary causal attention, the output for position t is a softmax-weighted sum of all earlier value vectors. The softmax is what forces you to keep every past key and value around (the KV cache): you cannot compute the normalization without revisiting all of them. If you drop the softmax (or replace it with a simple feature map), the output becomes

```
oₜ = Σ_{i ≤ t} vᵢ (kᵢ · qₜ)  =  ( Σ_{i ≤ t} vᵢ kᵢᵀ ) · qₜ  =  Sₜ · qₜ
where   Sₜ = S_{t-1} + vₜ kₜᵀ
```

Walkthrough: the trick is just re-associating a matrix product. Instead of comparing the new query against every stored key, you keep a running **state matrix** Sₜ (size d_v × d_k per head — say 128 × 128) that is the sum of outer products vᵢkᵢᵀ of every key/value pair seen so far. Reading from it is one matrix-vector product with the query. Each new token updates S by adding one outer product. The crucial consequence: **the state is the same size whether you have read 100 tokens or 1,000,000** — per-token compute and memory are constant, exactly the RNN-style property SSMs were after. Think of S as a tiny associative memory ("fast weights") that maps keys to values.

**The problem with plain linear attention.** The update only ever *adds*. Nothing is ever erased, so as the sequence grows, many different key/value pairs get superimposed in the same fixed-size matrix and interfere with each other — retrieval gets blurrier the longer the context. Plain linear attention historically underperformed softmax attention noticeably on language modeling for this reason.

### Step 2 — two fixes: gated decay and the delta rule

**Fix A: gated decay (forgetting).** Multiply the old state by a data-dependent factor αₜ between 0 and 1 before adding the new pair: `Sₜ = αₜ S_{t-1} + vₜkₜᵀ`. When αₜ is near 1 the memory is preserved; when it is small, the memory is largely cleared (e.g., at a topic boundary). This is the "selectivity" idea from Mamba expressed in linear-attention form, and it is essentially what Mamba-2 and Gated Linear Attention do.

**Fix B: the delta rule (targeted overwriting).** Before writing, *read* what the memory currently returns for the new key, and write only the correction:

```
Sₜ = S_{t-1} − βₜ (S_{t-1} kₜ − vₜ) kₜᵀ
```

Walkthrough: S_{t-1}kₜ is "what the memory currently thinks the value for key kₜ is." Subtracting the desired value vₜ gives the error; the update nudges the memory to reduce that error, with βₜ ∈ (0, 1) (computed from the input) acting like a per-token write strength. If you have seen this shape before, it is the classic Widrow-Hoff "delta rule" — literally one step of gradient descent on the squared error ‖S kₜ − vₜ‖², performed *inside the forward pass*. So instead of piling a new memory on top of an old one for the same key, the delta rule *replaces* it — which is what makes it much better at precise key-value recall than additive updates.

### Step 3 — Gated DeltaNet = both together

```
Sₜ = αₜ · S_{t-1} · (I − βₜ kₜ kₜᵀ)  +  βₜ vₜ kₜᵀ        (keys normalized to unit length)
oₜ = Sₜ · qₜ
```

Walkthrough: this is the same delta-rule update rewritten in matrix form, with the gated decay αₜ applied on top. The factor (I − βₜkₜkₜᵀ) erases (part of) whatever was stored along direction kₜ; the second term writes the new value there; αₜ lets the layer additionally fade out the whole memory when the content calls for it. The two mechanisms are complementary: decay is good at *wholesale* forgetting, the delta rule at *precise* replacement of one association. Like Mamba, training does not run this loop token by token — chunk-wise parallel algorithms process blocks of tokens with dense matrix multiplications so GPUs stay efficient; at inference it runs as a true recurrence with a fixed-size state.

**Why it mattered.** Gated DeltaNet was reported to outperform both Mamba-2 and plain DeltaNet at equal size on language modeling and, especially, on recall-heavy tasks — the category where fixed-state models had been weakest. Refinements followed quickly; for example Moonshot's **Kimi Delta Attention** (2025) is a refined Gated DeltaNet.

**Strengths & limitations.**
- Strengths: constant per-token inference cost and memory (no KV cache growth); parallelizable chunk-wise training; among the strongest recall results reported for fixed-state layers.
- Limitations: the state is still a *fixed-size, lossy* compression of the past — a model cannot reliably copy an arbitrary passage from 500,000 tokens ago if that detail was overwritten. Exact retrieval from very long context remains attention's advantage, which is precisely why the production designs below are hybrids rather than pure GDN stacks. Kernel/tooling maturity also lags far behind attention's.

## The hybrid architecture wave (2024-2026)

**Name & definition.** A **hybrid architecture** stacks two kinds of sequence-mixing layer in one model: mostly cheap fixed-state layers (an SSM such as Mamba-2, or a linear-attention layer such as Gated DeltaNet), plus a minority of ordinary full-attention layers at regular intervals. (The feedforward/MoE sub-layers are unchanged — see [mixture-of-experts.md](mixture-of-experts.md).)

**The intuition.** The fixed-state layers do most of the per-token work cheaply and carry a compressed running summary of the context; the occasional full-attention layers keep a real KV cache and can look up any earlier token exactly, covering the precise-retrieval weakness of fixed-state layers. Because only the attention layers store a KV cache, KV memory shrinks roughly in proportion to the fraction of layers that are attention — with 1 attention layer in every 4, roughly a quarter of what an all-attention model of the same depth and attention configuration would need (a simple-arithmetic approximation; real savings depend on the rest of the design).

**A 3:1 layer schedule, pictured.** The pattern reported for Qwen3.5-397B-A17B: 60 layers = 15 repetitions of [3 Gated DeltaNet layers + 1 gated full-attention layer], each followed by an MoE feedforward sub-layer.

```
 layer:    1     2     3     4     5     6     7     8    ...   57    58    59    60
 mixer:   GDN   GDN   GDN  ATTN   GDN   GDN   GDN  ATTN   ...  GDN   GDN   GDN  ATTN
 FFN:     MoE   MoE   MoE   MoE   MoE   MoE   MoE   MoE   ...  MoE   MoE   MoE   MoE
         └───────── block 1 ────────┘└───────── block 2 ────────┘   └──────── block 15 ───────┘

 per-layer inference memory:
   GDN  → fixed d_v × d_k state per head (does not grow with context length)
   ATTN → KV cache that grows linearly with context length (15 of the 60 layers)
```

**The 2024-2026 timeline.**
- **Jamba** (AI21 Labs, 2024) — an early openly released hybrid, interleaving Mamba layers with a minority of attention layers (plus MoE).
- **Nemotron-H / Nemotron Nano 2** (NVIDIA, 2025) — production hybrid Mamba-2 + attention model families.
- **Qwen3-Next** (Alibaba Qwen, September 2025) — Gated DeltaNet and *gated* full attention (see the gated-attention discussion in [transformer-architecture.md](transformer-architecture.md)) in a roughly 3:1 ratio, combined with an ultra-sparse MoE.
- **Kimi Linear** (Moonshot AI, October 2025) — Kimi Delta Attention hybridized with Multi-head Latent Attention (MLA — a compressed-KV attention variant; see [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md)) layers at roughly 3:1.
- **NVFP4 pretraining** (NVIDIA, "Pretraining LLMs with NVFP4," arXiv 2509.25149, 2025; revised March 2026) — the model used to demonstrate 4-bit pretraining was itself a **12B hybrid Mamba-Transformer** trained on 10T tokens, reaching 62.58% on MMLU-Pro versus 62.62% for an FP8 baseline (reported). Notable here as evidence that hybrids are now the *default test vehicle* for other frontier techniques, not an exotic side-branch; the precision format itself is covered in [quantization.md](../06-inference-optimization/quantization.md).
- **Qwen3.5** (early 2026) — brought the 3:1 Gated DeltaNet : gated-attention schedule into Qwen's *main* model line, e.g., Qwen3.5-397B-A17B (reported), making it one of the clearest cases of a linear-attention hybrid as a flagship rather than an experimental side release.

**A competing route to the same goal.** Hybrids attack attention's long-context cost by *replacing most attention layers*. The other 2025-2026 route keeps attention everywhere but makes each attention layer cheaper — compressing the KV cache (MLA) or attending only to a selected subset of past tokens (DeepSeek Sparse Attention, and DeepSeek-V4's compressed sparse attention, reported). DeepSeek-V4 also calls its design "hybrid attention," but it interleaves two *compressed/sparse attention* variants rather than linear-attention layers. These are covered in [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md). Which route wins — or whether frontier models combine both — is unresolved.

## Current adoption status vs. Transformers — an honest assessment

As of late 2026 the picture has shifted meaningfully from "promising research direction" to "production option," but in a specific form:

- **Hybrids are now in production flagship models.** Linear-attention/SSM hybrids ship in openly released frontier-class models from several major labs (Qwen3.5's main line, Kimi Linear, NVIDIA's Nemotron-H family), and hybrids are being used as the base for other frontier experiments (NVFP4 pretraining). The roughly 3:1 linear-to-full-attention ratio recurs across independent teams, which is a meaningful signal — though it is a widely-copied empirical choice, not a derived optimum.
- **Pure SSM / pure linear-attention models remain rare at the frontier.** Every prominent production design keeps *some* full attention layers. The fixed-state compression limitation (imprecise exact recall over very long contexts) has not been engineered away, and the field's revealed preference is to patch it with attention rather than to do without attention.
- **All-attention Transformers are still the majority.** Many frontier models — and, as far as public information goes, most closed frontier models, whose architectures are generally undisclosed — are still all-attention decoder-only Transformers, made cheaper with GQA, MLA, sliding-window, or sparse attention rather than with recurrent layers. Tooling, kernels, and training recipes for attention remain far more mature.
- **Where the case is strongest.** As before, the efficiency advantage matters most at very long context lengths and in high-throughput decoding, where KV-cache size dominates serving cost; at short-to-moderate contexts the gain is much smaller.

Whether hybrids become the dominant frontier template, or whether cheaper-attention designs (sparse/compressed attention) make them unnecessary, is genuinely open — see the explicitly-labeled speculation in [09-roadmaps/near-term-outlook.md](../09-roadmaps/near-term-outlook.md) rather than any confident claim here.

## Comparison table

| Architecture | Year | Key Innovation | Still Relevant (2026)? |
|---|---|---|---|
| S4 | 2021/2022 | Structured linear recurrence, convolution-equivalent training | Influential; largely superseded by Mamba for new work |
| Mamba | 2023 | Input-dependent ("selective") state space parameters | Yes — research and, via Mamba-2, production hybrids |
| RWKV | 2023 | Linear-attention-like recurrence, community-driven | Niche — smaller but active open-source lineage |
| Mamba-2 | 2024 | State space duality: selective SSM = linear attention with decay | Yes — SSM layer in production hybrids (e.g., Nemotron-H) |
| Gated DeltaNet | 2024/2025 | Delta-rule overwrite + gated decay on a fixed-size matrix state | Yes — linear-attention layer in Qwen3-Next/Qwen3.5; refined as Kimi Delta Attention |
| Hybrid (linear/SSM + full attention) | 2024-2026 | Mostly fixed-state layers, ~1 in 4 full attention (common ratio) | Yes — in production flagship open-weight models as of late 2026 |

## Relationship to other algorithms

- SSMs and linear attention are proposed as an efficiency-motivated alternative to the self-attention mechanism in [transformer-architecture.md](transformer-architecture.md), directly addressing that mechanism's O(n²) scaling; the full-attention layers inside modern hybrids often use the *gated attention* variant described there.
- The linear-recurrence-with-parallel-training trick echoes, and improves substantially on, the sequential-training limitation of vanilla RNNs/LSTMs (see [rnn-lstm-gru.md](rnn-lstm-gru.md)); the delta rule is the same error-correcting update as one step of gradient descent (see [optimization-algorithms.md](../01-foundations/optimization-algorithms.md)).
- Hybrid designs are directly relevant to the tradeoffs discussed in [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md), since one of their main selling points is shrinking the KV cache; that file also covers the competing "cheaper attention" route (MLA, sparse attention).
- Production hybrids pair the fixed-state layers with very sparse Mixture-of-Experts feedforward layers — see [mixture-of-experts.md](mixture-of-experts.md).
- 4-bit (NVFP4) pretraining, demonstrated on a hybrid model, is covered in [quantization.md](../06-inference-optimization/quantization.md).

## Sources

- Gu, Goel, Ré, "Efficiently Modeling Long Sequences with Structured State Spaces" (2021, released as preprint; 2022 conference version) [S4]
- Gu, Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- Dao, Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (2024) [Mamba-2]
- Peng et al., "RWKV: Reinventing RNNs for the Transformer Era" (2023)
- Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (2020)
- Schlag, Irie, Schmidhuber, "Linear Transformers Are Secretly Fast Weight Programmers" (2021) [delta rule in linear attention]
- Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient Training" (2023)
- Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length" (2024) [DeltaNet]
- Yang, Kautz, Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule" (2024; ICLR 2025) [Gated DeltaNet]
- Lieber et al. (AI21 Labs), "Jamba: A Hybrid Transformer-Mamba Language Model" (2024)
- NVIDIA, Nemotron-H and Nemotron Nano 2 technical reports (2025) [hybrid Mamba-2 + attention]
- Qwen Team, Qwen3-Next release (September 2025) [Gated DeltaNet + gated attention, ~3:1; ultra-sparse MoE]
- Qwen Team, Qwen3.5 release (early 2026) [3:1 Gated DeltaNet : gated attention in the main line; Qwen3.5-397B-A17B layer layout as reported]
- Moonshot AI, "Kimi Linear" technical report (October 2025) [Kimi Delta Attention + MLA hybrid, ~3:1]
- NVIDIA, "Pretraining LLMs with NVFP4" (arXiv 2509.25149, 2025; v2 March 2026) [12B hybrid Mamba-Transformer, 10T tokens; MMLU-Pro figures as reported]
