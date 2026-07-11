# State Space Models

State Space Models (SSMs) are a family of sequence architectures proposed as an alternative to attention, aiming to keep the long-range modeling ability of Transformers (see [transformer-architecture.md](transformer-architecture.md)) while avoiding attention's quadratic compute cost. This file is honest, per this repository's quality bar, that as of 2026 this remains a minority, still-maturing approach rather than a proven Transformer replacement.

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

## RWKV

**Name & definition.** RWKV (an acronym derived from the four main parameter matrices in its recurrence: Receptance, Weight, Key, Value) is another linear-recurrence-based architecture, developed with an explicit design goal of combining Transformer-like parallelizable training with RNN-like constant-memory inference.

**Origin.** Peng et al., "RWKV: Reinventing RNNs for the Transformer Era" (2023), developed initially as an open-source community project.

**Core mechanism (brief).** RWKV uses a linear attention-like mechanism with an explicit decay term (older information is down-weighted over time in a structured way) that, similarly to S4, can be computed either as a parallelizable form during training or as an efficient recurrence during inference.

**Current status.** A smaller but active open-source lineage, with continued iteration; used in some open-weight model releases valuing its inference efficiency, though it has not reached the mainstream frontier-lab adoption of Mamba-style SSMs or standard attention-based Transformers.

## Current adoption status vs. Transformers — an honest assessment

As of 2026, state space models remain a **minority, still-experimental approach relative to Transformers**, not a proven replacement — this is worth stating plainly rather than overselling the trend. Several major labs have published research and some production experimentation with SSM and SSM-Transformer **hybrid** architectures (combining SSM layers, for their efficiency on long stretches of sequence, with a smaller number of full attention layers, for their stronger content-based, any-to-any modeling power) — hybrid designs are, if anything, a more consistently reported direction than pure-SSM architectures replacing attention outright. The practical case for SSMs is strongest specifically for very long context lengths, where attention's quadratic cost is most punishing; for shorter-to-moderate context lengths (which cover a large fraction of real-world usage), the efficiency advantage is much less pronounced, and Transformers' more mature tooling, more extensively battle-tested training recipes, and stronger empirical track record at the largest scales keep them the default choice for most frontier general-purpose models. Whether SSMs (or SSM-attention hybrids) become a larger share of frontier architectures over the next few years is a genuinely open question — see the honest, explicitly-labeled speculation in [09-roadmaps/near-term-outlook.md](../09-roadmaps/near-term-outlook.md) rather than any confident claim here.

## Comparison table

| Architecture | Year | Key Innovation | Still Relevant (2026)? |
|---|---|---|---|
| S4 | 2021/2022 | Structured linear recurrence, convolution-equivalent training | Influential; largely superseded by Mamba for new work |
| Mamba | 2023 | Input-dependent ("selective") state space parameters | Yes — active research and some production use, esp. for long context |
| RWKV | 2023 | Linear-attention-like recurrence, community-driven | Niche — smaller but active open-source lineage |

## Relationship to other algorithms

- SSMs are proposed as an efficiency-motivated alternative to the self-attention mechanism in [transformer-architecture.md](transformer-architecture.md), directly addressing that mechanism's O(n²) scaling.
- The linear-recurrence-with-parallel-training trick echoes, and improves substantially on, the sequential-training limitation of vanilla RNNs/LSTMs (see [rnn-lstm-gru.md](rnn-lstm-gru.md)).
- SSM-Transformer hybrid designs are directly relevant to the tradeoffs discussed in [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md), since one of SSMs' main selling points is avoiding a large KV cache altogether.

## Sources

- Gu, Goel, Ré, "Efficiently Modeling Long Sequences with Structured State Spaces" (2021, released as preprint; 2022 conference version) [S4]
- Gu, Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- Peng et al., "RWKV: Reinventing RNNs for the Transformer Era" (2023)
