# Algorithm Comparison Master Table

This file is pure synthesis — it introduces no algorithms not already covered elsewhere in this repository. Each table below pulls together one major category, using a consistent column structure, so you can compare across a whole category at a glance rather than reading file by file. Every row links back to the file where that algorithm is covered in full depth (origin, mechanism, strengths/limitations, sources).

## Optimizers

Full treatment: [optimization-algorithms.md](../01-foundations/optimization-algorithms.md)

| Algorithm | Year | Key Innovation | Compute Profile | Still Relevant (late 2026)? | Typical Use Case |
|---|---|---|---|---|---|
| SGD (mini-batch) | 1950s (theory) / 1980s (NN use) | Base iterative minimization via mini-batch gradients | Lowest memory (no extra state) | Yes — universal substrate | Base case; rarely used unadorned at scale |
| Momentum / NAG | 1964 / 1983 | Smooths trajectory using gradient history | Low (one extra buffer) | Component of Adam; rare standalone | Smaller-scale training |
| Adagrad | 2011 | Per-parameter adaptive learning rate | Moderate (one extra buffer, grows) | Superseded (monotonic decay flaw) | Sparse convex problems |
| RMSProp | 2012 | Decaying-average adaptive scaling | Moderate | Rare standalone; Adam ancestor | Superseded by Adam/AdamW |
| Adam | 2015 | Momentum + adaptive scaling combined | Moderate-high (2 extra buffers/param) | Common outside LLM pretraining | General deep learning |
| AdamW | 2019 | Decoupled weight decay | Moderate-high | Dominant default | LLM pretraining/fine-tuning |
| Lion | 2023 | Sign-based update, less optimizer memory | Lower than Adam (1 buffer) | Real but minority adoption | Large-scale pretraining (some labs) |
| Sophia | 2023 | Lightweight second-order estimate | Moderate-high | Promising, limited adoption | LLM pretraining (experimental) |
| Muon (incl. MuonClip) | 2024 / 2025 | Orthogonalized matrix-aware updates (+ QK-clip for logit stability) | Lower than Adam | Yes — validated at trillion-parameter scale (Kimi K2; DeepSeek-V4, reported); credible AdamW successor | Large-scale pretraining |
| LAMB / LARS | 2017 / 2019 | Layer-wise LR rescaling for large batches | Moderate-high | Standard for large-batch phases | Large-batch distributed pretraining |
| L-BFGS / Natural Gradient | Pre-deep-learning | True curvature-aware steps | Very high (near-infeasible at scale) | Rare at LLM scale | Small/medium-scale optimization |

## Architectures

Full treatment: [03-deep-learning-architectures](../03-deep-learning-architectures/)

| Algorithm | Year | Key Innovation | Compute Profile | Still Relevant (late 2026)? | Typical Use Case |
|---|---|---|---|---|---|
| CNN (ResNet-era) | 1998-2015 | Convolution + pooling + skip connections | Efficient, data-efficient | Yes | Vision, resource-constrained deployment |
| RNN/LSTM/GRU | 1990-2014 | Recurrent hidden state with gating | Sequential (slow to parallelize) | Niche | Smaller-scale/on-device sequence tasks |
| Transformer (decoder-only) | 2017-2018 | Self-attention, fully parallelizable | High (quadratic in sequence length) | Yes — dominant | General-purpose LLMs |
| Mixture of Experts | 2017-2021 (fine-grained / aux-loss-free: 2024) | Sparse routing decouples params from compute | High total params, lower active compute | Yes — standard at frontier scale (1T+ total, a few % active) | Largest frontier and open-weight models |
| State Space Models (Mamba) | 2021-2023 | Linear-time selective recurrence | Efficient at long context | Minority as a pure architecture; lives on inside hybrids | Long-context research, on-device, hybrids |
| Hybrid linear-attention models (Gated DeltaNet + attention) | 2024-2025 | Most layers keep a fixed-size recurrent state; a few full-attention layers keep precise recall | Near-linear in context length; small KV cache | Yes — in production flagships (Qwen3.5, Kimi Linear, Nemotron-H) | Long-context open-weight LLMs |
| Graph Neural Networks | 2017-2018 | Message passing over graph structure | Efficient for graph-shaped data | Yes — domain-specific | Molecules, recommendation, knowledge graphs |

## Generative model families

Full treatment: [04-generative-models](../04-generative-models/)

| Algorithm | Year | Key Innovation | Compute Profile | Still Relevant (late 2026)? | Typical Use Case |
|---|---|---|---|---|---|
| GANs | 2014 | Adversarial generator/discriminator training | Fast sampling (single pass) | Niche (face gen/editing, real-time) | Superseded by diffusion for open-domain images |
| VAEs | 2013 | Probabilistic latent space, ELBO training | Fast sampling; stable training | Superseded standalone; essential as a component | Compression stage inside latent diffusion |
| Diffusion models | 2015/2020 | Iterative denoising, classifier-free guidance | Slow sampling (multi-step) | Yes — dominant | Image/video/audio generation |
| Flow matching / rectified flow | 2022-2023 | Velocity-field training; straighter, fewer-step paths | Faster sampling than classic diffusion | Yes — increasingly used for new image/video models | Image/video generation |
| Autoregressive generation | Concept: classical; LLM-scale: 2018+ | Next-token prediction + sampling strategies | Sequential generation (mitigated by speculative decoding) | Yes — dominant | Text generation, general-purpose LLMs |
| Diffusion language models (dLLMs) | 2021-2025 | Parallel iterative unmasking of text tokens | Many tokens per step, fewer sequential steps | Emerging — speed-focused products and active research | Low-latency text/code generation |
| Flow-based models | 2016-2018 | Exact likelihood via invertible transforms | Restricted architecture, less flexible | Niche | Exact-likelihood/density-estimation use cases |

## PEFT methods

Full treatment: [finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md)

| Algorithm | Year | Key Innovation | Compute Profile | Still Relevant (late 2026)? | Typical Use Case |
|---|---|---|---|---|---|
| Full fine-tuning | — | Update all parameters | Highest (full optimizer state) | Yes, when justified | Maximum task performance |
| Adapters | 2019 | Small inserted bottleneck layers | Low trainable params; adds inference latency | Established, less common than LoRA | Early PEFT baseline |
| Prefix/prompt tuning | 2021 | Learned soft context vectors | Very low trainable params; adds inference latency | Niche, variable performance | Extremely constrained fine-tuning budgets |
| LoRA | 2021 | Low-rank additive weight update | Low trainable params; no added latency if merged | Yes — standard default | Most production fine-tuning |
| QLoRA | 2023 | LoRA + quantized frozen base | Lowest fine-tuning memory footprint | Yes — standard for constrained hardware | Consumer/limited-hardware fine-tuning |

## Inference optimization techniques

Full treatment: [06-inference-optimization](../06-inference-optimization/)

| Algorithm | Year | Key Innovation | Compute Profile | Still Relevant (late 2026)? | Typical Use Case |
|---|---|---|---|---|---|
| Quantization (GPTQ/AWQ) | 2022-2023 | Low-bit weight representation | Cuts memory ~2-4x+ | Yes — standard | Cost-sensitive deployment |
| Pruning (structured) | 2015+ | Removes whole neurons/channels/heads | Real speedup on standard hardware | Yes, situationally | Latency-constrained deployment |
| Knowledge distillation | 2015 | Smaller student imitates larger teacher | Produces a genuinely smaller model | Yes — widely used | "Mini"/fast model variants |
| FlashAttention | 2022 | I/O-aware exact attention computation | Faster, same exact result | Yes — near-universal | All Transformer training/inference |
| PagedAttention | 2023 | OS-paging-style KV cache memory management | Higher achievable batch concurrency | Yes — standard in serving systems | High-throughput LLM serving |
| MQA / GQA | 2019 / 2023 | Shared K/V projections across heads | Shrinks KV cache substantially | Yes — GQA is a common default | Long-context, high-concurrency serving |
| Multi-head Latent Attention (MLA) | 2024 | Cache one compressed latent per token, not per-head K/V | Much smaller KV cache than GQA at similar quality | Yes — DeepSeek and Kimi model lines | Long-context MoE serving |
| Sparse / compressed attention (DSA, CSA/HCA) | 2025 / 2026 | Indexer selects top-k (compressed) past entries per query | Sub-quadratic attention at 100K-1M tokens | Yes — growing (DeepSeek-V3.2, V4) | Million-token contexts |
| Low-precision floating point (FP8, MXFP4, NVFP4) | 2022-2025 | Block-scaled 8-/4-bit floats with hardware support | 2-4x less memory/bandwidth than BF16 | Yes — FP8 standard; FP4 spreading to training and serving | Training and serving on recent accelerators |
| Speculative decoding (incl. EAGLE-3, MTP drafters) | 2023 / 2025 | Draft-then-verify generation | Fewer expensive sequential model calls | Yes — standard in major serving stacks | Latency-sensitive generation, no quality cost |
| Continuous batching | 2022 | Dynamically refill batch slots | Higher throughput than static batching | Yes — standard | High-throughput LLM serving |
| Prefill/decode disaggregation + prefix caching | 2024-2025 | Separate GPU pools for compute-bound prefill and bandwidth-bound decode; reuse KV for shared prefixes | Better utilization and latency control at scale | Yes — mainstream at large deployments | Large-scale serving, agents with long shared prompts |

## LLM post-training (RL and preference) methods

Full treatment: [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md) and [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)

| Algorithm | Year | Key Innovation | Compute Profile | Still Relevant (late 2026)? | Typical Use Case |
|---|---|---|---|---|---|
| PPO-based RLHF | 2017 / 2022 | Clipped policy updates against a learned reward model | High (policy + value model + reward model + reference) | Yes, but less dominant | Preference alignment |
| DPO | 2023 | Preference optimization without an explicit reward model or RL loop | Low (supervised-style) | Yes — common for preference tuning | Chat/style alignment |
| GRPO | 2024-2025 | Group-relative advantages; no value model | Moderate (many samples per prompt) | Yes — default family for reasoning RL | RL on verifiable rewards (math, code) |
| DAPO / Dr. GRPO | 2025 | Fixes GRPO's clipping, sampling and normalization biases | Same as GRPO | Yes — common refinements | Long chain-of-thought RL |
| GSPO | 2025 | Sequence-level importance ratio and clipping | Same as GRPO | Yes — notably for MoE RL stability | Large MoE reasoning models |
| On-policy distillation (incl. multi-teacher) | 2023-2024 / 2026 | Student samples its own outputs; teacher(s) give per-token targets | Lower than RL (dense per-token signal) | Yes — widely reported 2026 consolidation pattern | Merging specialist RL models into one |

## Notes on this table's scope

Per this repository's methodology, this file is intentionally a synthesis layer, not a new source of claims — every "Still Relevant (late 2026)?" and "Compute Profile" judgment here is a compressed restatement of the fuller discussion (with sources) in the linked file, and should be read alongside that file's "Current status," "Strengths & limitations," and "Sources" sections rather than as a standalone verdict.

## Relationship to other files

- [compute-cost-comparison.md](compute-cost-comparison.md) goes deeper specifically on training/inference cost tradeoffs.
- [when-to-use-what.md](when-to-use-what.md) turns these comparisons into practical decision guidance.
- [taxonomy-map.md](../00-overview/taxonomy-map.md) is the visual, relationship-oriented counterpart to this file's tabular, comparison-oriented view.
