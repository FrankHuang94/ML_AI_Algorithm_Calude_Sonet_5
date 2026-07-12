# Changelog

## v2 — Opus review-and-expand pass

A full second pass over every content file to correct any technical imprecision, deepen explanations, and make the material more understandable. Total content grew from ~69,600 to ~78,000+ words, and hand-drawn visuals were added throughout — not as padding, but to make the hardest-to-picture mechanisms concrete.

**Correctness / precision fixes**
- Corrected self-attention's "permutation-invariant" to the precise term **permutation-equivariant** (with the distinction explained) in the cornerstone Transformer file.
- Clarified the classical-vs-normalized (EMA) **momentum** formulations, so readers aren't confused when a paper's equation doesn't match their framework's code.

**Understandability additions (worked examples + ASCII/Mermaid visuals)**
- Concrete numeric walkthroughs added where they most help: a gradient-descent step, cross-entropy loss values, an attention weighted-sum, a Q-learning update (with TD error), LoRA's parameter savings, INT4 quantization "ladder rungs."
- New diagrams: optimizer family-tree, train/validation loss curve (overfitting), loss-function shapes, decision-boundary shapes per classifier, bagging-vs-boosting, k-means-vs-DBSCAN on concentric rings, EM loop, attention data-flow, convolution edge-detector, LSTM conveyor-belt, MoE routing, message passing, GAN adversarial loop, diffusion forward/reverse, reparameterization gradient-flow, sampling strategies, RLHF pipeline, parallelism types, KV-cache recompute, speculative draft-verify, agent-environment loop, PPO clipping.

**Content brought current to mid-2026 (genuine gaps filled)**
- Optimizers: **Muon** (2024) and the **WSD** learning-rate schedule.
- Regularization: **GroupNorm**, **QK-norm**, stochastic depth, and an **implicit regularization** section (SGD noise, flat minima).
- Losses: **Huber loss**, and the **forward-vs-reverse KL** (mode-covering vs mode-seeking) distinction.
- Transformers: the **feedforward sub-layer** (previously only in the block diagram), gated activations (GELU/SwiGLU), and the train-parallel-vs-generate-sequential clarification tied to the KV cache.
- MoE: fine-grained + shared experts (DeepSeek-style).
- Diffusion: **flow matching / rectified flow**, DDIM, and distillation-based fast samplers.
- VAEs: posterior collapse, β-VAE, and **VQ-VAE** (the bridge to autoregressive image/audio generation).
- Training: a **tokenization/BPE** section, **GRPO** (value-model-free RL for reasoning models), and the DoRA/LoRA family.
- Classical ML: the discriminative-vs-generative framing made explicit up front.

**Housekeeping**
- Glossary grew to ~105 terms, kept alphabetized and in sync with every addition; term-count table refreshed.
- Master comparison table updated (Muon, flow matching) for cross-file consistency.
- Re-verified: all internal links resolve, all 14 Mermaid diagrams are well-formed, every file has at least one visual.

## Build complete — ML/AI Algorithm Knowledge Base v1

The full repository build described in the original scope is complete: a comprehensive, indexed, markdown-based reference covering ML/AI algorithms from classical statistical methods through frontier 2026 LLM training and inference practice, plus their development history and near/long-term outlook.

**Total content:** ~69,600 words across 45 content files (excluding this changelog, contributing guide, and progress scratch file) — comfortably past the 50,000-word floor set in the original scope, achieved through genuine topic coverage rather than padding.

### Structure delivered

- **`docs/00-overview/`** — scope-and-methodology (including explicit audience calibration), a living alphabetized glossary (~85 terms), and a taxonomy-map Mermaid mindmap tying every section together.
- **`docs/01-foundations/`** — optimization algorithms (gradient descent through AdamW/Lion/Sophia/LAMB), regularization techniques, loss functions.
- **`docs/02-classical-ml/`** — supervised learning, ensemble methods, unsupervised learning, probabilistic models.
- **`docs/03-deep-learning-architectures/`** — CNNs, RNN/LSTM/GRU, the Transformer (cornerstone file, extra depth per the original spec), Mixture of Experts, state space models, graph neural networks.
- **`docs/04-generative-models/`** — GANs, VAEs, diffusion models, autoregressive generation, flow-based models.
- **`docs/05-training-methodology/`** — pretraining strategies, fine-tuning/PEFT (LoRA/QLoRA), RLHF and alignment (full pipeline plus DPO/RLAIF/Constitutional AI), distributed training, curriculum and data strategies.
- **`docs/06-inference-optimization/`** — quantization, pruning/distillation, KV cache and attention optimization (FlashAttention/PagedAttention/GQA), speculative decoding, serving and batching.
- **`docs/07-reinforcement-learning/`** — value-based methods, policy gradient methods (PPO in depth), model-based RL, and a dedicated RL-for-LLMs bridge file.
- **`docs/08-history/`** — four chronological timeline files (1950s-2000s, 2000s-2017, 2017-2023, 2023-present), each with a Mermaid timeline diagram, covering the Perceptron through the current RL-on-verifiable-rewards/agentic-training era.
- **`docs/09-roadmaps/`** — near-term (6-18mo) and long-term (2-5yr) outlook, both with `[Projection]` labeling separating sourced claims from speculation, plus an open-problems file covering catastrophic forgetting, hallucination, interpretability, agentic reliability, compute cost, and evaluation validity.
- **`docs/10-comparison-tables/`** — cross-cutting synthesis only: a master comparison table per major category, a compute-cost-specific comparison, and a practical decision guide with a Mermaid decision tree.
- **`README.md`** — full linked table of contents with a one-line description per file.

### Conventions followed throughout

- Every entry follows the nine-part per-algorithm template (name/definition, origin, mechanism, why it mattered, current status, strengths/limitations, usage today, relationships, sources) defined in `scope-and-methodology.md`.
- Every file includes at least one Mermaid diagram or a substantial comparison table.
- Every jargon term is defined on first use inline and cross-referenced in the living glossary.
- Historical claims are dated and attributed to specific papers; uncertain dates/attributions are flagged as such rather than stated as fact.
- Roadmap claims are visibly split between sourced/documented direction and `[Projection]`-labeled speculation.
- All 681 internal cross-links were verified to resolve correctly during the final consistency pass.

### Known limitations

- This is a mid-2026 snapshot; the RL-on-verifiable-rewards and agentic-training material in `08-history/timeline-2023-present.md` and `09-roadmaps/` is, by the repository's own stated methodology, the least settled part of the historical record and should be expected to look different in hindsight.
- Benchmark numbers are deliberately given as qualitative comparisons or ranges rather than precise figures, per the "no invented precision" quality bar in `scope-and-methodology.md`.
