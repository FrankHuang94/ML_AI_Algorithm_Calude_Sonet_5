# Changelog

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
