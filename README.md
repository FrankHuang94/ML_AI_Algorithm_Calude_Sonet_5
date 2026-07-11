# ML/AI Algorithm Knowledge Base

A comprehensive, indexed, markdown-based reference covering the algorithms and techniques that matter in machine learning and AI — classical ML and deep learning architectures, generative models, training methodology, inference optimization, and reinforcement learning — along with the development history behind each major family and where the field appears to be headed.

**Audience:** written for a reader with a bachelor's degree in computer science (solid programming, standard CS fundamentals, undergraduate linear algebra/calculus/probability) but no graduate-level ML background. Every file defines ML-specific jargon and notation on first use. See [scope-and-methodology.md](docs/00-overview/scope-and-methodology.md) for the full audience calibration and the per-algorithm template every entry follows.

## How to use this repository

- **New here?** Start with [taxonomy-map.md](docs/00-overview/taxonomy-map.md) for a visual overview of how everything relates, then browse by topic below.
- **Hit an unfamiliar term?** Check the living [glossary.md](docs/00-overview/glossary.md) — every jargon term used anywhere in this repository is defined there, alphabetized, with a link to where it's covered in depth.
- **Want the story, not the topic map?** Read [08-history](docs/08-history/) in order — four files, 1950s to present.
- **Want a quick comparison or a decision?** Go straight to [10-comparison-tables](docs/10-comparison-tables/).
- **Curious where things are headed?** [09-roadmaps](docs/09-roadmaps/) — clearly separates sourced claims from labeled `[Projection]` speculation.

## Table of contents

### 00 — Overview
| File | Description |
|---|---|
| [scope-and-methodology.md](docs/00-overview/scope-and-methodology.md) | What this repo covers, the audience calibration, and the nine-part per-algorithm template used throughout |
| [glossary.md](docs/00-overview/glossary.md) | Living, alphabetized glossary of every jargon term used in the repository |
| [taxonomy-map.md](docs/00-overview/taxonomy-map.md) | Visual mindmap of how every algorithm family relates to every other one |

### 01 — Foundations
| File | Description |
|---|---|
| [optimization-algorithms.md](docs/01-foundations/optimization-algorithms.md) | Gradient descent through AdamW, Lion, Sophia, LAMB/LARS, LR schedules, gradient clipping |
| [regularization-techniques.md](docs/01-foundations/regularization-techniques.md) | L1/L2/weight decay, dropout, BatchNorm/LayerNorm/RMSNorm, data augmentation, label smoothing |
| [loss-functions.md](docs/01-foundations/loss-functions.md) | MSE, cross-entropy, KL divergence, contrastive losses, focal loss, perplexity |

### 02 — Classical ML
| File | Description |
|---|---|
| [supervised-learning.md](docs/02-classical-ml/supervised-learning.md) | Linear/logistic regression, decision trees, SVMs, k-NN, Naive Bayes |
| [ensemble-methods.md](docs/02-classical-ml/ensemble-methods.md) | Bagging, Random Forest, boosting, XGBoost/LightGBM/CatBoost, stacking |
| [unsupervised-learning.md](docs/02-classical-ml/unsupervised-learning.md) | k-means, hierarchical clustering, DBSCAN, PCA, t-SNE, UMAP, autoencoders |
| [probabilistic-models.md](docs/02-classical-ml/probabilistic-models.md) | HMMs, Gaussian Mixture Models, the EM algorithm, Bayesian networks |

### 03 — Deep Learning Architectures
| File | Description |
|---|---|
| [cnn-family.md](docs/03-deep-learning-architectures/cnn-family.md) | Convolution/pooling from scratch, LeNet through ResNet and EfficientNet, detection architectures |
| [rnn-lstm-gru.md](docs/03-deep-learning-architectures/rnn-lstm-gru.md) | Vanilla RNNs, the vanishing gradient problem, LSTM/GRU gating, seq2seq, pre-Transformer attention |
| [transformer-architecture.md](docs/03-deep-learning-architectures/transformer-architecture.md) | **Cornerstone file.** Self-attention from first principles, positional encoding/RoPE, encoder/decoder variants, scaling laws |
| [mixture-of-experts.md](docs/03-deep-learning-architectures/mixture-of-experts.md) | Sparse routing, load balancing, Switch Transformer, GShard, Mixtral |
| [state-space-models.md](docs/03-deep-learning-architectures/state-space-models.md) | S4, Mamba, RWKV, and an honest assessment of adoption vs. Transformers |
| [graph-neural-networks.md](docs/03-deep-learning-architectures/graph-neural-networks.md) | Message passing, GCN, GraphSAGE, GAT, molecular/recommendation/knowledge-graph use cases |

### 04 — Generative Models
| File | Description |
|---|---|
| [gans.md](docs/04-generative-models/gans.md) | Adversarial training, mode collapse, DCGAN/StyleGAN, why diffusion displaced GANs |
| [vaes.md](docs/04-generative-models/vaes.md) | Reparameterization trick, the ELBO explained in plain English, relation to diffusion |
| [diffusion-models.md](docs/04-generative-models/diffusion-models.md) | DDPM, score-based modeling, latent diffusion/Stable Diffusion, classifier-free guidance |
| [autoregressive-generation.md](docs/04-generative-models/autoregressive-generation.md) | Next-token prediction, greedy/beam/top-k/nucleus sampling, degeneration and mitigations |
| [flow-based-models.md](docs/04-generative-models/flow-based-models.md) | Normalizing flows, RealNVP/Glow, and an honest account of why this family stayed niche |

### 05 — Training Methodology
| File | Description |
|---|---|
| [pretraining-strategies.md](docs/05-training-methodology/pretraining-strategies.md) | Masked vs. causal pretraining, CLIP-style contrastive pretraining, scaling laws in practice |
| [finetuning-and-peft.md](docs/05-training-methodology/finetuning-and-peft.md) | Full fine-tuning, LoRA/QLoRA, adapters, prefix/prompt tuning, cost comparison |
| [rlhf-and-alignment.md](docs/05-training-methodology/rlhf-and-alignment.md) | The full RLHF pipeline (SFT → reward model → PPO), DPO, RLAIF, Constitutional AI |
| [distributed-training.md](docs/05-training-methodology/distributed-training.md) | Data/tensor/pipeline parallelism, ZeRO stages, FSDP, communication bottlenecks |
| [curriculum-and-data-strategies.md](docs/05-training-methodology/curriculum-and-data-strategies.md) | Data curation, deduplication, curriculum learning, synthetic data, mixture weighting |

### 06 — Inference Optimization
| File | Description |
|---|---|
| [quantization.md](docs/06-inference-optimization/quantization.md) | INT8/INT4 mechanics, PTQ vs. QAT, GPTQ, AWQ, accuracy/latency/memory tradeoffs |
| [pruning-and-distillation.md](docs/06-inference-optimization/pruning-and-distillation.md) | Magnitude pruning, structured vs. unstructured, knowledge distillation |
| [kv-cache-and-attention-optimization.md](docs/06-inference-optimization/kv-cache-and-attention-optimization.md) | The KV cache from scratch, FlashAttention, PagedAttention, MQA/GQA |
| [speculative-decoding.md](docs/06-inference-optimization/speculative-decoding.md) | Draft-and-verify generation, why it's exact (not approximate), Medusa, lookahead decoding |
| [serving-and-batching.md](docs/06-inference-optimization/serving-and-batching.md) | Static vs. continuous batching, throughput/latency tradeoffs, vLLM/TensorRT-LLM |

### 07 — Reinforcement Learning
| File | Description |
|---|---|
| [value-based-methods.md](docs/07-reinforcement-learning/value-based-methods.md) | MDPs, Q-learning, DQN, Double DQN, Dueling DQN, experience replay |
| [policy-gradient-methods.md](docs/07-reinforcement-learning/policy-gradient-methods.md) | REINFORCE, actor-critic, A3C, TRPO, PPO (in depth), GAE |
| [model-based-rl.md](docs/07-reinforcement-learning/model-based-rl.md) | World models, MuZero, sample efficiency vs. model-free methods |
| [rl-for-llms.md](docs/07-reinforcement-learning/rl-for-llms.md) | Bridges general RL theory to LLM training; RL on verifiable rewards for reasoning |

### 08 — History
| File | Description |
|---|---|
| [timeline-1950s-2000s.md](docs/08-history/timeline-1950s-2000s.md) | Perceptron, the first AI winter, backpropagation, LeNet, LSTM, the 2006 "deep learning" moment |
| [timeline-2000s-2017.md](docs/08-history/timeline-2000s-2017.md) | AlexNet/ImageNet, word2vec, DQN, GANs, seq2seq+attention, ResNet, AlphaGo, the Transformer |
| [timeline-2017-2023.md](docs/08-history/timeline-2017-2023.md) | BERT/GPT-1, GPT-2/3, scaling laws, CLIP/RoPE, Chinchilla, InstructGPT, ChatGPT, LLaMA, Mixtral |
| [timeline-2023-present.md](docs/08-history/timeline-2023-present.md) | Multimodal convergence, long-context scaling, RL on verifiable rewards, agentic training |

### 09 — Roadmaps
| File | Description |
|---|---|
| [near-term-outlook.md](docs/09-roadmaps/near-term-outlook.md) | 6-18 month projections: inference cost, context length, agentic training, MoE adoption |
| [long-term-outlook.md](docs/09-roadmaps/long-term-outlook.md) | 2-5 year projections: attention alternatives, continual learning, test-time compute scaling |
| [open-problems.md](docs/09-roadmaps/open-problems.md) | Catastrophic forgetting, hallucination, interpretability, agentic reliability, compute cost, evaluation validity |

### 10 — Comparison Tables
| File | Description |
|---|---|
| [algorithm-comparison-master.md](docs/10-comparison-tables/algorithm-comparison-master.md) | One master table per category: optimizers, architectures, generative models, PEFT, inference optimization |
| [compute-cost-comparison.md](docs/10-comparison-tables/compute-cost-comparison.md) | Training and inference cost levers, how they stack, and the training-vs-serving-cost tension |
| [when-to-use-what.md](docs/10-comparison-tables/when-to-use-what.md) | Practical decision guidance with a Mermaid decision tree |

## Repository structure

```
/README.md                     <- you are here
/docs/
  00-overview/                 <- methodology, glossary, taxonomy map
  01-foundations/               <- optimization, regularization, loss functions
  02-classical-ml/              <- supervised/unsupervised/ensemble/probabilistic
  03-deep-learning-architectures/ <- CNNs, RNNs, Transformers, MoE, SSMs, GNNs
  04-generative-models/         <- GANs, VAEs, diffusion, autoregressive, flows
  05-training-methodology/      <- pretraining, PEFT, RLHF, distributed training, data
  06-inference-optimization/    <- quantization, pruning, KV cache, speculative decoding, serving
  07-reinforcement-learning/    <- value-based, policy gradient, model-based, RL-for-LLMs
  08-history/                   <- four chronological timeline files, 1950s-present
  09-roadmaps/                  <- near-term/long-term outlook, open problems
  10-comparison-tables/         <- cross-cutting synthesis tables and decision guidance
/assets/diagrams/               <- (Mermaid diagrams are inlined in-file; reserved for future exports)
/CONTRIBUTING.md
/CHANGELOG.md
/PROGRESS.md                    <- build scratch file, not part of the published reference
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the per-algorithm template, sourcing conventions, and glossary maintenance expectations this repository follows.
