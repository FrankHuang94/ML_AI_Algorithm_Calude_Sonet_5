# 分类图

这是仓库的可视化入口点：一张显示此处涵盖的每个算法系列如何与其他算法系列相关的地图。在深入研究特定文件之前，使用它来确定自己的方向，或者当您大致知道要查找的内容但不知道确切的文件名时，可以使用它来找到正确的邻居。

## 如何阅读此内容

下面的思维导图按算法所扮演的角色（基础、经典 ML、深度架构、生成模型、训练方法、推理优化、强化学习）对算法进行分组，而不是按时间顺序 - 有关时间顺序视图，请参阅 [08-history](../08-history/)。分支并不是详尽的列表；他们命名了主要条目，以便您可以跳转到正确的文件。完整的清单存在于每个部分的文件中。

```mermaid
mindmap
root((ML/AI Algorithms))
Foundations
Optimization
SGD/Momentum
Adam family
Second-order methods
Regularization
Dropout
Norm layers
Loss functions
Cross-entropy
KL divergence
Classical ML
Supervised
Linear/Logistic Regression
Decision Trees
SVM
Ensembles
Random Forest
Gradient Boosting
Unsupervised
k-means
PCA/UMAP
Probabilistic
HMM
GMM + EM
Deep Architectures
CNNs
ResNet lineage
RNN/LSTM/GRU
Pre-Transformer attention
Transformers
Self-attention
Encoder/decoder variants
Mixture of Experts
State Space Models
Mamba
Graph Neural Networks
Generative Models
GANs
VAEs
Diffusion Models
Autoregressive Generation
Flow-Based Models
Training Methodology
Pretraining
Fine-tuning and PEFT
LoRA/QLoRA
RLHF and Alignment
DPO
Distributed Training
FSDP/ZeRO
Inference Optimization
Quantization
Pruning and Distillation
KV Cache and Attention
FlashAttention
GQA/MQA
Speculative Decoding
Serving and Batching
Reinforcement Learning
Value-Based
DQN
Policy Gradient
PPO
Model-Based RL
RL for LLMs
```

## 横切关系

上面的思维导图按部分进行分组，但 ML 中许多最重要的关系都*跨越*这些部分。这是结缔组织读者最常错过的：

- **优化是一切的基础。** `03-deep-learning-architectures` 中的每个架构和 `04-generative-models` 中的每个生成模型都使用 `01-foundations/optimization-algorithms.md` 的优化器进行训练（此时几乎总是 Adam 或 AdamW）。优化选择与架构选择正交。
- **正则化和规范化是架构组件，而不是事后的想法。** LayerNorm 和 RMSNorm (`01-foundations/regularization-techniques.md`) 实际上位于 Transformer 块 (`03-deep-learning-architectures/transformer-architecture.md`) 内部 - 它们没有用螺栓固定在顶部。
- **EM 算法** (`02-classical-ml/probabilistic-models.md`) 是变分推理中使用的推理的概念祖先，它再次出现在 VAE (`04-generative-models/vaes.md`) 中。
- **注意力是重复使用三次的单一机制。** `03-deep-learning-architectures/rnn-lstm-gru.md` 中的 Bahdanau/Luong 注意力是 `03-deep-learning-architectures/transformer-architecture.md` 中自注意力的直接概念前身，它本身就是 `06-inference-optimization` 中每个文件变得更便宜的机制。
- **RLHF是`07-reinforcement-learning`和`05-training-methodology`合并的地方。** `07-reinforcement-learning/policy-gradient-methods.md`（尤其是PPO）中的策略梯度理论是`05-training-methodology/rlhf-and-alignment.md`底层的RL引擎； `07-reinforcement-learning/rl-for-llms.md`被专门写为两者之间的桥梁。
- **生成模型系列是同一问题的竞争解决方案**（对数据分布进行足够好的建模，以便从中采样新的现实示例）：GAN、VAE、扩散模型、自回归模型和基于流的模型是五个不同的答案，`04-generative-models/diffusion-models.md` 和 `04-generative-models/gans.md` 都从自己的角度解释了为什么扩散取代了 GAN 来生成图像。
- **推理优化技术堆栈。** 量化、修剪/蒸馏、KV 缓存优化、推测性解码和服务/批处理并不是相互排斥的替代方案 - 生产 LLM 服务系统同时结合了多个替代方案。 `10-comparison-tables/compute-cost-comparison.md` 将它们视为可组合的堆栈而不是单一选择。
- **缩放法则连接架构、预训练和路线图。** Kaplan 等人。 Chinchilla 缩放定律（`03-deep-learning-architectures/transformer-architecture.md` 和 `05-training-methodology/pretraining-strategies.md`）也是 `09-roadmaps/near-term-outlook.md` 中外推的经验基础。

## 下一步去哪里

- 仓库新手？从 [README](../../README.md) 目录开始，然后是此文件，然后是 `01-foundations`。
- 正在寻找特定的算法？使用 [glossary](glossary.md) - 每个行话术语都链接到深入介绍的文件。
- 想要历史叙述而不是时事叙述？直接进入[08-history](../08-history/)。
