# ML/AI 算法知识库

全面、带索引、基于 Markdown 的参考，涵盖机器学习和 AI 中重要的算法和技术 — 经典 ML 和深度学习架构、生成模型、训练方法、推理优化和强化学习 — 以及每个主要系列背后的发展历史和该领域的发展方向。

**读者：** 为拥有计算机科学学士学位（扎实的编程、标准 CS 基础知识、本科线性代数/微积分/概率）但没有研究生水平 ML 背景的读者编写。每个文件都在首次使用时定义了 ML 特定的术语和符号。请参阅 [scope-and-methodology.md](docs/00-overview/scope-and-methodology.md) 了解完整的受众校准以及每个条目遵循的每个算法模板。

## 如何使用此仓库

- **这里是新功能？** 从 [taxonomy-map.md](docs/00-overview/taxonomy-map.md) 开始，以直观地了解所有内容如何关联，然后按下面的主题浏览。
- **遇到一个不熟悉的术语？** 检查生活中的 [glossary.md](docs/00-overview/glossary.md) - 此仓库中任何地方使用的每个术语都在那里定义，按字母顺序排列，并带有指向其深入介绍的链接。
- **想要故事，而不是主题图？** 按顺序阅读 [08-history](docs/08-history/) — 四个文件，从 1950 年代至今。
- **想要快速比较或做出决定？** 直接前往 [10-比较表](docs/10-comparison-tables/)。
- **好奇事情的发展方向？** [09-路线图](docs/09-roadmaps/) - 清楚地将来源声明与标记的 `[Projection]` 猜测分开。

## 目录

### 00 — 概述
|文件|描述 |
|---|---|
| [scope-and-methodology.md](docs/00-overview/scope-and-methodology.md) |该仓库涵盖的内容、受众校准以及整个过程中使用的由九部分组成的每种算法模板 |
| [glossary.md](docs/00-overview/glossary.md) |仓库中使用的每个术语的按字母顺序排列的生动词汇表 |
| [taxonomy-map.md](docs/00-overview/taxonomy-map.md) |每个算法系列如何相互关联的可视化思维导图 |

### 01 — 基础
|文件|描述 |
|---|---|
| [optimization-algorithms.md](docs/01-foundations/optimization-algorithms.md) |通过 AdamW、Lion、Sophia、LAMB/LARS、LR 计划、梯度裁剪的梯度下降 |
| [regularization-techniques.md](docs/01-foundations/regularization-techniques.md) | L1/L2/权重衰减、dropout、BatchNorm/LayerNorm/RMSNorm、数据增强、标签平滑 |
| [loss-functions.md](docs/01-foundations/loss-functions.md) | MSE、交叉熵、KL 散度、对比损失、焦点损失、困惑度 |

### 02 — 经典 ML
|文件|描述 |
|---|---|
| [supervised-learning.md](docs/02-classical-ml/supervised-learning.md) |线性/逻辑回归、决策树、SVM、k-NN、朴素贝叶斯 |
| [ensemble-methods.md](docs/02-classical-ml/ensemble-methods.md) |装袋、随机森林、Boosting、XGBoost/LightGBM/CatBoost、堆叠 |
| [unsupervised-learning.md](docs/02-classical-ml/unsupervised-learning.md) | k-means、层次聚类、DBSCAN、PCA、t-SNE、UMAP、自动编码器 |
| [probabilistic-models.md](docs/02-classical-ml/probabilistic-models.md) | HMM、高斯混合模型、EM 算法、贝叶斯网络 |

### 03 — 深度学习架构
|文件|描述 |
|---|---|
| [cnn-family.md](docs/03-deep-learning-architectures/cnn-family.md) |从头开始的卷积/池化、LeNet 通过 ResNet 和 EfficientNet、检测架构 |
| [rnn-lstm-gru.md](docs/03-deep-learning-architectures/rnn-lstm-gru.md) | Vanilla RNN、梯度消失问题、LSTM/GRU 门控、seq2seq、预 Transformer 注意力 |
| [transformer-architecture.md](docs/03-deep-learning-architectures/transformer-architecture.md) | **基石文件。** 来自第一原理的自注意力、位置编码/RoPE、编码器/解码器变体、缩放法则 |
| [mixture-of-experts.md](docs/03-deep-learning-architectures/mixture-of-experts.md) |稀疏路由、负载均衡、Switch Transformer、GShard、Mixtral |
| [state-space-models.md](docs/03-deep-learning-architectures/state-space-models.md) | S4、Mamba、RWKV 以及对采用与Transformer的诚实评估 |
| [graph-neural-networks.md](docs/03-deep-learning-architectures/graph-neural-networks.md) |消息传递、GCN、GraphSAGE、GAT、分子/推荐/知识图用例 |

### 04 — 生成模型
|文件|描述 |
|---|---|
| [gans.md](docs/04-generative-models/gans.md) |对抗性训练、模式崩溃、DCGAN/StyleGAN、为什么扩散取代了 GAN |
| [vaes.md](docs/04-generative-models/vaes.md) |重新参数化技巧，ELBO 用简单的英语解释，与扩散的关系 |
| [diffusion-models.md](docs/04-generative-models/diffusion-models.md) | DDPM、基于评分的建模、潜在扩散/稳定扩散、无分类器指导 |
| [autoregressive-generation.md](docs/04-generative-models/autoregressive-generation.md) |下一个词元预测、贪婪/束/top-k/核采样、退化和缓解 |
| [flow-based-models.md](docs/04-generative-models/flow-based-models.md) |流量正常化、RealNVP/Glow 以及对这个家族为何保持小众地位的诚实说明 |

### 05 — 培训方法
|文件|描述 |
|---|---|
| [pretraining-strategies.md](docs/05-training-methodology/pretraining-strategies.md) |掩蔽预训练与因果预训练、CLIP 式对比预训练、实践中的缩放法则 |
| [finetuning-and-peft.md](docs/05-training-methodology/finetuning-and-peft.md) |全面微调、LoRA/QLoRA、适配器、前缀/提示调整、成本比较 |
| [rlhf-and-alignment.md](docs/05-training-methodology/rlhf-and-alignment.md) |完整的 RLHF 管道（SFT → 奖励模型 → PPO）、DPO、RLAIF、宪法人工智能 |
| [distributed-training.md](docs/05-training-methodology/distributed-training.md) |数据/张量/管道并行性、ZeRO 阶段、FSDP、通信瓶颈 |
| [curriculum-and-data-strategies.md](docs/05-training-methodology/curriculum-and-data-strategies.md) |数据管理、重复数据删除、课程学习、合成数据、混合加权 |

### 06 — 推理优化
|文件|描述 |
|---|---|
| [quantization.md](docs/06-inference-optimization/quantization.md) | INT8/INT4 机制、PTQ 与 QAT、GPTQ、AWQ、准确性/延迟/内存权衡 |
| [pruning-and-distillation.md](docs/06-inference-optimization/pruning-and-distillation.md) |幅度修剪、结构化与非结构化、知识蒸馏 |
| [kv-cache-and-attention-optimization.md](docs/06-inference-optimization/kv-cache-and-attention-optimization.md) |从头开始的KV缓存、FlashAttention、PagedAttention、MQA/GQA |
| [speculative-decoding.md](docs/06-inference-optimization/speculative-decoding.md) |草稿和验证生成，为什么它是精确的（不是近似的），Medusa，前瞻解码 |
| [serving-and-batching.md](docs/06-inference-optimization/serving-and-batching.md) |静态与连续批处理、吞吐量/延迟权​​衡、vLLM/TensorRT-LLM |

### 07 — 强化学习
|文件|描述 |
|---|---|
| [value-based-methods.md](docs/07-reinforcement-learning/value-based-methods.md) | MDP、Q-learning、DQN、Double DQN、Dueling DQN、经验回放 |
| [policy-gradient-methods.md](docs/07-reinforcement-learning/policy-gradient-methods.md) | REINFORCE、演员评论家、A3C、TRPO、PPO（深入）、GAE |
| [model-based-rl.md](docs/07-reinforcement-learning/model-based-rl.md) |世界模型、MuZero、样本效率与无模型方法 |
| [rl-for-llms.md](docs/07-reinforcement-learning/rl-for-llms.md) |将一般强化学习理论与法学硕士培训联系起来；强化学习可验证的推理奖励 |

### 08 — 历史
|文件|描述 |
|---|---|
| [timeline-1950s-2000s.md](docs/08-history/timeline-1950s-2000s.md) |感知机、第一个AI寒冬、反向传播、LeNet、LSTM、2006年“深度学习”时刻 |
| [timeline-2000s-2017.md](docs/08-history/timeline-2000s-2017.md) | AlexNet/ImageNet、word2vec、DQN、GAN、seq2seq+attention、ResNet、AlphaGo、Transformer |
| [timeline-2017-2023.md](docs/08-history/timeline-2017-2023.md) | BERT/GPT-1、GPT-2/3、缩放法则、CLIP/RoPE、Chinchilla、InstructGPT、ChatGPT、LLaMA、Mixtral |
| [timeline-2023-present.md](docs/08-history/timeline-2023-present.md) |多模态融合、长上下文扩展、可验证奖励的强化学习、代理训练 |

### 09 — 路线图
|文件|描述 |
|---|---|
| [near-term-outlook.md](docs/09-roadmaps/near-term-outlook.md) | 6-18 个月预测：推理成本、上下文长度、代理训练、MoE 采用 |
| [long-term-outlook.md](docs/09-roadmaps/long-term-outlook.md) | 2-5 年预测：注意力替代方案、持续学习、测试时计算扩展 |
| [open-problems.md](docs/09-roadmaps/open-problems.md) |灾难性遗忘、幻觉、可解释性、代理可靠性、计算成本、评估有效性 |

### 10 — 比较表
|文件|描述 |
|---|---|
| [algorithm-comparison-master.md](docs/10-comparison-tables/algorithm-comparison-master.md) |每个类别一个主表：优化器、架构、生成模型、PEFT、推理优化 |
| [compute-cost-comparison.md](docs/10-comparison-tables/compute-cost-comparison.md) |训练和推理成本杠杆、它们如何叠加以及训练与服务成本之间的紧张关系 |
| [when-to-use-what.md](docs/10-comparison-tables/when-to-use-what.md) |美人鱼决策树的实用决策指导 |

## 仓库结构

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

## 贡献

请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解每个算法的模板、采购约定和术语表该仓库遵循的维护期望。
