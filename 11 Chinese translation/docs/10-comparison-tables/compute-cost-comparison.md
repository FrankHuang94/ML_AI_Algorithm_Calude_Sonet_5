# 计算成本比较

本文件专门关注本仓库所涵盖技术在训练与推理成本上的权衡——只做综合，不引入新算法。[algorithm-comparison-master.md](algorithm-comparison-master.md) 为每个类别给出了一个概括性的"计算特征"列，而本文件更深入地讨论具体的成本杠杆，以及它们如何叠加组合。

## 训练成本杠杆

| 杠杆 | 权衡的是什么 | 详见 |
|---|---|---|
| 模型规模（参数量） | 更大容量 vs. 每步更多计算/内存 | [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)（缩放定律） |
| 数据规模（token 数） | 更好的泛化 vs. 更多总计算量（参数量 × token 数） | [pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md) |
| 批量大小 | 更好的硬件利用率/并行度 vs. 优化器稳定性挑战 | [optimization-algorithms.md](../01-foundations/optimization-algorithms.md)（LAMB/LARS） |
| 并行策略（数据/张量/流水线/ZeRO） | 容纳更大的模型/批量 vs. 通信开销 | [distributed-training.md](../05-training-methodology/distributed-training.md) |
| 全量微调 vs. PEFT | 最高任务性能 vs. 大幅降低的计算/内存 | [finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md) |
| RLHF vs. DPO | 可复用的奖励模型 + 细粒度 RL 控制 vs. 简单/稳定 | [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md) |
| MoE vs. 稠密架构 | 更大的总容量 vs. 额外的路由/服务复杂度 | [mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md) |
| 训练精度（BF16 → FP8 → FP4） | 更少的内存/带宽和更快的矩阵乘法 vs. 更多的数值工程（缩放、离群值处理） | [quantization.md](../06-inference-optimization/quantization.md)、[pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md) |
| RL 后训练 vs. 在线策略蒸馏 | 通过探索发现新行为 vs. 来自现有教师的更稠密、更便宜的逐 token 信号 | [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)、[pruning-and-distillation.md](../06-inference-optimization/pruning-and-distillation.md) |

**Chinchilla 时代的框架（见 [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md) 和 [pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md)）：** 对于固定的训练计算预算，模型规模与数据规模之间的分配本身就是一个成本决策，而不是可以随意设定的变量——在相同总计算成本下，用更多数据训练一个更小的模型，可以追平甚至超过用相应更少数据训练的更大模型，而且得到的模型之后*服务*起来更便宜（见下文推理成本）——这就是为什么自 2022 年以来，现实中的预训练决策常常把数据量推到严格计算最优点之外：多花一点预训练计算，换取显著更低的长期推理成本。

## 推理成本杠杆

| 杠杆 | 权衡的是什么 | 详见 |
|---|---|---|
| 量化（位宽） | 更低内存/更快推理 vs. 精度损失（位宽越低损失越大） | [quantization.md](../06-inference-optimization/quantization.md) |
| 剪枝/蒸馏 | 更小/更快的模型 vs. 前期压缩成本 + 由原模型/教师模型设定的能力上限 | [pruning-and-distillation.md](../06-inference-optimization/pruning-and-distillation.md) |
| KV 缓存管理（PagedAttention、GQA/MQA） | 更高的批处理并发 vs. 一定的质量权衡（GQA/MQA）或工程复杂度（分页） | [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md) |
| 投机解码 | 更少的昂贵顺序模型调用 vs. 草稿模型的开销/复杂度 | [speculative-decoding.md](../06-inference-optimization/speculative-decoding.md) |
| 批处理策略（静态 vs. 连续） | 吞吐量 vs. 单个请求的延迟 | [serving-and-batching.md](../06-inference-optimization/serving-and-batching.md) |
| 上下文长度 | 更多可用输入 vs. 二次方的注意力成本和更大的 KV 缓存 | [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)、[kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md) |
| 注意力设计（GQA → MLA → 稀疏/压缩 → 混合线性） | 长上下文下更小的 KV 缓存和次二次方成本 vs. 需在架构阶段就做出承诺，并存在一定的回忆风险 | [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md)、[state-space-models.md](../03-deep-learning-architectures/state-space-models.md) |
| 推理长度（测试时计算） | 在难题上更高的准确率 vs. 每个请求多得多的解码 token | [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md) |
| 预填充/解码分离 | 独立扩缩容和更好的利用率 vs. KV 缓存传输成本和系统复杂度 | [serving-and-batching.md](../06-inference-optimization/serving-and-batching.md) |

**一个值得直白点出的不对称：** 上表中大多数推理成本杠杆（量化、剪枝、蒸馏）都是用一些精度换取速度/成本——这是需要有人刻意做出的真实权衡决策。有两个值得注意的例外，正因为它们不寻常，才在各自的文件中单独介绍：FlashAttention（输出完全相同，只是更快、更省内存——见 [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md)）和投机解码（输出分布完全相同，只是更快——见 [speculative-decoding.md](../06-inference-optimization/speculative-decoding.md)）在本仓库的内容中近乎"免费午餐"，这也正是它们相较于需要显式"质量换速度"决策的技术，能得到如此广泛、迅速且没有争议的采用的原因。

## 这些杠杆在实践中如何叠加

正如 [06-inference-optimization](../06-inference-optimization/) 各文件所描述的，一个现实的生产级 LLM 服务栈通常会同时组合其中若干项，而不是只选一项：一个按 Chinchilla 思路（或超越 Chinchilla、考虑服务成本）进行数据扩展训练的模型 → 量化到 4-8 位用于部署（到 2026 年末，越来越多地采用有原生硬件支持的 FP8 或块缩放 FP4 格式）→ 使用 GQA 或 MLA 提供服务，可能还带有稀疏注意力或混合线性注意力层（这些是在预训练时做出的架构选择）→ 采用 PagedAttention 式的 KV 缓存管理和前缀缓存 → 使用连续批处理，在大规模下还把预填充和解码拆分到不同的 GPU 池上 → 最上层再叠加投机解码（通常借助模型自身的 MTP 或 EAGLE 式草稿头）。每一层都针对总成本方程的不同部分（模型大小、内存占用、批处理并发、顺序调用次数），本仓库对 vLLM 和 TensorRT-LLM 的介绍（见 [serving-and-batching.md](../06-inference-optimization/serving-and-batching.md)）反映的正是这种组合式的技术栈，而不是某一种万能技术。

## 推理模型带来的转变：推理计算成为一条扩展轴

自推理模型出现以来（2024 年的 o1、2025 年的 DeepSeek-R1——见 [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)），总计算中越来越大的份额花在了推理时：一个困难请求在作答前可能生成数千到数万个"思考" token。这改变了哪些杠杆最重要。解码侧的优化——KV 缓存大小（MLA、稀疏注意力、混合架构）、投机解码、低精度服务和分离式服务——如今在成本方程中所占的比重，比典型回复还很短的时候更大；而"模型应该思考多久？"本身也已成为一个按请求调节的成本/质量旋钮。

## 训练成本 vs. 推理成本：真实存在的张力

**直白地说明这一权衡。** 在训练时花更多计算（按照上面 Chinchilla 时代的逻辑使用更多数据，或采用 QAT 而不是更便宜的 PTQ 等技术——见 [quantization.md](../06-inference-optimization/quantization.md)），可以得到一个在给定质量标准下推理更便宜的模型。考虑到一个已部署的模型可能在很长的生命周期内服务海量用户，这种"用训练换服务节省"的权衡值得付出可观的额外前期训练计算——这是贯穿 [pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md) 和 [quantization.md](../06-inference-optimization/quantization.md) 的一个反复出现的主题，本仓库把它视为塑造前沿模型研发决策的、较为重要（尽管不那么吸引眼球）的经济现实之一。

## 与其他文件的关系

- [algorithm-comparison-master.md](algorithm-comparison-master.md) 提供了更广泛的逐算法比较，本文件则专门在成本方面深入展开。
- [when-to-use-what.md](when-to-use-what.md) 把这些成本权衡转化为具体的决策指导。
