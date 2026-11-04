# 词汇表

此仓库中使用的每个术语的按字母顺序排列的动态列表。每个条目都有一个单行定义和一个指向该文件的链接，该文件在该文件中进行了深入介绍（如果任何单个文件“拥有”它）。该文件会附加到整个构建过程中 - 如果您发现文档文件中使用的术语未在此处列出，则这是一个错误；请添加。

## 如何使用此

此仓库中的文件根据 [scope-and-methodology.md](scope-and-methodology.md).该术语表的存在使您不必通过文件来重新查找定义，因此术语在整个仓库范围内保持一致（同一概念永远不应在两个不同的文件中具有两个不同的名称）。

## 按字母的术语计数（截至最后一次完整构建）

| A | B | C | D | E | F | G | H | I | K | L | M | N | O | P | Q | R | S | T | U | V | W | Z | 总计 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 3 | 9 | 13 | 8 | 5 | 12 | 6 | 5 | 3 | 7 | 15 | 1 | 4 | 7 | 3 | 9 | 12 | 7 | 1 | 2 | 3 | 1 | 147 |

---

## A

- **消融** - 一项实验，您删除或禁用系统的一个组件（一层、损失项、训练技巧）并重新测量性能，以找出该组件实际贡献了多少。如果删除它不会影响性能，那么它就没有发挥作用。涵盖于：[scope-and-methodology.md](scope-and-methodology.md)。
- **演员批评家** - 一种将策略（“演员”，选择动作）与学习价值函数（“批评家”，判断这些动作相对于期望有多好）相结合的强化学习方法，用于减少策略梯度方差。涵盖于：[policy-gradient-methods.md](../07-reinforcement-learning/policy-gradient-methods.md)。
- **优势（RL）** - 相对于该状态的基线期望，行动结果好坏多少，用于扩展策略梯度更新。涵盖于：[policy-gradient-methods.md](../07-reinforcement-learning/policy-gradient-methods.md)。
- **代理** — 描述了一种经过训练/用于采取多步骤操作（使用工具、调用 API、在减少逐步监督的情况下进行操作）以实现更长期目标的模型，而不是仅仅回答单个提示。涵盖于：[timeline-2023-present.md](../08-history/timeline-2023-present.md)。
- **智能体强化学习（Agentic RL）** — 在多轮交互回合上进行的强化学习：模型在沙箱中调用工具或运行代码，并因完成整体任务而获得奖励。涵盖于：[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)。
- **对齐** — 训练模型按照人类意图和价值观（有帮助、诚实、无害）行事，而不是仅仅预测统计上可能的文本。涵盖于：[rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)。
- **全归约** - 一种集体通信操作，分布式训练作业中的每个设备最终都会得到所有设备上计算的某个数量（通常是梯度）的组合（例如求和/平均）值。涵盖于：[distributed-training.md](../05-training-methodology/distributed-training.md)。
- **全对全通信（All-to-all）** — 一种集合通信操作：每个设备向其他每个设备发送不同的数据块；用于把 token 分发到位于不同设备上的专家，再把结果收回。涵盖于：[distributed-training.md](../05-training-methodology/distributed-training.md)。
- **注意力** - 一种计算查询和一组候选项目之间的相关性分数、将它们转换为权重（通常通过 softmax）并获取候选项目的加权和的机制；起源于 seq2seq 翻译（Bahdanau/Luong 注意力）并推广为 Transformer 自注意力。涵盖于：[rnn-lstm-gru.md](../03-deep-learning-architectures/rnn-lstm-gru.md)；另请参见[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **注意力汇聚点（Attention sink）** — 注意力头"停放"多余注意力权重的 token（通常是第一个）——因为 softmax 权重之和必须为 1，即使没有任何相关内容。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **无辅助损失负载均衡** — 通过给路由分数加上逐专家的偏置（过载专家调低、闲置专家调高）来均衡 MoE 专家的使用，而不是额外添加一个损失项；由 DeepSeek-V3 推广。涵盖于：[mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md)。

## B

- **基准** - 标准化数据集/任务对（加上评分规则），用于在相同条件下相互比较模型。 “基准 X 上的 SOTA”意味着“特定标准化测试的最佳发布分数”，比“总体最佳模型”范围更窄。涵盖于：[scope-and-methodology.md](scope-and-methodology.md)。
- **偏差-方差权衡** — 模型假设错误导致的系统误差（偏差）与对所使用的特定训练样本的敏感性（方差）之间的紧张关系；像 bagging 这样的集成主要是减少方差。涵盖于：[ensemble-methods.md](../02-classical-ml/ensemble-methods.md)。
- **引导（RL 意义）** — 使用模型自身当前（不完美）的未来价值估计来更新价值估计，而不是等待完全观察到的结果。涵盖于：[value-based-methods.md](../07-reinforcement-learning/value-based-methods.md)。

## C

- **校准** — 模型的置信度（预测概率）与其实际经验准确性的匹配程度。一个经过良好校准的模型显示“80% 的置信度”在大约 80% 的情况下是正确的。涵盖于：[regularization-techniques.md](../01-foundations/regularization-techniques.md)。
- **灾难性遗忘** - 在新数据上训练模型时会降低或消除以前学习的能力，这是实际持续/在线学习的核心障碍。涵盖于：[open-problems.md](../09-roadmaps/open-problems.md)。
- **因果掩蔽** - 在自注意力中，阻止每个位置关注后面的位置（通过在 softmax 之前将它们的分数设置为负无穷大），因此模型无法“看到未来”它应该预测的。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **思维链（Chain of thought, CoT）** — 模型在给出最终答案前以 token 形式写出的中间推理步骤；推理模型被训练成生成很长的思维链。涵盖于：[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)。
- **压缩稀疏注意力（CSA）/ 重度压缩注意力（HCA）** — DeepSeek-V4 的注意力变体（2026 年，据报道）：CSA 把小组 token 压缩成单个 KV 条目，并关注被选中的 top-k 条目和一个近期窗口；HCA 则在压缩程度高得多的块上做稠密注意力。涵盖于：[kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md)。
- **计算最优** — 在模型大小和数据大小之间分配固定的训练计算预算，最大限度地减少损失；龙猫论文的核心发现。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **信用分配问题** - RL 难以确定哪些较早的行为应该因较晚到达的奖励而得到赞扬（或责备）；几乎每个强化学习算法都有不同的答案。涵盖于：[value-based-methods.md](../07-reinforcement-learning/value-based-methods.md)。
- **交叉注意力** - 一种注意力变体，其中查询来自一个序列（例如解码器），键/值来自另一个序列（例如编码器的输出），让一个序列回顾另一个序列。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **交叉熵** — 衡量预测概率分布与真实目标分布之间差距的损失函数；等于真实分布的熵加上真实分布和预测分布之间的 KL 散度。分类和语言建模的标准损失。涵盖于：[loss-functions.md](../01-foundations/loss-functions.md)。

## D

- **DAPO** — 2025 年对 GRPO 的改进（字节跳动 Seed）：引入更高的截断上界（"clip-higher"）、丢弃全对/全错组的动态采样，以及 token 级损失。涵盖于：[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)。
- **数据泄漏** - 当有关目标变量的信息在预处理过程中无意中泄漏到特征中时，会以一种无法泛化的方式夸大训练/验证性能。涵盖于：[ensemble-methods.md](../02-classical-ml/ensemble-methods.md)。
- **DDIM（去噪扩散隐式模型）** — 一种以更少、更大、确定性步骤从经过训练的扩散模型中进行采样的技术，无需重新训练即可加速生成。涵盖于：[diffusion-models.md](../04-generative-models/diffusion-models.md)。
- **决策边界** — 特征空间中的划分面，分类器从预测一个类切换到另一个类；它的形状（直线、阶梯、平滑曲线）表征了分类器。涵盖于：[supervised-learning.md](../02-classical-ml/supervised-learning.md)。
- **退化** - 似然最大化文本解码（贪婪/束搜索）的失败模式，产生重复、通用或循环输出，而不是变化的、类似人类的文本。涵盖于：[autoregressive-generation.md](../04-generative-models/autoregressive-generation.md)。
- **Delta 规则（线性注意力中）** — 一种记忆写入规则：先读出记忆对某个键当前返回的值，再只写入修正量；等价于对记忆的预测误差做一步梯度下降。涵盖于：[state-space-models.md](../03-deep-learning-architectures/state-space-models.md)。
- **扩散语言模型（dLLM）** — 一种文本生成模型：从完全被掩码的序列出发，在几步细化中并行地揭开大量 token，而不是一次生成一个 token。涵盖于：[diffusion-models.md](../04-generative-models/diffusion-models.md)。
- **分离式服务（预填充/解码分离）** — 把 LLM 推理中计算密集的预填充阶段和内存带宽密集的解码阶段放在不同的 GPU 池上运行，并在两者之间传输 KV 缓存。涵盖于：[serving-and-batching.md](../06-inference-optimization/serving-and-batching.md)。
- **判别模型与生成模型** — 判别模型直接学习类之间的边界 (P(class\|features))；生成模型对每个类的数据进行建模 (P(features\|class)) 并应用贝叶斯规则进行分类。涵盖于：[supervised-learning.md](../02-classical-ml/supervised-learning.md)。
- **解缠结** - 潜在表示的一种属性，其中单独的维度捕获单独的、独立的变化因素（例如，旋转与大小）；受到β-VAE的鼓励。涵盖于：[vaes.md](../04-generative-models/vaes.md)。
- **蒸馏（知识蒸馏）** - 训练一个较小的“学生”模型来重现较大的“教师”模型的输出分布，产生一个更小、更便宜的模型，该模型通常优于仅在硬标签上从头开始训练的模型。涵盖于：[pruning-and-distillation.md](../06-inference-optimization/pruning-and-distillation.md)。
- **Dr. GRPO** — 2025 年的一种 GRPO 变体（"正确实现的 GRPO"），去掉了长度归一化和标准差归一化——它们会让训练偏向冗长的错误答案，以及过易或过难的题目。涵盖于：[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)。
- **两用能力 / 分级发布** — 一种既可用于防御也可用于攻击的能力（例如发现软件漏洞）；分级（分阶段）发布是指先把此类模型提供给经过审核的用户，再广泛开放。涵盖于：[open-problems.md](../09-roadmaps/open-problems.md)。

## E

- **EAGLE（投机解码）** — 一类草稿头：利用目标模型自身的内部特征预测后续 token，而不是使用单独的小模型；EAGLE-3（2025 年）使用来自多个层的特征。涵盖于：[speculative-decoding.md](../06-inference-optimization/speculative-decoding.md)。
- **ELBO（证据下界）** - 生成模型（否则难以处理）数据似然的可计算下界，由重建项和 KL 散度正则化项组成； VAE 的培训目标。涵盖于：[vaes.md](../04-generative-models/vaes.md)。
- **嵌入** — 表示学习空间中的输入（单词、图像、用户等）的密集数值向量，其中几何距离对应于语义相似性。涵盖范围：[loss-functions.md](../01-foundations/loss-functions.md)（对比损失）；另请参见[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **熵** — 概率分布中固有不确定性（或信息内容）的度量；零表示完全可预测的结果，最大表示所有结果的可能性相同。涵盖于：[loss-functions.md](../01-foundations/loss-functions.md)。
- **专家并行** — 把不同的 MoE 专家放在不同的设备上，并把每个 token 路由到持有其所选专家的设备。涵盖于：[distributed-training.md](../05-training-methodology/distributed-training.md)。
- **暴露偏差（Exposure bias）** — 当模型只模仿参考文本、从未学会从自身错误中恢复时产生的训练/测试不一致；在线策略训练可以缓解它。涵盖于：[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)。
- **𝔼（期望）** — 数量的平均值，按每个结果的可能性进行加权；机器学习论文中“对数据进行平均”或“对分布样本进行平均”的标准表示法。涵盖于：[gans.md](../04-generative-models/gans.md)。
- **体验重放** - 将 RL 代理过去的经验存储在缓冲区中，并对其随机采样的批次进行训练，而不是在其经历的时间相关流上进行训练，以稳定训练。涵盖于：[value-based-methods.md](../07-reinforcement-learning/value-based-methods.md)。

## F

- **前馈网络 (FFN)** — 每个 Transformer 块中的位置两层子网络（同样应用于每个词元）；保存了块的大部分参数，并且是专家混合 (Mixture-of-Experts) 所取代的许多专家。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **少样本（学习/提示）** - 在推理时（在提示中，而不是通过权重更新）为模型提供少量（通常为 1-100）示例输入/输出对，并期望它将模式泛化为新输入。与零射击对比。涵盖于：[scope-and-methodology.md](scope-and-methodology.md)。
- **FLOPs（浮点运算）** — 计算成本的衡量标准；用于量化模型处理词元、训练或运行推理所需的计算量。涵盖于：[mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md)。
- **流匹配/整流流** - 生成模型训练目标，学习沿（理想情况下是直的）路径将噪声传输到数据的速度场；与扩散密切相关，并且越来越多地用于大型图像/视频生成器。涵盖于：[diffusion-models.md](../04-generative-models/diffusion-models.md)。
- **FP8 / 混合精度训练** — 用低精度格式（8 位浮点）完成大部分矩阵乘法，同时把主权重和敏感运算保留在更高精度；DeepSeek-V3 在前沿规模上验证了这一做法。涵盖于：[pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md)。

## G

- **门控注意力（Gated attention）** — 对每个注意力头的输出施加逐元素的 sigmoid 门，让注意力头可以"关闭"；能减少注意力汇聚点并提升训练稳定性（NeurIPS 2025 最佳论文）。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **Gated DeltaNet** — 一种线性注意力层，把 delta 规则与对其固定大小状态的、依赖输入的可学习衰减结合起来；是 Qwen3-Next、Qwen3.5 等 2025-2026 年混合模型的主力层。涵盖于：[state-space-models.md](../03-deep-learning-architectures/state-space-models.md)。
- **GELU / SwiGLU** — 现代 Transformer 的前馈子层中使用的平滑 (GELU) 和门控 (SwiGLU) 激活函数，代替旧的 ReLU。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **泛化** - 模型在未训练的数据上的表现如何，而不是仅仅记住其训练集。正则化的全部目的就是改善这一点。涵盖于：[regularization-techniques.md](../01-foundations/regularization-techniques.md)。
- **梯度** — 损失函数相对于每个模型参数的偏导数向量；点位于损失增加最陡的方向，因此训练步骤与其相反。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **梯度裁剪** - 重新调整梯度向量，使其范数永远不会超过固定阈值，以防止单个异常大的梯度破坏训练的稳定性。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **梯度爆炸** - 一种故障模式，其中梯度在通过多个层或时间步反向传播时呈倍增增长，产生巨大的、不稳定的更新。与消失的梯度形成对比。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **梯度消失** - 与梯度爆炸相反的故障模式：梯度在通过许多层/时间步向后传播时收缩到零，因此早期层停止学习。涵盖于：[rnn-lstm-gru.md](../03-deep-learning-architectures/rnn-lstm-gru.md)。
- **GroupNorm（组标准化）** — 一种标准化方案，可对每个示例的一组特征通道进行标准化；像 LayerNorm 一样与批次无关，并且在扩散模型 U-Net 中很常见。涵盖于：[regularization-techniques.md](../01-foundations/regularization-techniques.md)。
- **GRPO（组相对策略优化）** — 一种面向 LLM 的 RL 算法，用一组采样回答的平均得分作为基线，从而省去 PPO 单独的价值模型；在推理模型训练中被广泛使用，并在 2025 年经 DAPO、Dr. GRPO 和 GSPO 加以改进。涵盖于：[rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)；另见 [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md) 和 [policy-gradient-methods.md](../07-reinforcement-learning/policy-gradient-methods.md)。
- **GSPO（组序列策略优化）** — 2025 年的一种 GRPO 变体（Qwen）：按整条序列而不是逐 token 计算重要性比率并进行截断；更稳定，尤其适用于 MoE 模型上的 RL。涵盖于：[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)。
- **指导尺度** - 在扩散模型中，可调强度参数，控制通过无分类器指导将生成推向条件信号（例如文本提示）的强度。涵盖于：[diffusion-models.md](../04-generative-models/diffusion-models.md)。

## H

- **幻觉** - 当模型生成流畅、听起来自信的输出时，实际上是不正确的；这是训练目标优化合理性而不是验证事实的结果。涵盖于：[open-problems.md](../09-roadmaps/open-problems.md)。
- **Hessian** — 函数的二阶导数矩阵；描述损失景观的局部曲率。太大（参数²）而无法直接计算深度网络，这就是为什么大多数训练仅使用一阶导数（梯度）信息的原因。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **隐藏状态** - 由循环网络在每个时间步维护和更新的向量，总结了迄今为止网络在序列中看到的所有相关内容。涵盖于：[rnn-lstm-gru.md](../03-deep-learning-architectures/rnn-lstm-gru.md)。
- **Huber 损失** — 一种回归损失，其对于小错误的行为类似于 MSE，对于大错误的行为类似于 MAE，将平滑梯度与对异常值的鲁棒性结合起来；用于稳健回归和 DQN 值目标。涵盖于：[loss-functions.md](../01-foundations/loss-functions.md)。
- **混合架构（线性 + 全注意力）** — 把大部分固定状态层（SSM 或线性注意力）与少量全注意力层交错排列的模型（例如 3:1 的比例），在保留精确回忆能力的同时降低长上下文成本。涵盖于：[state-space-models.md](../03-deep-learning-architectures/state-space-models.md)。
- **超连接 / mHC** — 把残差连接推广为由可学习矩阵混合的多条并行残差流；DeepSeek 的 mHC 把这些矩阵约束为双随机矩阵，使大规模训练保持稳定。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。

## I

- **隐式正则化** — 正则化效果是标准训练选择（SGD 梯度噪声、提前停止、有限精度）的副作用，而不是故意添加的；这是非常大的模型过度拟合少于其参数数量所建议的部分原因。涵盖于：[regularization-techniques.md](../01-foundations/regularization-techniques.md)。
- **上下文学习（In-context learning）** — 仅凭输入中提供的示例、在一次前向传播内完成新任务，而不更新任何权重；LLM 和 TabPFN 等表格基础模型都使用这种方式。涵盖于：[ensemble-methods.md](../02-classical-ml/ensemble-methods.md)。
- **归纳偏差** - 模型架构中内置的假设（而不是从数据中学习），使某些模式更容易学习；例如，卷积假设附近的像素是相关的。涵盖于：[cnn-family.md](../03-deep-learning-architectures/cnn-family.md)。
- **归纳与转导学习** - 归纳模型泛化到新的、看不见的节点/图表/示例；转导模型仅在训练期间看到的固定数据上定义，并且不会自然地扩展到它之外。涵盖于：[graph-neural-networks.md](../03-deep-learning-architectures/graph-neural-networks.md)。
- **可解释性** — 模型的内部计算和行为可以用人类可理解的术语解释的程度；大型神经网络的一个主要开放问题。涵盖于：[open-problems.md](../09-roadmaps/open-problems.md)。

## K

- **内核技巧** - 一种数学捷径，让算法（经典的 SVM）像数据已映射到更高维度的空间一样运行，而无需通过核函数直接计算点积来显式计算该映射。涵盖于：[supervised-learning.md](../02-classical-ml/supervised-learning.md)。
- **KL 散度（Kullback-Leibler 散度）** — 衡量一个概率分布与另一个概率分布（不对称）的差异程度；用于惩罚偏离参考分布的分布的标准工具。涵盖于：[loss-functions.md](../01-foundations/loss-functions.md)。
- **KV 缓存** — 存储序列中先前位置的键和值向量，在自回归生成期间重用，以避免在每一步中冗余地重新计算它们。涵盖于：[kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md)。

## L

- **潜在空间/潜在表示** - 一种学习的、压缩的数据向量表示（例如，自动编码器的瓶颈层），捕获其重要的潜在变化因素，而不是其原始输入形式。涵盖于：[unsupervised-learning.md](../02-classical-ml/unsupervised-learning.md)；另请参见[vaes.md](../04-generative-models/vaes.md)。
- **学习率 (η)** — 标量步长，控制单个优化步骤沿（负）梯度方向移动参数的距离。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **闪电索引器（Lightning indexer）** — DeepSeek 稀疏注意力中的轻量级打分模块，决定每个查询要关注哪些过去的 token（或压缩条目）。涵盖于：[kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md)。
- **线性注意力** — 去掉 softmax 的注意力，可改写为在固定大小矩阵状态上的循环，因此每个 token 的计算和内存不随上下文长度增长。涵盖于：[state-space-models.md](../03-deep-learning-architectures/state-space-models.md)。
- **Logits** — 分类器在转换为概率之前生成的原始非标准化分数（通常通过 softmax）。涵盖于：[loss-functions.md](../01-foundations/loss-functions.md)。
- **损失景观** — 通过针对模型参数的每种可能设置绘制损失函数值而形成的高维表面；训练就是在这个表面上寻找低点的过程。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **低秩分解** - 将大矩阵更新近似为两个小得多（薄）矩阵的乘积，大大减少可训练参数； LoRA 背后的机制。涵盖于：[finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md)。

## M

- **马尔可夫决策过程 (MDP)** — 马尔可夫属性下的顺序决策的标准形式化：状态、行动、奖励和策略。涵盖于：[value-based-methods.md](../07-reinforcement-learning/value-based-methods.md)。
- **马尔可夫性质** - 假设未来仅通过当前状态而不是完整的历史依赖于过去；马尔可夫链、HMM 和 MDP 的定义假设。涵盖于：[probabilistic-models.md](../02-classical-ml/probabilistic-models.md)；另请参见[value-based-methods.md](../07-reinforcement-learning/value-based-methods.md)。
- **微缩放格式（MXFP4 / NVFP4）** — 让一小块数值共享一个缩放因子的低位宽浮点格式：MXFP4 使用 32 个值一块、缩放因子为 2 的幂；NVFP4 使用 16 个值一块、FP8 缩放因子外加一个逐张量缩放因子。涵盖于：[quantization.md](../06-inference-optimization/quantization.md)。
- **中期训练（Mid-training）** — 介于大规模预训练与后训练之间、非正式命名的阶段，使用经过精选的高质量、偏推理的数据，并常常扩展上下文窗口。涵盖于：[pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md)。
- **最小最大游戏/目标** - 一种优化设置，其中两方具有直接相反的目标（一个最大化，一个最小化相同的表达式）； GAN 背后的训练框架。涵盖于：[gans.md](../04-generative-models/gans.md)。
- **专家混合 (MoE) / 路由器** — 一种架构模式，其中存在许多并行子网络（“专家”），但小型路由器网络仅选择少数几个来处理每个词元，从而将总参数计数与每个词元的计算成本解耦。涵盖于：[mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md)。
- **模式崩溃** - 一种 GAN 故障模式，其中生成器仅产生一小部分非多样化的输出，这些输出恰好欺骗了当前鉴别器，而不是覆盖真实数据分布的全部多样性。涵盖于：[gans.md](../04-generative-models/gans.md)。
- **模式覆盖与模式寻求** — KL 散度的两个方向产生的两种行为：模式覆盖（前向 KL）扩展概率以覆盖所有目标分布；模式搜索（反向 KL）锁定一种模式并忽略其他模式。涵盖于：[loss-functions.md](../01-foundations/loss-functions.md)。
- **模型崩溃** - 当模型在其他模型生成的数据上迭代训练而没有充分的真实、多样化的数据基础时，可能会发生质量/多样性的下降。涵盖于：[curriculum-and-data-strategies.md](../05-training-methodology/curriculum-and-data-strategies.md)。
- **无模型（相对于基于模型）RL** — 无模型方法（Q 学习、DQN、策略梯度）直接从经验中学习，无需环境动态的显式模型；基于模型的方法学习或使用这样的模型。涵盖于：[value-based-methods.md](../07-reinforcement-learning/value-based-methods.md)；另请参见[model-based-rl.md](../07-reinforcement-learning/model-based-rl.md)。
- **动量** - 一种优化技术，可累积过去梯度的运行平均值和该平均方向上的步数，而不是原始当前梯度，从而平滑轨迹。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **多头潜在注意力（MLA）** — 一种注意力设计（DeepSeek-V2，2024 年）：每个 token 只缓存一个很小的压缩潜在向量，再从中重建每个头的键和值，使 KV 缓存远小于 GQA。涵盖于：[kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md)。
- **多 token 预测（MTP）** — 一种额外的训练目标，用额外的预测头预测往后若干位置的 token；该预测头之后还可以充当投机解码的草稿。涵盖于：[pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md)。
- **Muon** — 2024 年提出的优化器，对网络的二维权重矩阵施加正交化、感知矩阵结构的更新；在 2025-2026 年已在万亿参数规模上得到验证（见 MuonClip），如今是大规模预训练中可替代 AdamW、且经过生产验证的可信选择，优化器状态内存也更低。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **MuonClip / QK-clip** — Muon 加上一种保护机制：当某个注意力头的最大注意力 logit 超过阈值时，重新缩放其查询/键权重，防止 logit 爆炸；被用于训练约 1T 参数的 Kimi K2。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。

## N

- **嵌套学习（Nested Learning）** — Google 2025 年提出的框架：把模型视为一组嵌套的优化问题，各部分以不同频率更新（"连续谱记忆"）；是一种处于早期阶段的持续学习方法。涵盖于：[long-term-outlook.md](../09-roadmaps/long-term-outlook.md)。

## O

- **在线策略蒸馏（On-policy distillation）** — 一种蒸馏方式：学生模型自己生成输出，教师对每个 token 打分（逐 token 反向 KL）；其多教师版本把多个专家 RL 模型合并进一个学生模型，是 2026 年被广泛报道的后训练模式。涵盖于：[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)。
- **优化器状态** - 优化器维护的额外每个参数缓冲区（例如 Adam 的两矩估计）；在法学硕士规模上，这可能超过模型自身权重的大小，并且是内存分片技术的主要目标。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **过度拟合** - 当模型与训练数据（包括其噪声/怪癖）拟合得如此紧密时，其在新的、看不见的数据上的表现比拟合程度较低的模型更差。与欠拟合对比。涵盖于：[regularization-techniques.md](../01-foundations/regularization-techniques.md)。
- **过度参数化** — 描述了一个网络，其参数多于其学习功能严格所需的参数；剪枝的经验基础（可以删除许多权重，而精度损失很小）。涵盖于：[pruning-and-distillation.md](../06-inference-optimization/pruning-and-distillation.md)。

## P

- **PEFT（参数高效微调）** — 微调方法（LoRA、适配器、前缀/提示调整）的总称，冻结大部分预训练模型的权重并仅训练少量附加/修改参数。涵盖于：[finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md)。
- **Perplexity** — 语言模型交叉熵损失 (e^loss) 的人类可解释的变换；粗略地说，“该模型的不确定性就像在每个词元的众多选项中进行统一选择一样。”越低越好。涵盖于：[loss-functions.md](../01-foundations/loss-functions.md)。
- **策略** - 在强化学习中，模型/函数被训练以选择动作（例如，生成哪个词元）； RLHF 的 RL 阶段将优化语言模型作为策略。涵盖于：[rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)；另请参见[value-based-methods.md](../07-reinforcement-learning/value-based-methods.md)。
- **后训练（Post-training）** — 预训练之后对模型所做的一切：监督微调、偏好微调、强化学习和蒸馏。涵盖于：[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)。
- **后崩溃** - 一种 VAE 故障模式，其中潜在代码变得无信息（编码器仅输出先验代码），因为强大的解码器学会忽略它，同时仍将 KL 项驱动为零。涵盖于：[vaes.md](../04-generative-models/vaes.md)。
- **前缀缓存（Prefix caching）** — 在多个请求之间复用共享提示词前缀（系统提示词、文档、对话历史）已经计算好的 KV 缓存，而不是重新计算。涵盖于：[serving-and-batching.md](../06-inference-optimization/serving-and-batching.md)。
- **先验** - 在观察数据之前假设变量的概率分布（例如，VAE 假设的潜在向量的标准正态分布），用作推理正则化的参考/目标。涵盖于：[vaes.md](../04-generative-models/vaes.md)。

## Q

- **QK-norm** - 在计算分数之前对注意力内的查询和关键向量进行归一化，在大型转换器中使用，以防止注意力逻辑增长到极端程度并破坏训练的稳定性。涵盖于：[regularization-techniques.md](../01-foundations/regularization-techniques.md)。
- **量化** - 降低用于表示模型权重/激活的数值精度（例如，从 16 位到 4 位），缩小内存占用量，并通常以一定的准确性成本加快推理速度。涵盖于：[quantization.md](../06-inference-optimization/quantization.md)。
- **查询、键、值 (Q/K/V)** — 每个位置在自注意力输入中的三个学习投影：查询表示位置正在寻找什么，键表示位置广告的内容，值表示位置在关注时提供的信息。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。

## R

- **推理模型（Reasoning model）** — 主要通过基于可验证奖励的强化学习训练、在回答前先写出长思维链的 LLM；其准确率随测试时计算的增加而提升（例如 o1、DeepSeek-R1）。涵盖于：[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)。
- **感受野** — 给定神经元输出有效影响的原始输入区域；在 CNN 的更深层中变得更大。涵盖于：[cnn-family.md](../03-deep-learning-architectures/cnn-family.md)。
- **ReLU（修正线性单元）** — 激活函数 max(0, x)；负输入输出零，正输入保持不变。 CNN 中的标准，并在其他地方广泛使用，因为它的梯度不会因正输入而缩小。涵盖于：[cnn-family.md](../03-deep-learning-architectures/cnn-family.md)。
- **重新参数化技巧** - 将随机样本重写为学习参数加上固定外部随机性的确定性函数，因此梯度可以通过其他不可微分的采样步骤反向传播。涵盖于：[vaes.md](../04-generative-models/vaes.md)。
- **残差流（Residual stream）** — Transformer 中持续传递的隐藏状态向量，每一层都从中读取，并通过残差连接把自己的输出加回其中。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **奖励黑客**——当一项政策利用学习奖励模型中的缺陷来获得高分时，却实际上没有产生人类认为良好的输出。涵盖于：[rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)。
- **奖励模型** - 一种经过训练来预测人类偏好判断的模型，在 RL 训练期间用作人类反馈的快速自动化代理。涵盖于：[rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)。
- **RL 环境（RL environment）** — 打包好的任务、沙箱与自动评分器，用于以强化学习训练智能体模型；是 2025-2026 年的一个主要投入方向。涵盖于：[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)。
- **RLVR（基于可验证奖励的强化学习）** — 奖励来自自动检验正确性（核对数学答案、通过单元测试）而非学习到的奖励模型的强化学习。涵盖于：[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)。

## S

- **样本效率** - 代理或模型需要多少真实世界的交互/数据才能达到给定的性能水平；相对于无模型方法，基于模型的强化学习旨在改进这一点。涵盖于：[model-based-rl.md](../07-reinforcement-learning/model-based-rl.md)。
- **缩放法则** - 语言模型的损失与其参数计数、数据集大小和训练计算之间的经验幂律关系，用于规划大型训练运行。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **评分函数** — 数据分布的对数概率相对于数据本身的梯度；在多个噪声级别上学习这一点是基于分数的生成模型的基础，与 DDPM 的噪声预测目标密切相关。涵盖于：[diffusion-models.md](../04-generative-models/diffusion-models.md)。
- **自我监督学习** - 使用从输入数据本身自动派生的标签进行训练（例如，预测屏蔽或下一个标记），而不需要人工注释。涵盖于：[pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md)。
- **分片** — 将大型对象（模型参数、梯度、优化器状态）分割成分布在多个设备上的片段，因此没有一个设备需要保存整个对象。涵盖于：[distributed-training.md](../05-training-methodology/distributed-training.md)。
- **滑动窗口（局部）注意力** — 只关注最近 W 个 token 的注意力；常与少量全（全局）注意力层交错使用，以降低长上下文成本。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **Softmax** — 通过取幂和归一化将原始分数（logits）向量转换为有效概率分布（均为正数，总和为 1）的函数。涵盖于：[loss-functions.md](../01-foundations/loss-functions.md)。
- **SOTA（最先进的技术）** — 在给定时间点上给定基准的最佳公开报告结果。一个移动的目标，而不是一个固定的方法——“SOTA”描述的是排行榜位置，而不是任何一种算法。涵盖于：[scope-and-methodology.md](scope-and-methodology.md)。
- **稀疏注意力（DeepSeek 稀疏注意力）** — 每个查询只关注过去 token 中被选出的一个子集（由闪电索引器等低成本打分步骤选出），而不是全部 token 的注意力。涵盖于：[kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md)。
- **稀疏奖励** - 一种 RL 设置，其中反馈仅在长序列操作结束时可用（例如，仅当完整的 LLM 响应完成时）而不是连续的，这使得学分分配更加困难。涵盖于：[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)。
- **稀疏性** — 网络权重为零（或被删除）的比例；修剪增加稀疏性以减小模型大小。涵盖于：[pruning-and-distillation.md](../06-inference-optimization/pruning-and-distillation.md)。
- **随机舍入（Stochastic rounding）** — 按与相邻两个可表示值的接近程度成比例的概率随机向上或向下舍入，使舍入误差相互抵消，而不是朝一个方向累积。涵盖于：[pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md)。

## T

- **表格基础模型（TabPFN）** — 在数百万个合成表格上预训练的 Transformer，通过上下文学习在一次前向传播中对新的小型表格做预测，无需在该表格上训练（TabPFN v2，《自然》2025）。涵盖于：[ensemble-methods.md](../02-classical-ml/ensemble-methods.md)。
- **温度** — 采样期间在 softmax 之前除以 logits 的标量；低于 1 会锐化（更具确定性）模型的输出分布，高于 1 会使其变得平坦（更加随机/多样化）。涵盖于：[autoregressive-generation.md](../04-generative-models/autoregressive-generation.md)。
- **时间差异（TD）误差** - 在强化学习中，价值估计和更好的“目标”估计之间的差距（奖励加上折扣的下一状态值）；驱动 Q 学习和价值模型更新的信号。涵盖于：[value-based-methods.md](../07-reinforcement-learning/value-based-methods.md)。
- **测试时计算** — 在推理时花费的计算（例如生成更长的中间推理）以提升输出质量，是一条不同于预训练计算的扩展轴；推理模型（2024 年的 o1、2025 年的 DeepSeek-R1）使其成为实用的扩展轴。涵盖于：[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)；另见 [long-term-outlook.md](../09-roadmaps/long-term-outlook.md)。
- **测试时训练（Test-time training）** — 在推理过程中针对当前输入更新一部分模型参数（"快速权重"）；是通往持续学习的一个早期研究方向。涵盖于：[long-term-outlook.md](../09-roadmaps/long-term-outlook.md)。
- **词元** — 语言模型将其视为单个输入/输出单元的文本块（通常是单词或子单词片段）。涵盖于：[autoregressive-generation.md](../04-generative-models/autoregressive-generation.md)。
- **Tokenizer / BPE（字节对编码）** — 通过将频繁的相邻对贪婪地合并为单个标记，将原始文本转换为模型消耗的固定子词词汇的组件（和主导算法）。涵盖于：[pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md)。

## U

- **欠拟合** - 当模型太简单或训练不足而无法捕获训练数据中的真实模式时，在训练和新数据上都表现不佳。与过拟合对比。涵盖于：[regularization-techniques.md](../01-foundations/regularization-techniques.md)。

## V

- **价值模型/评论家** - 在强化学习中，一种学习估计器，用于评估状态（或状态动作）在预期未来奖励方面的好坏；由 Actor-Critic 方法和 PPO 使用，并由 GRPO 删除。涵盖于：[policy-gradient-methods.md](../07-reinforcement-learning/policy-gradient-methods.md)。
- **VQ-VAE（矢量量化 VAE）** — 具有离散潜在的 VAE 变体（每个潜在位置都捕捉到学习密码本中最近的条目），使图像/音频能够通过自回归 Transformer 将图像/音频建模为词元序列。涵盖于：[vaes.md](../04-generative-models/vaes.md)。

## W

- **热身** - 以较小的学习率开始训练，并在第一个步骤中逐步提高，以避免梯度/方差估计仍然有噪声时不稳定。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **世界模型** — 对环境如何演化（给定当前状态和动作时的下一个状态与奖励）的学习得到或给定的表示，用于规划或生成模拟经验；近期的例子既有 Dreamer（在学到的世界模型中训练智能体），也有 Genie 3/4（实时、可漫游的生成世界）。涵盖于：[model-based-rl.md](../07-reinforcement-learning/model-based-rl.md)。
- **WSD（预热-稳定-衰减）计划** - 一种学习率计划，它会预热，在大部分训练中保持恒定的高速率，然后在最后急剧衰减；当总训练长度未预先确定时很方便。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。

## Z

- **零样本（学习/提示）** - 要求模型执行在推理时从未给出明确示例的任务，完全依赖于在训练/预训练期间学到的内容。与少射对比。涵盖于：[scope-and-methodology.md](scope-and-methodology.md)。

---

*（在编写构建中的每个后续文件时，此处会附加其他术语。）*
