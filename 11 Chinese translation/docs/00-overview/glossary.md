# 词汇表

此仓库中使用的每个术语的按字母顺序排列的动态列表。每个条目都有一个单行定义和一个指向该文件的链接，该文件在该文件中进行了深入介绍（如果任何单个文件“拥有”它）。该文件会附加到整个构建过程中 - 如果您发现文档文件中使用的术语未在此处列出，则这是一个错误；请添加。

## 如何使用此

此仓库中的文件根据 [scope-and-methodology.md](scope-and-methodology.md).该术语表的存在使您不必通过文件来重新查找定义，因此术语在整个仓库范围内保持一致（同一概念永远不应在两个不同的文件中具有两个不同的名称）。

## 按字母的术语计数（截至最后一次完整构建）

|一个 |乙| C | d |电子| F | G |哈 |我|克 |左 |中号 |哦|普 |问 |右 | S | T |你| V |西 | Z|总计 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | 3 | 7 | 7 | 5 | 4 | 9 | 4 | 4 | 3 | 5 | 10 | 10 3 | 5 | 3 | 5 | 9 | 5 | 1 | 2 | 3 | 1 | 105 | 105

---

## A

- **消融** - 一项实验，您删除或禁用系统的一个组件（一层、损失项、训练技巧）并重新测量性能，以找出该组件实际贡献了多少。如果删除它不会影响性能，那么它就没有发挥作用。涵盖于：[scope-and-methodology.md](scope-and-methodology.md)。
- **演员批评家** - 一种将策略（“演员”，选择动作）与学习价值函数（“批评家”，判断这些动作相对于期望有多好）相结合的强化学习方法，用于减少策略梯度方差。涵盖于：[policy-gradient-methods.md](../07-reinforcement-learning/policy-gradient-methods.md)。
- **优势（RL）** - 相对于该状态的基线期望，行动结果好坏多少，用于扩展策略梯度更新。涵盖于：[policy-gradient-methods.md](../07-reinforcement-learning/policy-gradient-methods.md)。
- **代理** — 描述了一种经过训练/用于采取多步骤操作（使用工具、调用 API、在减少逐步监督的情况下进行操作）以实现更长期目标的模型，而不是仅仅回答单个提示。涵盖于：[timeline-2023-present.md](../08-history/timeline-2023-present.md)。
- **对齐** — 训练模型按照人类意图和价值观（有帮助、诚实、无害）行事，而不是仅仅预测统计上可能的文本。涵盖于：[rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)。
- **全归约** - 一种集体通信操作，分布式训练作业中的每个设备最终都会得到所有设备上计算的某个数量（通常是梯度）的组合（例如求和/平均）值。涵盖于：[distributed-training.md](../05-training-methodology/distributed-training.md)。
- **注意力** - 一种计算查询和一组候选项目之间的相关性分数、将它们转换为权重（通常通过 softmax）并获取候选项目的加权和的机制；起源于 seq2seq 翻译（Bahdanau/Luong 注意力）并推广为 Transformer 自注意力。涵盖于：[rnn-lstm-gru.md](../03-deep-learning-architectures/rnn-lstm-gru.md)；另请参见[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。

## B

- **基准** - 标准化数据集/任务对（加上评分规则），用于在相同条件下相互比较模型。 “基准 X 上的 SOTA”意味着“特定标准化测试的最佳发布分数”，比“总体最佳模型”范围更窄。涵盖于：[scope-and-methodology.md](scope-and-methodology.md)。
- **偏差-方差权衡** — 模型假设错误导致的系统误差（偏差）与对所使用的特定训练样本的敏感性（方差）之间的紧张关系；像 bagging 这样的集成主要是减少方差。涵盖于：[ensemble-methods.md](../02-classical-ml/ensemble-methods.md)。
- **引导（RL 意义）** — 使用模型自身当前（不完美）的未来价值估计来更新价值估计，而不是等待完全观察到的结果。涵盖于：[value-based-methods.md](../07-reinforcement-learning/value-based-methods.md)。

## C

- **校准** — 模型的置信度（预测概率）与其实际经验准确性的匹配程度。一个经过良好校准的模型显示“80% 的置信度”在大约 80% 的情况下是正确的。涵盖于：[regularization-techniques.md](../01-foundations/regularization-techniques.md)。
- **灾难性遗忘** - 在新数据上训练模型时会降低或消除以前学习的能力，这是实际持续/在线学习的核心障碍。涵盖于：[open-problems.md](../09-roadmaps/open-problems.md)。
- **因果掩蔽** - 在自注意力中，阻止每个位置关注后面的位置（通过在 softmax 之前将它们的分数设置为负无穷大），因此模型无法“看到未来”它应该预测的。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **计算最优** — 在模型大小和数据大小之间分配固定的训练计算预算，最大限度地减少损失；龙猫论文的核心发现。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **信用分配问题** - RL 难以确定哪些较早的行为应该因较晚到达的奖励而得到赞扬（或责备）；几乎每个强化学习算法都有不同的答案。涵盖于：[value-based-methods.md](../07-reinforcement-learning/value-based-methods.md)。
- **交叉注意力** - 一种注意力变体，其中查询来自一个序列（例如解码器），键/值来自另一个序列（例如编码器的输出），让一个序列回顾另一个序列。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **交叉熵** — 衡量预测概率分布与真实目标分布之间差距的损失函数；等于真实分布的熵加上真实分布和预测分布之间的 KL 散度。分类和语言建模的标准损失。涵盖于：[loss-functions.md](../01-foundations/loss-functions.md)。

## D

- **数据泄漏** - 当有关目标变量的信息在预处理过程中无意中泄漏到特征中时，会以一种无法泛化的方式夸大训练/验证性能。涵盖于：[ensemble-methods.md](../02-classical-ml/ensemble-methods.md)。
- **DDIM（去噪扩散隐式模型）** — 一种以更少、更大、确定性步骤从经过训练的扩散模型中进行采样的技术，无需重新训练即可加速生成。涵盖于：[diffusion-models.md](../04-generative-models/diffusion-models.md)。
- **决策边界** — 特征空间中的划分面，分类器从预测一个类切换到另一个类；它的形状（直线、阶梯、平滑曲线）表征了分类器。涵盖于：[supervised-learning.md](../02-classical-ml/supervised-learning.md)。
- **退化** - 似然最大化文本解码（贪婪/束搜索）的失败模式，产生重复、通用或循环输出，而不是变化的、类似人类的文本。涵盖于：[autoregressive-generation.md](../04-generative-models/autoregressive-generation.md)。
- **判别模型与生成模型** — 判别模型直接学习类之间的边界 (P(class\|features))；生成模型对每个类的数据进行建模 (P(features\|class)) 并应用贝叶斯规则进行分类。涵盖于：[supervised-learning.md](../02-classical-ml/supervised-learning.md)。
- **解缠结** - 潜在表示的一种属性，其中单独的维度捕获单独的、独立的变化因素（例如，旋转与大小）；受到β-VAE的鼓励。涵盖于：[vaes.md](../04-generative-models/vaes.md)。
- **蒸馏（知识蒸馏）** - 训练一个较小的“学生”模型来重现较大的“教师”模型的输出分布，产生一个更小、更便宜的模型，该模型通常优于仅在硬标签上从头开始训练的模型。涵盖于：[pruning-and-distillation.md](../06-inference-optimization/pruning-and-distillation.md)。

## E

- **ELBO（证据下界）** - 生成模型（否则难以处理）数据似然的可计算下界，由重建项和 KL 散度正则化项组成； VAE 的培训目标。涵盖于：[vaes.md](../04-generative-models/vaes.md)。
- **嵌入** — 表示学习空间中的输入（单词、图像、用户等）的密集数值向量，其中几何距离对应于语义相似性。涵盖范围：[loss-functions.md](../01-foundations/loss-functions.md)（对比损失）；另请参见[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **熵** — 概率分布中固有不确定性（或信息内容）的度量；零表示完全可预测的结果，最大表示所有结果的可能性相同。涵盖于：[loss-functions.md](../01-foundations/loss-functions.md)。
- **𝔼（期望）** — 数量的平均值，按每个结果的可能性进行加权；机器学习论文中“对数据进行平均”或“对分布样本进行平均”的标准表示法。涵盖于：[gans.md](../04-generative-models/gans.md)。
- **体验重放** - 将 RL 代理过去的经验存储在缓冲区中，并对其随机采样的批次进行训练，而不是在其经历的时间相关流上进行训练，以稳定训练。涵盖于：[value-based-methods.md](../07-reinforcement-learning/value-based-methods.md)。

## F

- **前馈网络 (FFN)** — 每个 Transformer 块中的位置两层子网络（同样应用于每个词元）；保存了块的大部分参数，并且是专家混合 (Mixture-of-Experts) 所取代的许多专家。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **少样本（学习/提示）** - 在推理时（在提示中，而不是通过权重更新）为模型提供少量（通常为 1-100）示例输入/输出对，并期望它将模式泛化为新输入。与零射击对比。涵盖于：[scope-and-methodology.md](scope-and-methodology.md)。
- **FLOPs（浮点运算）** — 计算成本的衡量标准；用于量化模型处理词元、训练或运行推理所需的计算量。涵盖于：[mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md)。
- **流匹配/整流流** - 生成模型训练目标，学习沿（理想情况下是直的）路径将噪声传输到数据的速度场；与扩散密切相关，并且越来越多地用于大型图像/视频生成器。涵盖于：[diffusion-models.md](../04-generative-models/diffusion-models.md)。

## G

- **GELU / SwiGLU** — 现代 Transformer 的前馈子层中使用的平滑 (GELU) 和门控 (SwiGLU) 激活函数，代替旧的 ReLU。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **泛化** - 模型在未训练的数据上的表现如何，而不是仅仅记住其训练集。正则化的全部目的就是改善这一点。涵盖于：[regularization-techniques.md](../01-foundations/regularization-techniques.md)。
- **梯度** — 损失函数相对于每个模型参数的偏导数向量；点位于损失增加最陡的方向，因此训练步骤与其相反。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **梯度裁剪** - 重新调整梯度向量，使其范数永远不会超过固定阈值，以防止单个异常大的梯度破坏训练的稳定性。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **梯度爆炸** - 一种故障模式，其中梯度在通过多个层或时间步反向传播时呈倍增增长，产生巨大的、不稳定的更新。与消失的梯度形成对比。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **梯度消失** - 与梯度爆炸相反的故障模式：梯度在通过许多层/时间步向后传播时收缩到零，因此早期层停止学习。涵盖于：[rnn-lstm-gru.md](../03-deep-learning-architectures/rnn-lstm-gru.md)。
- **GroupNorm（组标准化）** — 一种标准化方案，可对每个示例的一组特征通道进行标准化；像 LayerNorm 一样与批次无关，并且在扩散模型 U-Net 中很常见。涵盖于：[regularization-techniques.md](../01-foundations/regularization-techniques.md)。
- **GRPO（组相对策略优化）** — 一种针对 LLM 的 RL 算法，通过使用一组采样响应的平均分数作为基线来消除 PPO 的单独值模型；在推理模型训练中表现突出。涵盖于：[rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)。
- **指导尺度** - 在扩散模型中，可调强度参数，控制通过无分类器指导将生成推向条件信号（例如文本提示）的强度。涵盖于：[diffusion-models.md](../04-generative-models/diffusion-models.md)。

## H

- **幻觉** - 当模型生成流畅、听起来自信的输出时，实际上是不正确的；这是训练目标优化合理性而不是验证事实的结果。涵盖于：[open-problems.md](../09-roadmaps/open-problems.md)。
- **Hessian** — 函数的二阶导数矩阵；描述损失景观的局部曲率。太大（参数²）而无法直接计算深度网络，这就是为什么大多数训练仅使用一阶导数（梯度）信息的原因。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **隐藏状态** - 由循环网络在每个时间步维护和更新的向量，总结了迄今为止网络在序列中看到的所有相关内容。涵盖于：[rnn-lstm-gru.md](../03-deep-learning-architectures/rnn-lstm-gru.md)。
- **Huber 损失** — 一种回归损失，其对于小错误的行为类似于 MSE，对于大错误的行为类似于 MAE，将平滑梯度与对异常值的鲁棒性结合起来；用于稳健回归和 DQN 值目标。涵盖于：[loss-functions.md](../01-foundations/loss-functions.md)。

## 我

- **隐式正则化** — 正则化效果是标准训练选择（SGD 梯度噪声、提前停止、有限精度）的副作用，而不是故意添加的；这是非常大的模型过度拟合少于其参数数量所建议的部分原因。涵盖于：[regularization-techniques.md](../01-foundations/regularization-techniques.md)。
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
- **Logits** — 分类器在转换为概率之前生成的原始非标准化分数（通常通过 softmax）。涵盖于：[loss-functions.md](../01-foundations/loss-functions.md)。
- **损失景观** — 通过针对模型参数的每种可能设置绘制损失函数值而形成的高维表面；训练就是在这个表面上寻找低点的过程。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **低秩分解** - 将大矩阵更新近似为两个小得多（薄）矩阵的乘积，大大减少可训练参数； LoRA 背后的机制。涵盖于：[finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md)。

## M

- **马尔可夫决策过程 (MDP)** — 马尔可夫属性下的顺序决策的标准形式化：状态、行动、奖励和策略。涵盖于：[value-based-methods.md](../07-reinforcement-learning/value-based-methods.md)。
- **马尔可夫性质** - 假设未来仅通过当前状态而不是完整的历史依赖于过去；马尔可夫链、HMM 和 MDP 的定义假设。涵盖于：[probabilistic-models.md](../02-classical-ml/probabilistic-models.md)；另请参见[value-based-methods.md](../07-reinforcement-learning/value-based-methods.md)。
- **最小最大游戏/目标** - 一种优化设置，其中两方具有直接相反的目标（一个最大化，一个最小化相同的表达式）； GAN 背后的训练框架。涵盖于：[gans.md](../04-generative-models/gans.md)。
- **专家混合 (MoE) / 路由器** — 一种架构模式，其中存在许多并行子网络（“专家”），但小型路由器网络仅选择少数几个来处理每个词元，从而将总参数计数与每个词元的计算成本解耦。涵盖于：[mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md)。
- **模式崩溃** - 一种 GAN 故障模式，其中生成器仅产生一小部分非多样化的输出，这些输出恰好欺骗了当前鉴别器，而不是覆盖真实数据分布的全部多样性。涵盖于：[gans.md](../04-generative-models/gans.md)。
- **模式覆盖与模式寻求** — KL 散度的两个方向产生的两种行为：模式覆盖（前向 KL）扩展概率以覆盖所有目标分布；模式搜索（反向 KL）锁定一种模式并忽略其他模式。涵盖于：[loss-functions.md](../01-foundations/loss-functions.md)。
- **模型崩溃** - 当模型在其他模型生成的数据上迭代训练而没有充分的真实、多样化的数据基础时，可能会发生质量/多样性的下降。涵盖于：[curriculum-and-data-strategies.md](../05-training-methodology/curriculum-and-data-strategies.md)。
- **无模型（相对于基于模型）RL** — 无模型方法（Q 学习、DQN、策略梯度）直接从经验中学习，无需环境动态的显式模型；基于模型的方法学习或使用这样的模型。涵盖于：[value-based-methods.md](../07-reinforcement-learning/value-based-methods.md)；另请参见[model-based-rl.md](../07-reinforcement-learning/model-based-rl.md)。
- **动量** - 一种优化技术，可累积过去梯度的运行平均值和该平均方向上的步数，而不是原始当前梯度，从而平滑轨迹。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **Muon** — 2024 年优化器，可对网络的 2D 权重矩阵应用正交化、矩阵感知更新； AdamW 的新兴挑战者，用于使用较低优化器状态内存进行大规模预训练。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。

## O

- **优化器状态** - 优化器维护的额外每个参数缓冲区（例如 Adam 的两矩估计）；在法学硕士规模上，这可能超过模型自身权重的大小，并且是内存分片技术的主要目标。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **过度拟合** - 当模型与训练数据（包括其噪声/怪癖）拟合得如此紧密时，其在新的、看不见的数据上的表现比拟合程度较低的模型更差。与欠拟合对比。涵盖于：[regularization-techniques.md](../01-foundations/regularization-techniques.md)。
- **过度参数化** — 描述了一个网络，其参数多于其学习功能严格所需的参数；剪枝的经验基础（可以删除许多权重，而精度损失很小）。涵盖于：[pruning-and-distillation.md](../06-inference-optimization/pruning-and-distillation.md)。

## P

- **PEFT（参数高效微调）** — 微调方法（LoRA、适配器、前缀/提示调整）的总称，冻结大部分预训练模型的权重并仅训练少量附加/修改参数。涵盖于：[finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md)。
- **Perplexity** — 语言模型交叉熵损失 (e^loss) 的人类可解释的变换；粗略地说，“该模型的不确定性就像在每个词元的众多选项中进行统一选择一样。”越低越好。涵盖于：[loss-functions.md](../01-foundations/loss-functions.md)。
- **策略** - 在强化学习中，模型/函数被训练以选择动作（例如，生成哪个词元）； RLHF 的 RL 阶段将优化语言模型作为策略。涵盖于：[rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)；另请参见[value-based-methods.md](../07-reinforcement-learning/value-based-methods.md)。
- **后崩溃** - 一种 VAE 故障模式，其中潜在代码变得无信息（编码器仅输出先验代码），因为强大的解码器学会忽略它，同时仍将 KL 项驱动为零。涵盖于：[vaes.md](../04-generative-models/vaes.md)。
- **先验** - 在观察数据之前假设变量的概率分布（例如，VAE 假设的潜在向量的标准正态分布），用作推理正则化的参考/目标。涵盖于：[vaes.md](../04-generative-models/vaes.md)。

## Q

- **QK-norm** - 在计算分数之前对注意力内的查询和关键向量进行归一化，在大型转换器中使用，以防止注意力逻辑增长到极端程度并破坏训练的稳定性。涵盖于：[regularization-techniques.md](../01-foundations/regularization-techniques.md)。
- **量化** - 降低用于表示模型权重/激活的数值精度（例如，从 16 位到 4 位），缩小内存占用量，并通常以一定的准确性成本加快推理速度。涵盖于：[quantization.md](../06-inference-optimization/quantization.md)。
- **查询、键、值 (Q/K/V)** — 每个位置在自注意力输入中的三个学习投影：查询表示位置正在寻找什么，键表示位置广告的内容，值表示位置在关注时提供的信息。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。

## R

- **感受野** — 给定神经元输出有效影响的原始输入区域；在 CNN 的更深层中变得更大。涵盖于：[cnn-family.md](../03-deep-learning-architectures/cnn-family.md)。
- **ReLU（修正线性单元）** — 激活函数 max(0, x)；负输入输出零，正输入保持不变。 CNN 中的标准，并在其他地方广泛使用，因为它的梯度不会因正输入而缩小。涵盖于：[cnn-family.md](../03-deep-learning-architectures/cnn-family.md)。
- **重新参数化技巧** - 将随机样本重写为学习参数加上固定外部随机性的确定性函数，因此梯度可以通过其他不可微分的采样步骤反向传播。涵盖于：[vaes.md](../04-generative-models/vaes.md)。
- **奖励黑客**——当一项政策利用学习奖励模型中的缺陷来获得高分时，却实际上没有产生人类认为良好的输出。涵盖于：[rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)。
- **奖励模型** - 一种经过训练来预测人类偏好判断的模型，在 RL 训练期间用作人类反馈的快速自动化代理。涵盖于：[rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)。

## S

- **样本效率** - 代理或模型需要多少真实世界的交互/数据才能达到给定的性能水平；相对于无模型方法，基于模型的强化学习旨在改进这一点。涵盖于：[model-based-rl.md](../07-reinforcement-learning/model-based-rl.md)。
- **缩放法则** - 语言模型的损失与其参数计数、数据集大小和训练计算之间的经验幂律关系，用于规划大型训练运行。涵盖于：[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)。
- **评分函数** — 数据分布的对数概率相对于数据本身的梯度；在多个噪声级别上学习这一点是基于分数的生成模型的基础，与 DDPM 的噪声预测目标密切相关。涵盖于：[diffusion-models.md](../04-generative-models/diffusion-models.md)。
- **自我监督学习** - 使用从输入数据本身自动派生的标签进行训练（例如，预测屏蔽或下一个标记），而不需要人工注释。涵盖于：[pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md)。
- **分片** — 将大型对象（模型参数、梯度、优化器状态）分割成分布在多个设备上的片段，因此没有一个设备需要保存整个对象。涵盖于：[distributed-training.md](../05-training-methodology/distributed-training.md)。
- **Softmax** — 通过取幂和归一化将原始分数（logits）向量转换为有效概率分布（均为正数，总和为 1）的函数。涵盖于：[loss-functions.md](../01-foundations/loss-functions.md)。
- **SOTA（最先进的技术）** — 在给定时间点上给定基准的最佳公开报告结果。一个移动的目标，而不是一个固定的方法——“SOTA”描述的是排行榜位置，而不是任何一种算法。涵盖于：[scope-and-methodology.md](scope-and-methodology.md)。
- **稀疏奖励** - 一种 RL 设置，其中反馈仅在长序列操作结束时可用（例如，仅当完整的 LLM 响应完成时）而不是连续的，这使得学分分配更加困难。涵盖于：[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)。
- **稀疏性** — 网络权重为零（或被删除）的比例；修剪增加稀疏性以减小模型大小。涵盖于：[pruning-and-distillation.md](../06-inference-optimization/pruning-and-distillation.md)。

## T

- **温度** — 采样期间在 softmax 之前除以 logits 的标量；低于 1 会锐化（更具确定性）模型的输出分布，高于 1 会使其变得平坦（更加随机/多样化）。涵盖于：[autoregressive-generation.md](../04-generative-models/autoregressive-generation.md)。
- **时间差异（TD）误差** - 在强化学习中，价值估计和更好的“目标”估计之间的差距（奖励加上折扣的下一状态值）；驱动 Q 学习和价值模型更新的信号。涵盖于：[value-based-methods.md](../07-reinforcement-learning/value-based-methods.md)。
- **测试时计算** - 在推理时花费的计算（例如，生成更长的中间推理）以提高输出质量，作为与预训练计算不同的缩放轴。涵盖于：[long-term-outlook.md](../09-roadmaps/long-term-outlook.md)。
- **词元** — 语言模型将其视为单个输入/输出单元的文本块（通常是单词或子单词片段）。涵盖于：[autoregressive-generation.md](../04-generative-models/autoregressive-generation.md)。
- **Tokenizer / BPE（字节对编码）** — 通过将频繁的相邻对贪婪地合并为单个标记，将原始文本转换为模型消耗的固定子词词汇的组件（和主导算法）。涵盖于：[pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md)。

## U

- **欠拟合** - 当模型太简单或训练不足而无法捕获训练数据中的真实模式时，在训练和新数据上都表现不佳。与过拟合对比。涵盖于：[regularization-techniques.md](../01-foundations/regularization-techniques.md)。

## V

- **价值模型/评论家** - 在强化学习中，一种学习估计器，用于评估状态（或状态动作）在预期未来奖励方面的好坏；由 Actor-Critic 方法和 PPO 使用，并由 GRPO 删除。涵盖于：[policy-gradient-methods.md](../07-reinforcement-learning/policy-gradient-methods.md)。
- **VQ-VAE（矢量量化 VAE）** — 具有离散潜在的 VAE 变体（每个潜在位置都捕捉到学习密码本中最近的条目），使图像/音频能够通过自回归 Transformer 将图像/音频建模为词元序列。涵盖于：[vaes.md](../04-generative-models/vaes.md)。

## W

- **热身** - 以较小的学习率开始训练，并在第一个步骤中逐步提高，以避免梯度/方差估计仍然有噪声时不稳定。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。
- **世界模型** - 环境如何演变的学习或给定表示（给定当前状态和行动的下一个状态和奖励），用于规划或生成模拟体验。涵盖于：[model-based-rl.md](../07-reinforcement-learning/model-based-rl.md)。
- **WSD（预热-稳定-衰减）计划** - 一种学习率计划，它会预热，在大部分训练中保持恒定的高速率，然后在最后急剧衰减；当总训练长度未预先确定时很方便。涵盖于：[optimization-algorithms.md](../01-foundations/optimization-algorithms.md)。

## Z

- **零样本（学习/提示）** - 要求模型执行在推理时从未给出明确示例的任务，完全依赖于在训练/预训练期间学到的内容。与少射对比。涵盖于：[scope-and-methodology.md](scope-and-methodology.md)。

---

*（在编写构建中的每个后续文件时，此处会附加其他术语。）*
