# 剪枝与蒸馏

剪枝和蒸馏是除量化(见 [quantization.md](quantization.md))之外,另外两种让训练好的模型运行起来更便宜的方法:直接减少模型的"体量"(剪枝),或训练一个较小的模型去模仿一个较大的模型(蒸馏)。对 LLM 而言,蒸馏在压缩之外还发展出了第二个角色:下文介绍的**在线策略蒸馏(on-policy distillation)**,已成为把多个专家模型的能力合并到一个模型中的核心后训练工具。

## 幅值剪枝

**名称与定义。** 幅值剪枝(magnitude pruning)移除绝对值较小的权重,理由是小幅值权重对网络输出的贡献相对较小,可以被置零(移除)而对精度影响很小。

**核心机制。** 在训练之后(或训练期间),按绝对值大小对所有权重排序,把其中最小的一部分精确地置为零——通常随后还要再微调一段时间,让剩余权重做出调整,找回因移除连接而损失的部分精度。这一过程可以迭代进行(剪一点、微调、再剪、再微调),从而以比一次性全部剪掉更小的精度损失达到更高的稀疏度(被移除权重的比例)。

**为什么重要。** 幅值剪枝在许多架构上都证明了,神经网络对给定任务往往是大幅**过参数化(over-parameterized)**的——很大一部分权重(在一些被广泛报道的案例中远超一半)可以被移除而几乎不损失精度,这说明训练好的网络中,原始参数量的相当一部分对其所学到的函数并非严格必要。

## 结构化剪枝与非结构化剪枝

**非结构化剪枝(unstructured pruning)**在任何最小权重所在之处移除单个权重,不考虑任何更大的结构模式——这可以在精度损失极小的情况下达到很高的稀疏度,但由此产生的零权重分布不规则、零散,这与现代加速器硬件(GPU/TPU)真正获得速度的方式很不匹配:它们是为大规模、规则、稠密的矩阵乘法而构建的,标准硬件通常无法跳过零散分布的单个零值来真正节省计算时间,只有专门的稀疏计算硬件/kernel 才能做到,而且即便如此通常也只支持特定的稀疏模式。

**结构化剪枝(structured pruning)**则移除整个结构单元——整个神经元、整个通道(在 CNN 中——见 [cnn-family.md](../03-deep-learning-architectures/cnn-family.md))或整个注意力头(在 Transformer 中——见 [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md))——而不是零散的单个权重。这会得到一个更小、但仍完全稠密、形状规则的网络(例如神经元更少的一层,而不是神经元数量不变、只是部分连接被置零),无需专门的稀疏计算支持就能在标准硬件上获得真正的加速;代价是它通常不如非结构化剪枝那样精准(移除整个神经元/通道/头,比移除逐个挑选的权重更粗糙、破坏性更大),因此在参数量削减相当的情况下,一般会带来稍多的精度损失。

**实际权衡。** 这是剪枝中的核心实践矛盾:非结构化剪枝在纸面上"每移除一个参数损失的精度"更少,但结构化剪枝在常见硬件上真正转化为实际加速的可能性要大得多——这就是为什么在生产部署中,尽管结构化剪枝在原始参数移除方面理论上"效率较低",它却往往更受青睐。

## 知识蒸馏

**名称与定义。** 知识蒸馏(knowledge distillation)训练一个较小的"学生"模型去复现一个较大的、已训练好的"教师"模型的行为,而不是(或除了)直接用原始的硬标签训练学生。

**起源。** Hinton、Vinyals、Dean,"Distilling the Knowledge in a Neural Network"(2015),建立在更早的相关"模型压缩"思想之上(Bucilă、Caruana、Niculescu-Mizil,2006)。

**核心机制——师生框架。** 蒸馏不只是训练学生去匹配硬性的、one-hot 形式的正确标签,而是训练学生去匹配教师在所有类别/token 上的完整输出概率分布("软标签"或"软目标")——通常会在 softmax 中使用升高的**温度(temperature)**(温度在另一种语境中的作用见 [autoregressive-generation.md](../04-generative-models/autoregressive-generation.md)——这里它用在训练时而非采样时)来"软化"教师的分布,让它更多地暴露出对所有非首选项的相对置信度信息,而不只是它认为哪个答案是正确的。

```
Loss = α · CrossEntropy(student, true_label) + (1 − α) · CrossEntropy(student, teacher_soft_labels)
```

逐项解读:教师的完整输出分布所包含的信息比它的单一最高预测更多——例如,一个对图像分类的教师可能把大部分概率给"狗",但给"狼"的概率也明显高于"飞机",而这种相对相似性信息(有时被非正式地称为"暗知识(dark knowledge)")本身就是学生可以学习的有用训练信号,超出了"正确答案是狗"这一点。训练学生去匹配这个更完整的分布(与原始硬标签损失按 α 加权混合,或取而代之),能让较小的学生学到对较大教师所学内容的更细致近似,通常得到的小模型比同样大小、只用原始硬标签训练的模型表现更好。

**为什么重要。** 蒸馏提供了一种通用的、与任务无关的方法,把大模型压缩成更小、更便宜的模型用于部署,而不需要针对特定模型的压缩技术;它已被广泛应用于通用模型压缩,而且——值得注意的是——据报道,它也是用来训练更小、更快、能以一小部分推理成本逼近更大前沿模型能力的模型的若干技术之一。

## 在线策略蒸馏

**名称与定义。** 在线策略蒸馏(on-policy distillation)用*学生自己生成*的文本来训练学生语言模型:学生采样自己的回复,教师模型为这些回复的每一个 token 打分——给出它自己在该位置的概率分布——学生则被训练去匹配这个分布。

**起源。** Agarwal 等,"On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"(GKD,即"广义知识蒸馏 Generalized Knowledge Distillation",2023;ICLR 2024);Gu 等的 "MiniLLM"(2023)也用学生自己的样本、以反向 KL 训练学生。2025 年 Thinking Machines Lab 的一篇博客文章让这一技术广为人知,并将其定位为后训练中一种比 RL 更便宜的替代方案。

**离线策略蒸馏的问题。** 蒸馏 LLM 的经典方式是**离线策略(off-policy)**的:让教师写出大量示例输出(或为一个固定数据集打分),再微调学生去模仿它们。一个著名例子是 DeepSeek-R1(2025 年 1 月):通过在 R1 生成的推理轨迹上微调,它的推理能力被蒸馏进了若干个较小的稠密模型。这种做法有效,但存在一个结构性缺陷:学生始终只在*教师的*轨迹上训练。推理时它生成的是自己的轨迹,一旦它犯了教师绝不会犯的错误,就进入了训练中从未见过、也从未练习过如何挽回的境地——于是错误层层累积(这与任何对专家示范的模仿所面临的"暴露偏差(exposure bias)"是同一个问题)。

**核心机制。** 在线策略蒸馏把"由谁来写训练文本"反了过来:

```
repeat:
    prompt x  ← sample from training prompts
    y         ← student generates a full response to x              (on-policy: the student's own text)
    for each position t in y:
        teacher_probs_t ← teacher's next-token distribution given (x, y_<t)   (one teacher forward pass, no generation)
        student_probs_t ← student's next-token distribution given (x, y_<t)
    loss = Σ_t  KL( student_probs_t ‖ teacher_probs_t )               (reverse KL, per token)
    update student to reduce loss
```

逐项解读:KL 散度衡量两个概率分布有多不同;**反向 KL** `KL(student ‖ teacher)` 会惩罚学生把概率放在教师认为不太可能的 token 上。它是"寻模(mode-seeking)"的:一个无法复现教师全部行为的小学生,会被推动去专注于教师最可能的行为,而不是把自己摊得太薄——这很适合把大模型压缩进小模型。(标准蒸馏使用正向 KL `KL(teacher ‖ student)`,它反而会推动学生去覆盖教师的*所有*选项。)由于学生是在自己的轨迹上接受评分的,它学到的是*从学生自己实际会进入的状态出发*——包括在它自己犯错之后——教师会怎么做。GKD 对此做了推广,允许混合学生生成的数据与固定数据,并可选择不同的散度。

**与 RL 的联系。** 这看起来很像强化学习(见 [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)):学生采样自己的输出,并根据输出的好坏来更新。区别在于奖励。基于可验证奖励的 RL 对每个完整回复只给一个稀疏信号("最终答案对不对?");在线策略蒸馏给出的是**密集的、逐 token 的**信号——本质上是在每一步都告诉学生"教师在多大程度上认同这个 token?"。这让每个样本所含的信息量大得多,这也是为什么要达到某种教师已具备的能力时,它往往比 RL 便宜得多。代价是:学生无法像 RL 超越其起点那样超越教师;而且逐 token 目标要求教师和学生共享同一个分词器(或在两者之间建立某种映射)。

**多教师在线策略蒸馏——2026 年的后训练整合范式。** 2026 年,一种被广泛报道的前沿做法是用在线策略蒸馏来*合并*能力,而不是缩小模型:

1. 用 RL 训练若干个**领域专家**模型——例如数学、代码、智能体工具使用和通用对话各一个。
2. 把它们全部蒸馏进一个学生:学生针对某个领域的提示采样自己的回复,由对应领域的专家教师给出逐 token 目标(在学生自己的轨迹上使用反向 KL)。

```
   specialist RL runs:   [math teacher]  [code teacher]  [agentic teacher]  [general teacher]
                                │               │                │                 │
   student answers a code prompt ──► code teacher scores every token ──► reverse-KL update
   student answers a math prompt ──► math teacher scores every token ──► reverse-KL update
                                     ...one student absorbs all specialties
```

其动机是避免相互干扰:把所有领域放进同一次混合 RL 训练,往往会让各项能力此消彼长;而分开进行的专家训练可以各自单独调优。据报道的例子包括 NVIDIA 的 Nemotron 3 Ultra(从十余个教师蒸馏而来),以及 GLM-5 这样的分阶段流程(SFT → 推理 RL → 智能体 RL → 通用 RL,各阶段之间进行蒸馏);Lambert 与 Timbers 在 2026 年 6 月发表的后训练实践综述把多教师在线策略蒸馏描述为当前的前沿范式。注意,这里的教师可能与学生一样大——蒸馏在此充当的是*整合*工具,而不是压缩工具。这些工业流程的细节都来自各实验室自己的报告,尚未被独立复现。

**当前状态(2026 年底)。** 已在前沿模型和开放权重模型的后训练中站稳脚跟;离线策略蒸馏(在教师生成的数据上微调,如 R1 的蒸馏模型)在低成本产出小模型方面仍很常用,因为一旦教师数据生成完毕,训练学生就不再需要调用教师。

## 生产模型家族中的当前实际应用

截至 2026 年底,本节涵盖的三类技术(量化、剪枝和蒸馏)通常是**组合**使用的,而不是互斥的备选方案——一个典型的生产部署流程可能先把大模型蒸馏成较小的架构,再对得到的小模型做进一步量化以供部署,在硬件/延迟约束值得付出精度代价时,再额外加上一步结构化剪枝。尤其是蒸馏,已成为产出前沿模型家族中较小的"快速版"或"迷你版"变体的一项特别突出、被广泛报道的技术,这些变体面向对延迟和成本敏感、并不完全需要全尺寸旗舰模型能力的使用场景。

## 优势与局限

- **剪枝**——优势:能以适度的精度损失达到很高的稀疏度(尤其是精心实施的结构化剪枝);同时降低内存和(对结构化剪枝而言)实际计算成本。局限:没有专门支持时,非结构化剪枝往往无法转化为真正的硬件加速;与一次性压缩方法相比,迭代的"剪枝—微调"循环增加了工程和计算开销。
- **在线策略蒸馏**——优势:避免了模仿教师所写文本带来的训练/推理失配;密集的逐 token 信号使它在达到教师能力方面比 RL 样本效率高得多;是合并多个专家模型的一种干净方式。局限:训练中每个学生样本都需要一次教师前向传播;逐 token 目标需要共享(或可映射的)分词器;学生受限于其教师。
- **蒸馏**——优势:通用、与架构无关的压缩方法;能产出真正更小、更快的模型,而不只是同一架构的更稀疏版本;得到的学生模型往往比同样大小、只用硬标签从头训练的模型更准确。局限:学生训练过程中需要能运行教师模型(理想情况下还要采样大量教师输出),增加了压缩过程本身的计算成本;学生的能力从根本上受限于教师能教给它什么,不会自发超越一个选得好的教师自身的能力。

## 对比表

| 技术 | 减少的是什么 | 无需特殊支持也能获得真实硬件加速? | 是否需要重新训练/微调? |
|---|---|---|---|
| 非结构化幅值剪枝 | 参数量(稀疏) | 一般不能 | 是(恢复性微调) |
| 结构化剪枝 | 参数量(稠密、更小) | 能 | 是(恢复性微调) |
| 知识蒸馏 | 模型大小(新的更小架构) | 能(整个模型更小) | 是(完整的学生训练过程) |
| 在线策略蒸馏(LLM) | 模型大小,或独立专家模型的数量(整合) | 能(如果学生更小) | 是(学生采样,教师为每个 token 打分) |

## 与其他算法的关系

- 剪枝和蒸馏在生产压缩流程中通常与量化(见 [quantization.md](quantization.md))组合使用。
- 蒸馏的软标签/温度机制与自回归采样中使用的温度密切相关,但用途不同——见 [autoregressive-generation.md](../04-generative-models/autoregressive-generation.md)。
- 蒸馏还在一种相关但不同的意义上被用作数据生成策略(自蒸馏),而不纯粹是一种压缩技术——见 [curriculum-and-data-strategies.md](../05-training-methodology/curriculum-and-data-strategies.md)。
- 在线策略蒸馏与面向 LLM 的 RL 密切相关——同样是"从学生采样、再更新"的循环,只是用密集的逐 token 教师信号取代了稀疏奖励——见 [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md);多教师蒸馏中的专家教师本身通常也是用 RL 训练的,整体后训练流程见 [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)。

## 来源

- Bucilă, Caruana, Niculescu-Mizil, "Model Compression" (2006)
- Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network" (2015)
- Han, Pool, Tran, Dally, "Learning both Weights and Connections for Efficient Neural Networks" (2015) [幅值剪枝]
- Kim, Rush, "Sequence-Level Knowledge Distillation" (2016) [在教师生成的序列上蒸馏]
- Agarwal et al., "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes" (GKD;2023,ICLR 2024)
- Gu et al., "MiniLLM: Knowledge Distillation of Large Language Models" (2023) [在学生样本上做反向 KL 蒸馏]
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025) [蒸馏出的小型稠密模型——离线策略的例子]
- Thinking Machines Lab, "On-Policy Distillation" (博客文章,2025)
- Lambert & Timbers, 后训练实践综述 (2026 年 6 月) [多教师在线策略蒸馏作为 2026 年的前沿范式]
- NVIDIA, Nemotron 3 Ultra (2026) 与智谱 AI, GLM-5 (2026) 技术报告 [多教师/跨阶段蒸馏;据报道]
