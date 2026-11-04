# 状态空间模型

状态空间模型(State Space Models,SSM)是一类作为注意力替代方案提出的序列架构,目标是在保留 Transformer 长程建模能力(见 [transformer-architecture.md](transformer-architecture.md))的同时,避开注意力的二次方计算成本。本文件也涵盖与之密切相关的 **linear attention(线性注意力)** 模型家族(包括 **Gated DeltaNet**),因为到 2024-2025 年,这两条研究路线已基本汇合为同一个设计空间:一种带有固定大小状态的循环层。按照本知识库的质量标准,截至 2026 年末,诚实的结论有两面:**纯** SSM / 纯线性注意力模型在前沿仍然罕见,但 **混合(hybrid)架构**——以线性注意力层为主、穿插少量全注意力层——已经从研究走进了生产级旗舰模型。

## 动机:注意力的二次方成本问题

**问题所在。** 自注意力会为序列中每一对位置计算一个相似度分数(即 [transformer-architecture.md](transformer-architecture.md) 中介绍的 Q·Kᵀ 项)。对于长度为 n 的序列,所需的计算量和内存都与 n² 成正比——序列长度翻倍,成本变为四倍。对于短序列这没什么问题,但随着前沿 LLM 的上下文长度增长到数十万个词元,二次方项成为主导成本,训练时如此,推理时尤甚,因为它与 KV 缓存相互作用(见 [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md))。

**SSM 的主张。** 如果有一种序列架构,能像 RNN 的循环那样(见 [rnn-lstm-gru.md](rnn-lstm-gru.md)),以相对于已处理序列长度而言*恒定*的时间和内存处理每个新词元,同时又没有 RNN 的梯度消失问题和缓慢、严格串行的训练,那会怎样?从实际意义上说,状态空间模型就是一次"把循环带回来"的尝试——只是内部机制经过了精细得多的设计——其明确目的就是让成本随序列长度线性(而非二次方)增长。

## S4

**名称与定义。** S4(Structured State Space Sequence model,结构化状态空间序列模型)把一个经典的连续时间控制理论概念——线性状态空间模型(隐藏状态在输入驱动下按线性微分方程随时间连续演化)——改写成可用作深度学习序列层的形式。

**起源。** Gu、Goel、Ré,"Efficiently Modeling Long Sequences with Structured State Spaces"(2021/2022)。

**核心机制(概念层面)。** 连续时间状态空间模型由两条更新规则定义(这里写成离散化形式,因为神经网络处理的是离散的词元/时间步序列):

```
hₜ = A · h_{t-1} + B · xₜ
yₜ = C · hₜ
```

逐项解读:它在结构上与普通 RNN 的循环很像(见 [rnn-lstm-gru.md](rnn-lstm-gru.md))——隐藏状态 hₜ 由上一个隐藏状态和当前输入更新得到。关键区别在于:S4 给矩阵 A 施加了精心设计的数学结构(具体来说,让它对应一种良好的"记忆"——HiPPO 矩阵,出自同一研究组更早的相关工作,其设计目标是让隐藏状态能在很长的范围内较好地表示输入历史的压缩版本);并且——对实用性至关重要——由于这个循环是*线性*的(不同于 RNN 的循环在每一步都要经过 tanh 之类的非线性),训练时整条序列的输出可以一次性写成对整个输入的**卷积**来计算(在数学上等价于展开循环,但可以并行计算,避开了 RNN 严格串行训练的瓶颈),而在推理时它仍可作为真正逐步执行的循环来使用,使每个词元的成本和内存保持恒定。

**为什么重要。** 在专门用来考验长程依赖建模的长序列基准上,S4 证明了结构设计得当的线性循环可以在超长序列上显著优于 Transformer 和普通 RNN,同时成本随序列长度线性(而非二次方)增长——这是"无注意力架构可能在长上下文建模上具有竞争力"的第一个有力实证。

## Mamba——选择性状态空间

**名称与定义。** Mamba 在 S4 的基础上,让状态空间模型的核心矩阵(A、B、C 以及离散化步长)变为**依赖输入**的——根据当前输入动态计算,而不是固定不变(像 S4 那样对每个位置都相同)——作者把这一性质称为**选择性(selectivity)**。

**起源。** Gu、Dao,"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"(2023)。

**选择性为什么重要。** S4 的 A/B/C 矩阵固定不变,意味着模型对每个位置"记住多少、遗忘多少"的处理方式完全相同,与内容无关——这是实实在在的局限:理想情况下,模型应当根据内容,把关键词元(比如只出现过一次的人名)记得比填充性词元久得多,就像注意力基于内容的加权所允许的那样(见 [transformer-architecture.md](transformer-architecture.md))。让循环的参数成为当前输入的函数,Mamba 就能逐词元地有选择地决定往隐藏状态里写入多少、保留多少——从而大幅缩小 SSM 与注意力在基于内容加权方面的表达能力差距。代价是,这种选择性破坏了让 S4 易于并行训练的"等价于卷积"这一简洁技巧,因此 Mamba 改为依赖一种定制的、硬件感知的并行扫描(parallel scan)算法(对该循环的高效并行实现,专门针对 GPU 设计),以便在循环依赖输入的情况下仍保持训练高效。

**为什么重要。** Mamba 在语言建模基准上取得了与同等规模 Transformer 相当的结果,同时保留了线性时间、每词元恒定内存的推理——这一颇具说服力的结果重新点燃了研究界对无注意力架构的认真兴趣。

**Mamba-2(2024)。** Dao 与 Gu 的后续工作"Transformers are SSMs"(2024)把 Mamba 的状态矩阵简化为每一步一个"标量乘单位矩阵"的形式,并证明由此得到的选择性 SSM 在数学上就是一种*带衰减的线性注意力*——即"状态空间对偶性(state space duality)"。这正是通往下一节的桥梁:从 2024 年起,"SSM"和"线性注意力"最好被理解为描述同一类固定状态循环层的两套术语。下文所述的若干生产级混合模型中,所用的 SSM 层就是 Mamba-2。

## RWKV

**名称与定义。** RWKV(取自其循环中四个主要参数矩阵的首字母:Receptance、Weight、Key、Value)是另一种基于线性循环的架构,其明确的设计目标是把 Transformer 式的可并行训练与 RNN 式的恒定内存推理结合起来。

**起源。** Peng 等,"RWKV: Reinventing RNNs for the Transformer Era"(2023),最初是一个开源社区项目。

**核心机制(简述)。** RWKV 使用一种类似线性注意力的机制,带有显式的衰减项(较旧的信息会以结构化方式随时间降低权重);与 S4 类似,它在训练时可按可并行的形式计算,在推理时则可作为高效的循环运行。

**现状。** 一个规模较小但活跃的开源谱系,持续迭代;被一些看重其推理效率的开放权重模型所采用,但尚未达到下文所述生产级混合模型中 Mamba 式层或 Gated DeltaNet 那样的主流采用程度。

## 线性注意力与 Gated DeltaNet

**名称与定义。** **线性注意力(linear attention)** 是一类注意力替代方案:去掉 softmax,使注意力可以改写成在一个固定大小的矩阵"记忆"上运行的循环。**Gated DeltaNet**(GDN)是这一家族在 2024 年出现的成员,到 2026 年末已成为生产级 LLM 中最受瞩目的线性注意力层之一:它把 **delta 规则(delta rule)** 写入(覆盖记忆中某个键当前存储的内容,而不是简单地往上叠加)与 **门控衰减(gated decay)**(一个学习得到、依赖输入的遗忘因子)结合在一起。

**起源。** 线性注意力:Katharopoulos 等,"Transformers are RNNs"(2020)。delta 规则视角:Schlag、Irie、Schmidhuber,"Linear Transformers Are Secretly Fast Weight Programmers"(2021),后由 Yang 等(2024,"Parallelizing Linear Transformers with the Delta Rule over Sequence Length")使其能够大规模高效训练。门控/衰减:Gated Linear Attention(Yang 等,2023)与 Mamba-2(2024)。Gated DeltaNet 本身:Yang、Kautz、Hatamizadeh,"Gated Delta Networks: Improving Mamba2 with Delta Rule"(2024;ICLR 2025)。

### 第 1 步——去掉 softmax,注意力就变成了循环

在普通的因果注意力中,位置 t 的输出是此前所有 value 向量经 softmax 加权后的和。正是 softmax 迫使你保留所有过去的 key 和 value(即 KV 缓存):不重新访问它们全部,就无法计算归一化。如果去掉 softmax(或换成一个简单的特征映射),输出就变成

```
oₜ = Σ_{i ≤ t} vᵢ (kᵢ · qₜ)  =  ( Σ_{i ≤ t} vᵢ kᵢᵀ ) · qₜ  =  Sₜ · qₜ
其中   Sₜ = S_{t-1} + vₜ kₜᵀ
```

逐项解读:这个技巧只是对矩阵乘法重新结合。不再拿新的 query 去和每个已存储的 key 比较,而是维护一个持续累加的**状态矩阵** Sₜ(每个头的大小为 d_v × d_k——比如 128 × 128),它是迄今见过的每一对 key/value 的外积 vᵢkᵢᵀ 之和。读取时只需用 query 做一次矩阵-向量乘法;每来一个新词元,就往 S 上加一个外积。关键后果是:**无论你已经读了 100 个词元还是 1,000,000 个,状态的大小都一样**——每词元的计算和内存都是常数,这正是 SSM 所追求的 RNN 式性质。可以把 S 看作一个把 key 映射到 value 的微型联想记忆("快速权重",fast weights)。

**朴素线性注意力的问题。** 这种更新只会*累加*,从不擦除任何东西。于是随着序列变长,大量不同的 key/value 对被叠加进同一个固定大小的矩阵,相互干扰——上下文越长,检索越模糊。正因如此,朴素线性注意力在语言建模上历来明显不如 softmax 注意力。

### 第 2 步——两种修补:门控衰减与 delta 规则

**修补 A:门控衰减(遗忘)。** 在加入新的一对之前,先把旧状态乘以一个依赖数据、取值在 0 到 1 之间的因子 αₜ:`Sₜ = αₜ S_{t-1} + vₜkₜᵀ`。αₜ 接近 1 时记忆得以保留;αₜ 很小时记忆基本被清空(例如在话题切换处)。这就是 Mamba 的"选择性"思想在线性注意力形式下的表达,本质上也正是 Mamba-2 和 Gated Linear Attention 所做的事。

**修补 B:delta 规则(定向覆盖)。** 写入之前,先*读出*记忆对这个新 key 当前返回什么,然后只写入修正量:

```
Sₜ = S_{t-1} − βₜ (S_{t-1} kₜ − vₜ) kₜᵀ
```

逐项解读:S_{t-1}kₜ 是"记忆目前认为 key kₜ 对应的 value"。减去期望的 value vₜ 就得到误差;更新会推动记忆去减小这个误差,其中 βₜ ∈ (0, 1)(由输入计算得到)相当于每个词元的写入强度。如果你见过这种形式,它就是经典的 Widrow-Hoff "delta 规则"——字面意义上就是对平方误差 ‖S kₜ − vₜ‖² 做一步梯度下降,只不过是在*前向传播内部*完成的。因此,对于同一个 key,delta 规则不是把新记忆叠在旧记忆之上,而是*替换*它——这正是它在精确的键-值召回上远胜于累加式更新的原因。

### 第 3 步——Gated DeltaNet = 两者结合

```
Sₜ = αₜ · S_{t-1} · (I − βₜ kₜ kₜᵀ)  +  βₜ vₜ kₜᵀ        (key 归一化为单位长度)
oₜ = Sₜ · qₜ
```

逐项解读:这就是把同一个 delta 规则更新改写成矩阵形式,再在其上施加门控衰减 αₜ。因子 (I − βₜkₜkₜᵀ) 会擦除(部分)沿 kₜ 方向存储的内容;第二项在那里写入新的 value;αₜ 则让该层在内容需要时额外把整个记忆淡化掉。两种机制互为补充:衰减擅长*整体性*遗忘,delta 规则擅长*精确*替换某一条关联。与 Mamba 一样,训练时并不是逐词元地跑这个循环——分块(chunk-wise)并行算法用稠密矩阵乘法一次处理一整块词元,让 GPU 保持高效;推理时它则作为真正的循环运行,状态大小固定。

**为什么重要。** 据报道,在同等规模下,Gated DeltaNet 在语言建模上、尤其是在召回密集型任务上(这恰恰是固定状态模型一向最弱的领域)优于 Mamba-2 和朴素 DeltaNet。改进版本很快出现;例如 Moonshot 的 **Kimi Delta Attention**(2025)就是一种改进的 Gated DeltaNet。

**优势与局限。**
- 优势:推理时每词元成本和内存恒定(KV 缓存不会增长);可分块并行训练;在固定状态层中属于召回能力报告结果最强的一类。
- 局限:状态仍是对过去内容的*固定大小、有损*压缩——如果某个细节已被覆盖,模型就无法可靠地逐字复制 500,000 个词元之前的任意段落。从超长上下文中精确检索仍是注意力的优势,这也正是下文的生产级设计都采用混合架构、而非纯 GDN 堆叠的原因。其算子(kernel)与工具链的成熟度也远落后于注意力。

## 混合架构浪潮(2024-2026)

**名称与定义。** **混合架构(hybrid architecture)** 在同一个模型中堆叠两种序列混合层:大部分是廉价的固定状态层(如 Mamba-2 这样的 SSM,或 Gated DeltaNet 这样的线性注意力层),再按固定间隔插入少量普通的全注意力层。(前馈/MoE 子层保持不变——见 [mixture-of-experts.md](mixture-of-experts.md)。)

**直觉。** 固定状态层以低成本承担每个词元的大部分工作,并携带上下文的压缩式运行摘要;偶尔出现的全注意力层保留真正的 KV 缓存,可以精确查找任何更早的词元,弥补固定状态层在精确检索上的弱点。由于只有注意力层存储 KV 缓存,KV 内存大致按注意力层所占比例缩小——若每 4 层中有 1 层是注意力,则大约是同等深度、同等注意力配置的全注意力模型所需的四分之一(这是简单算术给出的近似值;实际节省取决于设计的其余部分)。

**3:1 层排布示意。** 据报道 Qwen3.5-397B-A17B 采用的模式:60 层 = 15 次重复的 [3 个 Gated DeltaNet 层 + 1 个门控全注意力层],每层之后都接一个 MoE 前馈子层。

```
 layer:    1     2     3     4     5     6     7     8    ...   57    58    59    60
 mixer:   GDN   GDN   GDN  ATTN   GDN   GDN   GDN  ATTN   ...  GDN   GDN   GDN  ATTN
 FFN:     MoE   MoE   MoE   MoE   MoE   MoE   MoE   MoE   ...  MoE   MoE   MoE   MoE
         └───────── block 1 ────────┘└───────── block 2 ────────┘   └──────── block 15 ───────┘

 每层推理内存:
   GDN  → 每个头固定为 d_v × d_k 的状态(不随上下文长度增长)
   ATTN → 随上下文长度线性增长的 KV 缓存(60 层中的 15 层)
```

**2024-2026 时间线。**
- **Jamba**(AI21 Labs,2024)——较早公开发布的混合模型,将 Mamba 层与少量注意力层交错排列(并配合 MoE)。
- **Nemotron-H / Nemotron Nano 2**(NVIDIA,2025)——生产级的 Mamba-2 + 注意力混合模型系列。
- **Qwen3-Next**(阿里巴巴 Qwen,2025 年 9 月)——以大约 3:1 的比例组合 Gated DeltaNet 与*门控*全注意力(见 [transformer-architecture.md](transformer-architecture.md) 中关于门控注意力的讨论),并配合超稀疏 MoE。
- **Kimi Linear**(Moonshot AI,2025 年 10 月)——Kimi Delta Attention 与多头潜在注意力(Multi-head Latent Attention,MLA——一种压缩 KV 的注意力变体;见 [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md))层以大约 3:1 混合。
- **NVFP4 预训练**(NVIDIA,"Pretraining LLMs with NVFP4",arXiv 2509.25149,2025;2026 年 3 月修订)——用于演示 4 位预训练的模型本身就是一个 **12B 混合 Mamba-Transformer**,在 10T 词元上训练,MMLU-Pro 达到 62.58%,而 FP8 基线为 62.62%(据报道)。之所以值得一提,是因为它表明混合模型如今已成为验证其他前沿技术的*默认试验载体*,而不是一个冷门的旁支;该精度格式本身见 [quantization.md](../06-inference-optimization/quantization.md)。
- **Qwen3.5**(2026 年初)——把 3:1 的 Gated DeltaNet : 门控注意力排布带入了 Qwen 的*主力*模型线,例如 Qwen3.5-397B-A17B(据报道),这是线性注意力混合架构作为旗舰模型、而非实验性旁支发布的最清晰案例之一。

**通往同一目标的另一条路线。** 混合架构通过*替换掉大部分注意力层*来应对注意力的长上下文成本。2025-2026 年的另一条路线则保留所有注意力层,但让每个注意力层更便宜——压缩 KV 缓存(MLA),或只关注过去词元中被选出的一个子集(DeepSeek Sparse Attention,以及据报道 DeepSeek-V4 的压缩稀疏注意力)。DeepSeek-V4 也把自己的设计称为"混合注意力",但它交错的是两种*压缩/稀疏注意力*变体,而不是线性注意力层。这些内容见 [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md)。哪条路线会胜出——或者前沿模型会不会两者兼用——目前尚无定论。

## 与 Transformer 相比的采用现状——诚实评估

截至 2026 年末,局面已从"有前景的研究方向"实质性地转变为"可用于生产的选项",但具体形态如下:

- **混合架构已进入生产级旗舰模型。** 线性注意力/SSM 混合架构已出现在多家主要实验室公开发布的前沿级模型中(Qwen3.5 主力线、Kimi Linear、NVIDIA 的 Nemotron-H 系列),并被用作其他前沿实验的基座(NVFP4 预训练)。大约 3:1 的线性/全注意力比例在彼此独立的团队中反复出现,这是一个有意义的信号——不过它是一种被广泛沿用的经验选择,而非推导出的最优解。
- **纯 SSM / 纯线性注意力模型在前沿仍然罕见。** 每一个知名的生产级设计都保留了*一些*全注意力层。固定状态压缩的局限(在超长上下文中精确召回不够准确)并没有被工程手段消除,业界用实际行动表明的偏好是用注意力来弥补它,而不是彻底放弃注意力。
- **全注意力 Transformer 仍占多数。** 许多前沿模型——就公开信息而言,也包括大多数闭源前沿模型(其架构通常不公开)——仍是全注意力的仅解码器 Transformer,其降本手段是 GQA、MLA、滑动窗口或稀疏注意力,而不是循环层。注意力的工具链、算子和训练配方仍然成熟得多。
- **优势最明显的场景。** 与以往一样,效率优势在超长上下文和高吞吐解码中最为重要,因为此时 KV 缓存大小主导着服务成本;在短到中等长度的上下文中,收益要小得多。

混合架构会成为主导性的前沿模板,还是更廉价的注意力设计(稀疏/压缩注意力)会让它们变得没有必要,这确实是一个开放问题——请参阅 [09-roadmaps/near-term-outlook.md](../09-roadmaps/near-term-outlook.md) 中明确标注的推测,而不要把这里当作任何确定的论断。

## 对比表

| 架构 | 年份 | 关键创新 | 是否仍然重要(2026)? |
|---|---|---|---|
| S4 | 2021/2022 | 结构化线性循环,可等价为卷积进行训练 | 影响深远;新工作中已基本被 Mamba 取代 |
| Mamba | 2023 | 依赖输入的("选择性")状态空间参数 | 是——研究中活跃,并经由 Mamba-2 用于生产级混合模型 |
| RWKV | 2023 | 类线性注意力的循环,社区驱动 | 小众——规模较小但活跃的开源谱系 |
| Mamba-2 | 2024 | 状态空间对偶性:选择性 SSM = 带衰减的线性注意力 | 是——生产级混合模型中的 SSM 层(如 Nemotron-H) |
| Gated DeltaNet | 2024/2025 | 在固定大小的矩阵状态上结合 delta 规则覆盖与门控衰减 | 是——Qwen3-Next/Qwen3.5 中的线性注意力层;改进版为 Kimi Delta Attention |
| 混合架构(线性/SSM + 全注意力) | 2024-2026 | 以固定状态层为主,约每 4 层 1 层全注意力(常见比例) | 是——截至 2026 年末已用于生产级旗舰开放权重模型 |

## 与其他算法的关系

- SSM 与线性注意力是出于效率考虑、针对 [transformer-architecture.md](transformer-architecture.md) 中自注意力机制提出的替代方案,直接针对该机制的 O(n²) 扩展问题;现代混合模型中的全注意力层常常使用该文件中介绍的*门控注意力*变体。
- "线性循环 + 并行训练"的技巧呼应并大幅改进了普通 RNN/LSTM 的串行训练限制(见 [rnn-lstm-gru.md](rnn-lstm-gru.md));delta 规则与梯度下降的一步是同一种误差修正更新(见 [optimization-algorithms.md](../01-foundations/optimization-algorithms.md))。
- 混合设计与 [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md) 中讨论的权衡直接相关,因为其主要卖点之一就是缩小 KV 缓存;该文件也涵盖了与之竞争的"更廉价注意力"路线(MLA、稀疏注意力)。
- 生产级混合模型把固定状态层与非常稀疏的混合专家(MoE)前馈层搭配使用——见 [mixture-of-experts.md](mixture-of-experts.md)。
- 在混合模型上演示的 4 位(NVFP4)预训练见 [quantization.md](../06-inference-optimization/quantization.md)。

## 来源

- Gu, Goel, Ré, "Efficiently Modeling Long Sequences with Structured State Spaces"(2021 年以预印本发布;2022 年会议版本)[S4]
- Gu, Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"(2023)
- Dao, Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"(2024)[Mamba-2]
- Peng 等, "RWKV: Reinventing RNNs for the Transformer Era"(2023)
- Katharopoulos 等, "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"(2020)
- Schlag, Irie, Schmidhuber, "Linear Transformers Are Secretly Fast Weight Programmers"(2021)[线性注意力中的 delta 规则]
- Yang 等, "Gated Linear Attention Transformers with Hardware-Efficient Training"(2023)
- Yang 等, "Parallelizing Linear Transformers with the Delta Rule over Sequence Length"(2024)[DeltaNet]
- Yang, Kautz, Hatamizadeh, "Gated Delta Networks: Improving Mamba2 with Delta Rule"(2024;ICLR 2025)[Gated DeltaNet]
- Lieber 等(AI21 Labs), "Jamba: A Hybrid Transformer-Mamba Language Model"(2024)
- NVIDIA, Nemotron-H 与 Nemotron Nano 2 技术报告(2025)[Mamba-2 + 注意力混合]
- Qwen 团队, Qwen3-Next 发布(2025 年 9 月)[Gated DeltaNet + 门控注意力,约 3:1;超稀疏 MoE]
- Qwen 团队, Qwen3.5 发布(2026 年初)[主力线采用 3:1 Gated DeltaNet : 门控注意力;Qwen3.5-397B-A17B 层排布据报道]
- Moonshot AI, "Kimi Linear" 技术报告(2025 年 10 月)[Kimi Delta Attention + MLA 混合,约 3:1]
- NVIDIA, "Pretraining LLMs with NVFP4"(arXiv 2509.25149,2025;2026 年 3 月 v2)[12B 混合 Mamba-Transformer,10T 词元;MMLU-Pro 数据据报道]
