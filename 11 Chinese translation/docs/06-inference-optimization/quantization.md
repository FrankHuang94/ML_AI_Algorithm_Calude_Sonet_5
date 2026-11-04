# 量化

量化(quantization)降低用来表示模型权重(有时也包括激活值)的数值精度,从而缩小内存占用,并常常加快计算——代价是损失一定的精度。本文介绍:量化为什么对推理尤其重要、低位宽整数表示的机制、应用量化的两大类方法、两种最常用的实用整数量化方法(GPTQ、AWQ),以及在 2024–2026 年间从研究走进官方模型发布、甚至走进预训练的低精度*浮点*格式(FP8,以及 4 位微缩放格式 MXFP4 和 NVFP4)。

## 为什么量化对推理成本很重要

**基准情况。** 训练期间,神经网络权重通常以 16 位或 32 位浮点格式存储和计算(更高的精度有助于基于梯度的优化保持数值稳定——见 [optimization-algorithms.md](../01-foundations/optimization-algorithms.md))。但训练好的模型权重并不需要永远保持这种格式——推理(用已训练好的模型做预测)对数值的要求与训练不同,往往更宽松。

**为什么这在 LLM 规模下如此重要。** 模型的内存占用主要由参数量乘以每个参数所占字节数决定。一个以 16 位精度存储的 700 亿参数模型,光权重就需要约 140 GB——这还没算 KV cache(见 [kv-cache-and-attention-optimization.md](kv-cache-and-attention-optimization.md))或其他运行时内存。把每个参数的存储量减半或减到四分之一(通过下文的 8 位或 4 位量化),就会直接、成比例地降低内存需求,这对服务成本影响巨大:它决定了托管并服务一个模型需要多少块(以及多贵的)加速器、在固定内存预算内能把多少请求合成一批(见 [serving-and-batching.md](serving-and-batching.md));而且由于许多 LLM 推理负载受限于显存带宽而非纯计算(瓶颈往往是把权重从显存搬到计算单元,而不是算术本身),更小的权重表示可以直接转化为更快的推理,而不只是更便宜的存储。

## INT8/INT4 的机制

**核心思想。** 不再把每个权重表示为 16 位或 32 位浮点数,而是表示为一个低位宽整数(常见为 8 位或 4 位),再配上少量额外的共享信息(一个**缩放因子(scale factor)**,有时还有一个**零点(zero-point)**偏移),用来在紧凑的整数表示和近似的实数值之间来回转换。

**核心机制。** 对一组权重(例如矩阵某一行的全部权重,或其他选定的分组),找出其中数值的范围,并定义一个缩放因子 s,使得:

```
quantized_value = round(real_value / s)
real_value_approx = quantized_value · s
```

可以把它想象成:把每个连续取值的权重"吸附"到一架只有寥寥几级横档的梯子上最近的那一级。INT4 只给你 16 级横档来表示整个取值范围:

```
   Original weights (continuous):   -0.9   -0.3    0.0   0.25   0.7   0.95
                                       │      │      │     │      │     │
                                       ▼      ▼      ▼     ▼      ▼     ▼
   INT4 grid (16 evenly-spaced rungs across the range, here −1 … +1):
     -1.0  -0.87 ... -0.33 -0.20  0.0  0.20  0.33 ...  0.73  0.87  1.0
       │           │      │            │     │              │
   snapped:      -0.87  -0.33   0.0   0.20  0.73  0.87   ← each weight rounds to its nearest rung
   error:        (−0.03)(+0.03) (0)  (−0.05)(+0.03)(−0.08)  ← small rounding error per weight
```

逐项解读:选择 s,使实数权重的完整范围映射到可用的整数范围上(例如有符号 8 位整数为 −128 到 127,4 位为 −8 到 7)。除以 s 再取整,就把每个权重压成一个小整数;再乘回 s,就恢复出原值的近似——并不精确,因为取整到最近的可用整数会损失一些精度,但只要选择得当就足够接近,模型的整体行为基本得以保留。它之所以行得通:大模型有数千亿个权重,单个取整误差很小且大致相互独立,因此它们倾向于相互抵消而不是累积——整个模型对"每个权重都有一点噪声"的鲁棒性远超直觉。INT8(256 级)几乎总是接近无损;INT4(16 级)开始需要认真挑选方法;再往下,误差就不再能干净地抵消,精度下降会更快。位宽越小,可用来表示同一范围的离散取值就越少,近似就越粗糙(越不精确)——这就是为什么从 8 位降到 4 位能省更多内存,却也冒着更大的精度损失风险;也是为什么分组方式(多少个权重共享一个缩放因子)和校准流程(如何确定范围/缩放因子)的具体选择,对实际损失多少精度影响极大。

## 训练后量化(PTQ)与量化感知训练(QAT)

**训练后量化(Post-training quantization,PTQ)。** 在模型完全训练好之后再对其权重做量化,通常会用一个小的校准数据集(把一些有代表性的输入跑一遍模型,观察真实的激活/权重范围,以便选出好的缩放因子),但不再做任何基于梯度的训练。这种方法应用起来又快又便宜(不需要重新训练),是让现有预训练模型以更低成本推理的标准选择。

**量化感知训练(Quantization-aware training,QAT)。** 在训练*过程中*就模拟量化的效果(通常在前向传播中把权重/激活取整到量化后的表示,同时仍按全精度值计算梯度,让模型调整自身权重,以补偿它在推理时将遇到的量化噪声)。在非常低的位宽下,它通常比 PTQ 精度更好,因为模型有机会在训练中真正适应精度损失,而不是事后被强加;但它的代价要高得多——需要一次完整(或相当大部分)的训练,而不只是一次快速校准,而且如果拿不到模型原始的训练数据/流程,就无法实际应用。

**当前状态。** 对于低成本部署现有大型预训练模型,PTQ 是远更常用的方法,正是因为它不需要重新训练;QAT 的使用更有选择性,用于那些值得为在极激进的量化水平下榨取最高精度而付出额外训练成本的场合。

## GPTQ

**起源。** Frantar、Ashkboos、Hoefler、Alistarh,"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"(2022)。

**核心贡献。** GPTQ 是一种 PTQ 方法,它逐层量化模型权重,并且——关键在于——按照一个精心选择的特定顺序量化每层内的权重,利用二阶信息(与 Hessian 矩阵相关的一种近似——见 [optimization-algorithms.md](../01-foundations/optimization-algorithms.md)),通过调整该层中*尚未量化*的剩余权重,来补偿量化每个权重所引入的误差,使误差不会在整层中各自独立地累积而得不到纠正。

**为什么重要。** GPTQ 证明了大语言模型可以被量化到非常低的位宽(常见为 4 位,有时更低),精度下降相对很小,而且无需任何重新训练,只需一个相对快速的一次性校准流程——这让激进的 PTQ 在 QAT 成本高得令人望而却步的规模上,对大型预训练模型变得切实可行。

## AWQ(激活感知权重量化)

**起源。** Lin 等,"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"(2023)。

**核心贡献。** AWQ 的关键洞见是:并非所有权重都同等需要被精确保留——具体来说,与幅值异常大的激活相互作用的权重,对模型输出的影响格外大;保护*这些*特定权重的精度(通过一次校准找出它们,并对其采用相对不那么激进的量化,例如在量化前重新缩放),比一视同仁地处理所有权重能保留多得多的原始模型精度。

**为什么重要。** AWQ 的精度可与 GPTQ 相媲美(在一些报告的对比中甚至更好),同时更简单、运行更快(不需要 GPTQ 那种更繁琐的、逐层且依赖顺序的误差补偿过程),因此成为量化大模型以供部署的热门实用选择。

## 低精度浮点格式:FP8、MXFP4、NVFP4

上文的一切都使用*整数*加缩放因子。另一类方法则保留极小的*浮点*数,自 2024 年以来,它已成为前沿模型以低精度训练和发布的主要方式。

**为什么在低位宽下使用浮点。** 一个浮点数把一部分位用于**指数(exponent)**(数值接近哪个 2 的幂),一部分用于**尾数(mantissa)**(在相邻两个 2 的幂之间的精细位置),再加一个符号位。格式按这种划分命名:**E4M3** 表示 4 位指数、3 位尾数;**E5M2** 用一位尾数换一位指数(范围更大、精度更低);**E2M1** 则是一种 4 位浮点。与刻度均匀分布的整数网格不同,浮点网格在零附近密、远离零处疏——这更契合权重、激活和梯度那种钟形、易出现离群值的分布:大多数值很小,能得到精细的分辨率,而少数很大的值仍然可以被表示。

```
 Magnitudes representable by a 4-bit format (positive half; sign bit gives the negatives):

 INT4 (evenly spaced):    0   1   2   3   4   5   6   7
 FP4 E2M1 (float):        0  0.5  1  1.5  2   3   4   6      ← fine steps near 0, coarse steps far out

 Either grid is then multiplied by a scale factor, exactly as in the INT8/INT4 section above.
```

### FP8 训练(DeepSeek-V3)

**它是什么。** 把*训练*(而不仅是推理)中的大规模矩阵乘法放在 8 位浮点(E4M3 和/或 E5M2)中运行,同时把权重的主副本、优化器状态(见 [optimization-algorithms.md](../01-foundations/optimization-algorithms.md))以及数值敏感的运算保留在更高精度中——即把"混合精度"训练在 16 位 BF16 标准的基础上再往前推一步。

**起源。** FP8 格式由 Micikevicius 等在 "FP8 Formats for Deep Learning"(2022)中提出,NVIDIA 的 Hopper 代 GPU 加入了 FP8 矩阵运算硬件。DeepSeek-V3(2024 年 12 月;总参数 671B / 激活 37B,混合专家模型)是 FP8 混合精度训练能在前沿规模下奏效的标志性公开证明。

**核心机制(它为何能行)。** 主要风险在于 8 位留下的余地太小,一个离群值就可能迫使与它共享缩放因子的所有值都采用粗糙的刻度。DeepSeek-V3 的做法是**细粒度缩放**——为每个矩阵的小分块/小瓦片分别设置缩放因子,而不是整个张量共用一个,这样离群值只会影响它所在的小块——并以更高精度累加矩阵乘法的部分和,使取整误差不会在长点积中层层堆积。

**为什么它对推理很重要。** 原生以 FP8 训练的模型可以直接以 FP8 提供服务,无需单独的训练后量化步骤,这模糊了上文 PTQ/QAT 的界线:模型实际上已在训练中适应了低精度。

### 微缩放格式:MXFP4 与 NVFP4

**思想。** 在 4 位下分辨率实在太少,每个张量(甚至每行)只用一个缩放因子远远太粗糙。**微缩放(microscaling)**让每一小块连续数值拥有各自共享的缩放因子——与上文 GPTQ/AWQ 采用的分组缩放思想相同,只不过被标准化成了硬件能直接理解的数值格式。开放计算项目(Open Compute Project)的微缩放(MX)规范(2023)得到多家芯片和云计算公司支持,定义了 MX 格式家族;NVIDIA 则定义了自己的变体 NVFP4,由其 Blackwell 代 GPU 原生支持。

```
 MXFP4 — block of 32 values, one power-of-two scale
   scale: E8M0  (8-bit exponent only → scale ∈ {…, 0.25, 0.5, 1, 2, 4, …})
   [v1][v2][v3] ... [v32]     each v = FP4 E2M1
   storage ≈ 4 + 8/32 = 4.25 bits per value

 NVFP4 — block of 16 values, one FP8 scale, plus one FP32 scale for the whole tensor
   scale: FP8 E4M3  (can be 1.375, 2.75, … — not just powers of two)
   [v1][v2] ... [v16]          each v = FP4 E2M1
   storage ≈ 4 + 8/16 = 4.5 bits per value  (+ one FP32 number per tensor, negligible)

 decoded value = v × block_scale            (MXFP4)
 decoded value = v × block_scale × tensor_scale   (NVFP4)
```

逐项解读:两种格式都把每个值存成同样的 4 位浮点数,区别只在于共享缩放因子如何选取。MXFP4 的缩放因子必须是 2 的幂,所以块内最大值通常无法恰好对准网格的顶端——要么够不着(让本就极小的 4 位网格有一部分闲置),要么超出而被截断。NVFP4 的缩放因子本身就是一个小浮点数,因此能把网格拉伸得更贴合每个块;由于 E4M3 缩放因子自身的范围有限,再用一个每张量的 FP32 缩放因子来处理整体量级。NVFP4 的块也更小(16 对 32),所以一个离群值波及的邻居更少。代价是存储略多(每个值 4.5 位对 4.25 位),且格式绑定于单一厂商的硬件。

### gpt-oss:以 MXFP4 发布的官方模型

OpenAI 的开放权重模型 gpt-oss(2025 年 8 月;gpt-oss-120b 和 gpt-oss-20b,均为混合专家模型)发布时,其 MoE 权重原生就是 MXFP4 格式。这有两点值得注意。第一,4 位权重*就是*官方检查点——而不是发布后由社区制作的量化版本。第二,它精准地瞄准了内存的大头:在 MoE 模型中,专家权重占参数的绝大多数(见 [mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md)),因此量化它们就拿到了大部分节省,而其余较小的部分可以保持较高精度。粗略估算:约 1200 亿参数 × 约 4.25 位 ≈ 64 GB,这就是为什么较大的那个模型可以用单块 80 GB 级加速器、而不是多块来提供服务。

### 用 4 位做预训练:NVFP4(NVIDIA,2025–2026)

**展示了什么。** NVIDIA 的 "Pretraining Large Language Models with NVFP4"(arXiv 2509.25149,2025 年 9 月;2026 年 3 月修订)用 NVFP4 矩阵乘法,在 10 万亿个 token 上训练了一个 120 亿参数的 Mamba-Transformer 混合模型,在 MMLU-Pro 基准上达到 62.58%,而 FP8 训练的基线为 62.62%——基本持平。

**让它奏效的技术**,每一项都针对 4 位训练出错的一种具体方式:
- **随机 Hadamard 变换。** 量化之前,先把数值块乘以一个由 ±1 构成的随机化正交矩阵(之后再撤销——由于该变换是正交的,它可以在矩阵乘法内部被抵消)。这会把单个大离群值摊到许多值上,使任何一个块的缩放因子都不会被某个尖峰主导。
- **二维块缩放。** 权重按正方形瓦片而非一维条带来缩放,这样无论矩阵是按行读取(前向传播)还是按列读取(反向传播中的转置使用),同一个权重得到的量化值都相同;否则前向和反向传播看到的权重会略有不同。
- **对梯度使用随机舍入(stochastic rounding)。** 不总是舍入到最近的网格点,而是以与距离远近成比例的概率随机向上或向下舍入。每次舍入噪声更大,但平均而言是*无偏*的,因此系统性的舍入误差不会在数百万次更新中累积。
- **少数敏感层保留更高精度**,以及**在训练后期切换到更高精度**——此时更新量变小,精度最为关键。

**状态。** 这是一个 120 亿参数规模的研究成果。它表明只要足够小心,4 位训练是*可行*的;截至 2026 年底,它能否成为前沿规模预训练的常规做法,尚无公开定论。

## 精度/延迟/内存权衡表

| 精度 | 相对内存(对比 16 位) | 典型精度影响 | 备注 |
|---|---|---|---|
| 16 位(FP16/BF16) | 100%(基线) | 无(标准训练/推理精度) | 大多数训练的默认;常见的推理基线 |
| 8 位整数(INT8) | ~50% | 小,校准良好时常常几乎可忽略 | 广泛使用,工具链成熟 |
| 8 位浮点(FP8,E4M3/E5M2) | ~50% | 小;既用于推理也用于训练 | 在新一代加速器上已成主流;DeepSeek-V3 展示了前沿规模的 FP8 训练 |
| 4 位整数(INT4,通过 GPTQ/AWQ) | ~25%(另加少量分组缩放开销) | 小到中等,取决于方法和模型 | 截至 2026 年底,社区/成本敏感型开放模型部署的标准做法 |
| MXFP4(FP4,32 值一块,2 的幂缩放) | ~27%(4.25 位) | 小到中等;模型专门为此训练/后训练时效果最好 | 开放标准;gpt-oss 的 MoE 权重即以此格式发布 |
| NVFP4(FP4,16 值一块,FP8 缩放 + 张量缩放) | ~28%(4.5 位) | 设计目标是在相同位宽下误差低于 MXFP4 | NVIDIA 专有;已在 120 亿参数规模的预训练中得到验证(2025) |
| 低于 4 位(实验性) | <25% | 更大、更不稳定、对方法质量更敏感 | 活跃的研究领域;生产可用性不够稳定 |

## 现状(2026 年底)

- **16 位(BF16)不再是前沿训练的精度下限。** 由 DeepSeek-V3 在大规模上验证的 FP8 混合精度,据广泛报道已成为大型训练的常见选择,训练出的模型可以直接以 FP8 提供服务。
- **4 位浮点已进入官方发布。** gpt-oss 以 MXFP4 权重发布;原生支持 FP4 的硬件让 4 位推理不仅更小,而且更快。
- **4 位训练是研究前沿**,NVIDIA 的 NVFP4 成果是目前主要的公开数据点。
- **整数 PTQ(GPTQ、AWQ)仍是日常工具**,用于在开放模型发布后对其进行压缩,尤其是在不原生支持 FP4/FP8 的硬件上。

## 与其他算法的关系

- 量化常与 LoRA 结合(即 QLoRA)用于节省内存的微调——见 [finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md)。
- 量化直接减少了与 KV cache 管理和服务/批处理决策相关的内存占用——见 [kv-cache-and-attention-optimization.md](kv-cache-and-attention-optimization.md) 和 [serving-and-batching.md](serving-and-batching.md)。KV cache 本身也可以用更低精度存储。
- 由于 LLM 解码受限于显存带宽,低精度权重对 decode 阶段的加速尤其明显——见 [serving-and-batching.md](serving-and-batching.md) 中关于 prefill/decode 的讨论。
- FP8 训练与 [distributed-training.md](../05-training-methodology/distributed-training.md) 中的分布式训练技术并列,都是降低大型训练内存与计算成本的手段。
- 量化与剪枝、蒸馏同属几种互补的压缩技术之一——见 [pruning-and-distillation.md](pruning-and-distillation.md)——在生产部署中通常叠加使用。

## 来源

- Frantar, Ashkboos, Hoefler, Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022)
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023)
- Micikevicius et al., "FP8 Formats for Deep Learning" (2022)
- Open Compute Project, "OCP Microscaling Formats (MX) Specification" v1.0 (2023);Rouhani et al., "Microscaling Data Formats for Deep Learning" (2023)
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024) [大规模 FP8 混合精度训练]
- OpenAI, gpt-oss-120b / gpt-oss-20b 模型卡 (2025 年 8 月) [MoE 权重采用 MXFP4]
- NVIDIA, "Pretraining Large Language Models with NVFP4," arXiv 2509.25149 (2025 年 9 月;2026 年 3 月 v2) [12B 模型、10T token;MMLU-Pro 数字为报告值]
