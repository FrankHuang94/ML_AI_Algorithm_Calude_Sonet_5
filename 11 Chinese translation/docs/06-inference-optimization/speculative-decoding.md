# 投机解码

投机解码(speculative decoding)通过重新组织 token 的*产生方式*来加速自回归生成(见 [autoregressive-generation.md](../04-generative-models/autoregressive-generation.md))——完全不改变模型,也完全不改变其输出的统计分布。本文介绍"先起草、再验证"的框架,解释它为何能精确保持输出分布,并介绍几大类起草器(drafter):独立的草稿模型、挂在目标模型上的额外预测头(Medusa、EAGLE 系列,以及在预训练中就训练进模型的多 token 预测模块),以及无需草稿的前瞻解码(lookahead decoding)。最后介绍扩散语言模型——它追求同一个目标(每次昂贵的计算产出不止一个 token),但走的是另一条*并不*精确的路线。

## "草稿模型 + 验证"的框架

**核心问题。** 标准自回归生成每产生一个 token 就要调用一次(通常非常庞大、昂贵的)模型,而且严格串行——第 51 个 token 必须等第 50 个完成后才能生成,每次调用都要对一个可能极其庞大的模型的每一层做一次完整前向传播;然而从信息论的角度看,生成任意一个额外的 token 往往是个"便宜"的决定(在给定上下文时,许多 token 高度可预测——想想在 "I live in New" 之后补全 "York")。

**起源。** Leviathan、Kalman、Matias(Google),"Fast Inference from Transformers via Speculative Decoding"(2023);Chen 等(DeepMind),"Accelerating Large Language Model Decoding with Speculative Sampling"(2023)——两篇相互独立、密切相关的论文,大约同时提出了相同的核心思想。

**核心机制。** 用一个小而快的**草稿模型**(draft model,一个小得多/便宜得多的模型,最好在行为上与大型目标模型相似)快速连续生成一小段候选 token——这之所以快,正是因为草稿模型小,每次前向传播都很便宜。然后让大而昂贵的**目标模型**(target model)只运行一次,在单次前向传播中同时检查草稿模型给出的所有候选 token(因为 Transformer 可以在一次并行计算中为一整段已选定的 token 打分,而不必每个 token 串行跑一遍——这正是让 Transformer *训练*高效的那种并行性,只不过这里用在验证而非生成上)。对每个候选 token,把大模型自己会赋予该 token 的概率与草稿模型的提议进行比较;大模型足够认同的 token 被接受,在第一个出现明显分歧的位置拒绝(并立即纠正),丢弃草稿模型在该位置之后生成的所有候选 token。

```
   Prompt: "The capital of France is"

   1. DRAFT (small, fast model guesses 4 tokens ahead, cheaply):
         "Paris" → "," → "a" → "city"

   2. VERIFY (big model checks all 4 at once, in ONE parallel pass):
         "Paris"  ✓ agrees      → ACCEPT
         ","      ✓ agrees      → ACCEPT
         "a"      ✓ agrees      → ACCEPT
         "city"   ✗ disagrees   → REJECT here, big model supplies "the" instead

   3. RESULT of this one big-model step: "Paris , a the"  (3 draft tokens accepted + 1 correction)
      → 4 tokens produced for the price of ONE big-model pass, instead of four.
```

逐项解读:你不必为每一个 token 都付出大模型的全部成本,而是让大多数 token 只付出(便宜得多的)草稿模型成本,大模型则每批候选只运行一次并行验证——只要草稿模型的猜测经常正确(对于匹配得不错的草稿模型和典型的、相当可预测的文本,这往往成立),就能用"一次昂贵的并行大模型调用 + 若干次便宜的草稿模型调用"换掉"多次昂贵的串行大模型调用",净加速可观。最坏情况也是安全的:如果草稿模型毫无用处、每个猜测都被拒绝,你会退回到每次大模型前向传播恰好产出一个 token——与不做投机时速度相同,在 token 质量上永远不会变差(只是在被拒绝的草稿上多浪费了一点计算)。

## 为什么它能加速生成却不改变输出分布

**关键的数学性质。** 投机解码使用一种特定的接受/拒绝规则(拒绝采样的一种形式,这是一种经典的统计采样技术),使*最终*实际产出的 token 序列与"只用大型目标模型按普通方式逐个生成每个 token"时具有完全相同的概率分布。草稿模型只负责*提议*候选,从来无权单方面决定实际输出——最终决定每个被接受 token 的是大模型的判断,任何大模型足够不认同的 token 都会在验证中被发现并纠正。

**为什么这很重要——一顿本领域少有的"免费午餐"。** 本节其他地方介绍的大多数推理优化技术,都要用一定的精度换速度(量化——见 [quantization.md](quantization.md)——以及剪枝/蒸馏——见 [pruning-and-distillation.md](pruning-and-distillation.md)——都明确接受一定的质量下降以换取效率)。投机解码在本知识库中与众不同,恰恰因为它是真正的加速,而且**输出的统计性质丝毫不变**——这与 FlashAttention 为注意力计算本身提供的保证相同(见 [kv-cache-and-attention-optimization.md](kv-cache-and-attention-optimization.md)),只不过这里作用于整个生成过程。

## Medusa

**起源。** Cai 等,"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"(2024)。

**核心贡献。** Medusa 不使用完全独立的草稿模型,而是直接在目标模型上额外加几个预测"头",每个头被训练去预测序列中更靠后的某个 token(一个头预测下一个 token,另一个预测再下一个,依此类推),从而由单个目标模型一次前向传播产生多个候选续写,再用与标准投机解码类似的基于树的接受机制进行验证。

**为什么重要。** Medusa 把"起草"能力直接内置到目标模型中,避免了训练、维护一个独立草稿模型并使其长期与目标模型保持匹配的运维复杂性(当目标模型被更新或微调时,独立的草稿模型可能逐渐与之失配)。

## EAGLE 系列(EAGLE、EAGLE-2、EAGLE-3)

**名称与定义。** EAGLE 是一种投机解码方法,其起草器是挂在目标模型上的一个小型额外网络,它在目标模型内部**特征**(即隐藏状态向量——模型对每个位置的内部表示)的层面工作,而不是在 token 层面工作。

**起源。** Li、Wei、Zhang、Zhang,"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"(2024);EAGLE-2(2024)引入了动态草稿树;EAGLE-3(2025 年 3 月)是当前版本。

**核心机制。** token 层面的起草器必须以小得多的容量,从头重新推导大模型早已"知道"的关于上下文的一切。EAGLE 的草稿头则直接拿目标模型对已有 token 计算出的隐藏状态——这是目标模型本来就会算出的丰富摘要——只需在其上加一个轻量层来猜测接下来的内容;再由目标模型自己的输出层把猜测转换成候选 token。候选被组织成一棵**草稿树**(draft tree,从每个位置分叉出多个备选续写),使一次验证就能检验许多种可能;EAGLE-2 会自适应地生长这棵树,在起草器有把握的地方扩展更多分支。

EAGLE-3 做了两处改动:
- **多层特征。** 它把目标模型多个深度(低层、中层、高层)的特征输入草稿头,而不只是最顶层,让起草器同时获得表层信息和抽象信息。
- **"训练时测试"(training-time test)。** 训练时,草稿头被喂入它*自己*先前的预测——与推理时它连续起草多个 token 的情形完全一致——而不是始终拿到真实特征。这消除了"在完美输入上训练"与"在自己不完美的输出上使用"之间的失配;否则,草稿中越靠后的 token 会越来越不可靠。

**为什么重要。** 据报道,把起草器挂在目标模型的特征上,比成本相近的独立小草稿模型能获得高得多的接受率,同时免去了让独立模型保持匹配的运维负担。验证环节不变,因此输出分布仍然精确等于目标模型的分布。

**当前状态(2026 年底)。** 据广泛报道,它已成为开源服务栈中默认的投机解码选择之一(vLLM 和 SGLang 都支持 EAGLE 式起草器);起草器需要针对每个目标模型单独训练。

## 复用多 token 预测(MTP)头作为起草器

**名称与定义。** **多 token 预测(Multi-Token Prediction,MTP)**目标在预训练期间训练模型,不仅预测下一个 token,还借助主网络之上的小型额外预测模块,预测更靠后的一个或多个 token。

**起源。** Gloeckle 等(Meta),"Better & Faster Large Language Models via Multi-token Prediction"(2024)把 MTP 作为训练目标进行了研究;DeepSeek-V3(2024 年 12 月)在前沿规模模型中使用了 MTP 目标。

**核心机制。** 在 DeepSeek-V3 中,MTP 模块作为主预训练目标的一部分进行训练——额外的预测目标让模型从每条序列中获得更密集的训练信号。推理时,这个模块可以直接丢弃,*也可以*保留下来复用为投机解码的起草器:它提议"下下个"token,主模型在下一次前向传播中按经典投机解码的方式对其进行验证。

**与 Medusa 的区别。** Medusa 的预测头是在训练*之后*才加装上去的,每个头都从同一个隐藏状态出发、各自独立地预测自己负责的偏移位置。MTP 模块则从一开始就与模型*联合*训练;在 DeepSeek-V3 的设计中,这些模块按顺序串联——每一个都建立在前一深度的表示之上,而非独立预测——从而保持了因果链的完整。实际效果是:起草器成了预训练的免费副产品,并且自动与目标模型相匹配。

**状态。** 已用于 DeepSeek 系列;与往常一样,它能带来多少加速取决于具体负载下的接受率。

## 前瞻解码(Lookahead decoding)

**起源。** Fu 等,"Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"(2024)。

**核心贡献。** 前瞻解码用目标模型自身来生成并验证候选续写(完全不需要独立的草稿模型或额外训练的预测头),它利用了对生成过程的一种数学重构(与可并行化的不动点迭代方法相关,和某些用于求解看似串行问题的经典并行算法属于同一大类思路),让多个未来 token 的猜测在若干次前向传播中被并行地逐步修正,最终收敛到模型逐个 token 生成时会产生的相同输出。

**为什么重要。** 由于无需任何辅助模型或架构改动,前瞻解码可以直接在现有、未经修改的目标模型上获得加速,代价是每步多花一些计算(同时生成并修正多个猜测),换来更少的总串行步数。

## 另一条路线:扩散语言模型的并行解码

上述所有方法都保留一个自回归目标模型,只改变产生其 token 的*过程*,这正是输出分布得以精确保持的原因。**扩散语言模型(diffusion language model,dLLM)**追求的是同一个目标——每次昂贵的模型调用产出不止一个 token——但改变的是*模型本身*。dLLM 从一段完全被遮盖(空白)的位置开始,在若干轮细化步骤中并行地填充其中许多位置,思路与图像扩散模型一脉相承(见 [diffusion-models.md](../04-generative-models/diffusion-models.md))。由于细化步数可以少于 token 数,按每个 token 计算的生成速度可以快得多。

实例:LLaDA(一个开放的 8B 掩码扩散语言模型,2025)、Inception Labs 的 Mercury(2025),以及 Google 的 Gemini Diffusion(2025 年 5 月发布的实验性模型)。Mercury 2(2026 年 2 月)被宣传为首个具备推理能力的 dLLM,速度约每秒 1,000 个 token——这是厂商的说法,而非独立测量。

**与投机解码的关键对比。** dLLM 是一个*不同的模型*,有它自己的输出分布;不存在"与慢速模型输出完全相同"的保证,而且减少细化步数通常意味着以质量换速度。截至 2026 年底,dLLM 能否在规模化后在质量上追平强大的自回归模型,仍是一个开放问题。(ICML 2026 的一篇杰出论文《The Flexibility Trap》认为,dLLM 能以任意顺序生成这一能力的价值不如人们通常设想的那么大——这提醒我们,dLLM 的主要卖点在于速度,而非生成顺序的灵活性。)

## 实际加速数字——定性描述,不夸大

按照本知识库的质量标准,本节刻意不把精确的加速倍数当作普遍事实来陈述,因为实际加速在很大程度上取决于草稿模型(或 Medusa 的预测头、或前瞻解码的猜测)与目标模型在所生成的具体文本类型上的实际行为有多匹配——高度可预测的文本(结构化、模板化或低熵的输出)获得的加速比高度创造性或难以预测的文本更大,因为后者的草稿猜测被接受得更少。可以有把握地定性说明的是:据广泛报道,投机解码及其变体在典型生产负载中能带来有意义、有时相当可观的实际生成加速,且对输出质量/分布毫无影响——这正是截至 2026 年底这些技术在生产 LLM 服务系统中被广泛、迅速采用的原因,与之形成对比的是量化、剪枝、蒸馏这类需要明确做出"以质量换速度"决策的技术。挂在目标模型上的起草器(EAGLE 式预测头、MTP 模块)之所以流行,正是因为它们免去了维护一个独立草稿模型的麻烦。随着推理模型的兴起,这一切的价值进一步提升:当模型在作答前要写出数千个"思考"token 时,解码速度主导了用户的等待时间(见 [serving-and-batching.md](serving-and-batching.md))。

## 对比表

| 技术 | 是否需要独立草稿模型? | 是否改变输出分布? | 关键机制 |
|---|---|---|---|
| 经典投机解码 | 是 | 否 | 小草稿模型提议,大模型并行验证 |
| Medusa | 否(目标模型上的额外预测头) | 否 | 多个内置预测头提议,按树结构验证 |
| 前瞻解码 | 否 | 否 | 对多个未来 token 猜测进行并行迭代修正 |
| EAGLE / EAGLE-3 | 否(挂在目标模型特征上的轻量草稿头) | 否 | 基于多层隐藏状态的特征级起草;草稿树一次验证 |
| 作为起草器的 MTP 模块 | 否(预训练时与模型一起训练的模块) | 否 | 预训练得到的额外预测模块提议"下下个"token |
| 扩散语言模型并行解码 | 不适用(另一类模型) | 是——并非任何自回归模型的精确加速 | 迭代地并行揭开许多被遮盖的位置 |

## 与其他算法的关系

- 投机解码运行在 [kv-cache-and-attention-optimization.md](kv-cache-and-attention-optimization.md) 中的 KV cache 管理技术之上,并与之完全兼容。
- 它是一种可与量化、剪枝/蒸馏(见 [quantization.md](quantization.md)、[pruning-and-distillation.md](pruning-and-distillation.md))叠加使用的互补技术——那些技术改变模型本身;投机解码只改变生成的*过程*。
- 服务系统把投机解码与 [serving-and-batching.md](serving-and-batching.md) 中的批处理策略结合起来,以最大化整体吞吐量。投机解码尤其能加速受显存带宽限制的 *decode* 阶段,因为一次验证多个 token 复用了同一批权重读取。
- 多 token 预测是一种预训练目标(见 [pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md)),其副产品就是一个起草器;大规模使用它的 DeepSeek-V3 是一个混合专家模型(见 [mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md))。
- 扩散语言模型(见 [diffusion-models.md](../04-generative-models/diffusion-models.md))是实现并行 token 生成的另一条非精确路线。

## 来源

- Leviathan, Kalman, Matias, "Fast Inference from Transformers via Speculative Decoding" (2023)
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023)
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (2024)
- Fu, Bai, Zhao, et al., "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding" (2024)
- Li, Wei, Zhang, Zhang, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (2024)
- Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees" (2024)
- Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test" (2025)
- Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (2024)
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024) [MTP 目标;MTP 模块可复用于投机解码]
- Nie et al., "Large Language Diffusion Models" (2025) [LLaDA]
- Inception Labs, Mercury (2025) 与 Mercury 2 (2026 年 2 月) 发布信息 [吞吐量数字为厂商说法]
- Google DeepMind, Gemini Diffusion (实验性,2025 年 5 月发布)
- "The Flexibility Trap: Rethinking the Value of Arbitrary Order in Diffusion Language Models," ICML 2026 杰出论文
