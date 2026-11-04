# KV cache 与注意力优化

注意力的计算成本随序列长度呈二次方增长(见 [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md) 和 [state-space-models.md](../03-deep-learning-architectures/state-space-models.md)),这在训练时是个问题;但在推理时,它还会引出一个独立的、可以说在运维上更重要的成本问题,其核心是所谓的 KV cache(键值缓存)。本文先从零解释这个问题,再把用来管理它的技术分成两组介绍:**服务层技术**——改变注意力的计算方式或缓存的存储方式,而不改动训练好的模型(FlashAttention、PagedAttention);以及**架构层技术**——在预训练阶段就确定,用来缩小、限窗、稀疏化或压缩需要缓存的内容(MQA/GQA、多头潜在注意力 MLA、滑动窗口注意力、DeepSeek 稀疏注意力,以及 DeepSeek-V4 的压缩注意力)。

## 从零解释 KV cache

**为什么自回归生成需要它。** 回顾 [autoregressive-generation.md](../04-generative-models/autoregressive-generation.md):逐个 token 生成文本,意味着要反复运行模型,每次都多处理一个追加到序列末尾的新 token。再回顾 [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md):在某个位置计算自注意力,需要该位置的**查询(query)**向量,以及从开头到该位置(含)*每一个*位置的**键(key)**和**值(value)**向量。

**朴素而浪费的做法。** 如果不做任何缓存,每生成一个新 token,都要把序列中*所有*先前位置的键和值向量从头重新算一遍——尽管这些较早 token 的键和值一经算出就不会再变(它们只取决于这些较早的 token 本身,而这些 token 一旦生成就固定了)。对长度为 n 的序列,n 步生成的总工作量大约按 n² 增长——大量重复计算之前步骤早已算过的东西。

**KV cache 的解决办法。** 改为在每个位置的键和值向量第一次算出时就把它们存(缓存)起来,之后每一步直接复用,每步只为最新的 token 计算新的查询、键和值。这样,原本冗余且不断膨胀的重复计算就变成了每步线性的新增工作量(为新 token 计算一次 K/V,再用新查询对所有已缓存的 K/V 做注意力)——对自回归生成来说是很大的实际加速。

把过程逐步摊开最容易看清浪费(以及修复)——大写 = 本步新计算,小写 = 从缓存复用:

```
   Generating "The cat sat on":

   WITHOUT cache (recomputes everything, every step):    WITH KV cache (compute only the new one):
   step 1  "The"            → K/V for: [THE]              → compute & store K/V[THE]
   step 2  "The cat"        → K/V for: [THE  CAT]         → reuse the, compute & store K/V[CAT]
   step 3  "The cat sat"    → K/V for: [THE  CAT  SAT]    → reuse the,cat, compute K/V[SAT]
   step 4  "...sat on"      → K/V for: [THE CAT SAT ON]   → reuse the,cat,sat, compute K/V[ON]
                              ^^^ recomputes the same           ^^^ each token's K/V computed
                              early tokens over and over        exactly ONCE, then reused
                              (work ∝ n² total)                 (work ∝ n total)
```

缓存是用内存换计算:你永远不必重算一个 token 的键/值,代价是要把它们全部存下来。对于长对话,这笔交易在计算上无疑非常划算——但存下来的缓存本身成了新的瓶颈,这正是本文其余部分要讨论的内容。

**它的内存代价。** KV cache 不是免费的——它必须存放在某处,其大小随序列长度线性增长,同时还随注意力头数、每个头的维度、模型层数以及批大小(同时生成的序列数量)成比例增长。对标准多头注意力写成公式:

```
KV cache bytes = 2 × n_layers × n_kv_heads × d_head × seq_len × batch_size × bytes_per_number
                 ↑ one K and one V
```

逐项解读:每一层都要为批中每条序列的每个 token、为每个 KV 头存一个键向量和一个值向量,每个向量长 `d_head`。每个因子都是一根可以撬动的杠杆:下文的技术分别针对 `n_kv_heads × d_head`(MQA、GQA、MLA)、`seq_len`(滑动窗口、序列压缩)、需要保留缓存的层数(线性注意力混合架构)以及 `bytes_per_number`(KV cache 量化——见 [quantization.md](quantization.md))。在长上下文和/或大批量时,KV cache 可能占到 GPU 总显存用量的很大一部分——在某些情况下甚至超过模型权重本身——因此高效管理它是服务成本和吞吐量的头等问题,与 [serving-and-batching.md](serving-and-batching.md) 中的批处理决策直接相关。

还有第二个更隐蔽的代价:生成过程中,每个新 token 做注意力时都要从显存中*读取*整个缓存。所以大缓存不仅占存储,还会在每一步解码时消耗显存带宽——这就是为什么*少读*一部分缓存(见下文稀疏注意力)即便没有减少存储量,也能提升速度。

## FlashAttention——I/O 感知的注意力

**起源。** Dao、Fu、Ermon、Rudra、Ré,"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"(2022);后续版本(FlashAttention-2、FlashAttention-3)进一步提升了硬件利用率。

**它解决的问题。** 在现代加速器上,数据在 GPU 较慢的主存(HBM,高带宽内存——虽然名为高带宽,仍比芯片上的快速存储慢得多)与快得多但小得多的片上存储之间搬运,往往才是注意力计算的真正瓶颈,而不是算术运算本身。注意力公式(见 [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md))的朴素实现会在中间步骤把完整的 n×n 注意力分数矩阵计算出来并存入慢速内存,带来大量本可避免的来回数据搬运。

**核心机制(概念层面——"I/O 感知",而不是另一种数学结果)。** FlashAttention 计算出的注意力输出与标准公式*完全相同*——这一点至关重要又容易被忽略:它不是近似——但它重组了计算过程:把输入切成能放进 GPU 快速片上存储的小块,逐块计算部分注意力结果并增量合并,使完整的 n×n 分数矩阵根本不需要在慢速内存中完整生成。这是一个纯工程层面的重组案例——只改变计算*如何*调度、中间结果放在哪里,数学结果丝毫不变——仅仅通过让计算的访存模式更契合硬件的实际性能特性,就带来了可观的实际加速。

**为什么重要。** FlashAttention 几乎成了所有正经 Transformer 训练与推理栈中标准注意力实现的默认、即插即用替代品,因为在数学结果完全相同的前提下,它严格更快、更省内存——在本知识库涵盖的内容中,这是少有的"几乎没有真正代价"的优化。

## PagedAttention——借鉴操作系统分页管理 KV cache 内存

**起源。** Kwon 等,"Efficient Memory Management for Large Language Model Serving with PagedAttention"(2023)——这是被广泛使用的 vLLM 服务系统背后的算法(见 [serving-and-batching.md](serving-and-batching.md))。

**它解决的问题。** 朴素的服务系统可能会为每个请求的 KV cache 预留一整块连续内存,大小按该请求可能达到的最大序列长度来定。这会浪费大量内存——大多数请求达不到最大长度,而提前按最坏情况预留,意味着这部分内存在请求的整个生命周期内闲置(却又无法给其他请求使用),严重限制了在固定内存预算内能并发服务的请求数量。

**核心机制——操作系统分页类比。** PagedAttention 像操作系统为运行中的程序管理虚拟内存那样管理 KV cache:不再为每个请求做一次大的连续分配,而是把 KV cache 切成小的、固定大小的**块**("页",直接对应操作系统的内存页),只在请求真正需要时(随着序列增长)才分配给它;再用一张轻量的查找表(对应操作系统的页表)记录每个请求序列中的哪个逻辑位置对应哪个物理块。这些块在内存中不必物理连续——由查找表负责间接寻址——从而消除了按最坏情况预先预留而造成的浪费,让服务系统能在同样的物理内存中塞下多得多的并发请求。

**为什么重要。** 据广泛报道,这一内存管理技术——比任何对注意力本身的单项算法改动都更显著——大幅提升了 LLM 推理可达到的服务吞吐量(每单位 GPU 内存能服务更多并发请求),其底层的块/页表设计已成为现代高吞吐 LLM 服务系统的标准架构模式。同样的块结构也让以相同提示前缀开头的请求之间*共享*缓存块变得很便宜——这正是前缀缓存(prefix caching)的基础,见 [serving-and-batching.md](serving-and-batching.md)。

## 多查询注意力(MQA)与分组查询注意力(GQA)

**它们要解决的问题。** 标准多头注意力(见 [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md))给每个注意力头各自独立的键和值投影——这意味着 KV cache 必须*按头*分别存储一套键和值,内存开销随头数成倍增加。

**多查询注意力(MQA)。** 起源:Shazeer,"Fast Transformer Decoding: One Write-Head is All You Need"(2019)。MQA 让所有注意力头共享*同一套*键和值投影(只有查询投影仍按头独立),由于真正被缓存的键和值不再按头重复,KV cache 大约缩小为原来的"头数分之一"——缩减幅度很大。

**分组查询注意力(GQA)。** 起源:Ainslie 等,"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"(2023)。GQA 介于标准多头注意力(每个头各有 K/V)和 MQA(所有头共享一套 K/V)之间:把头分成较少的若干组,组内所有头共享一套键/值投影。这样既获得了 MQA 的大部分 KV cache 节省,在经验上又比纯 MQA 保留了更多标准多头注意力的质量(保留适量相互独立的 K/V 投影,而非只有一套,似乎能保住多头带来的更多表达能力)。

**为什么重要。** 随着 KV cache 内存在长上下文和高并发请求下越来越成为主导成本,MQA/GQA 成为控制长期服务成本最有影响力的*架构*改动之一(它们在预训练时就要确定,不同于 FlashAttention/PagedAttention 这类无需改动训练好的模型的推理服务层优化)——这就是为什么截至 2026 年底,GQA 已成为许多现代 LLM 的标准架构选择:它以更小的质量代价,获得了 MQA 的大部分服务成本收益。

## 多头潜在注意力(MLA)

**名称与定义。** 多头潜在注意力(Multi-head Latent Attention,MLA)把每个 token 的键和值压缩成一个小的共享**潜在向量**(latent vector,一种模型自己学出来的紧凑摘要——"潜在"指它是模型内部学到的表示,而非预先定义含义的量),只缓存这个潜在向量,在计算注意力时再由它重建出每个头的键和值。

**起源。** DeepSeek-AI,"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"(2024 年 5 月);DeepSeek-V3 与 DeepSeek-R1 沿用。

**核心机制——直觉。** MQA 和 GQA 靠让多个头*共享*键和值来省内存,这会丢掉一部分多头的价值。MLA 押的是另一个赌注:一个 token 在所有头上的键和值高度冗余,因此可以把它们挤过一个狭窄的瓶颈,而*不必*强迫各头共享。可以把它想成存压缩包而不是原文件:缓存里只放一个小压缩包,需要时再"解压"成完整的每头 K 和 V——而由于"解压"是一个固定的线性映射,它大多可以被并入其他矩阵乘法,实际上并不需要真的执行。

```
c_t        = W_down · h_t           (h_t: token t's hidden state; c_t: small latent, THIS is cached)
k_t^(i)    = W_upK^(i) · c_t        (key for head i, rebuilt from the latent)
v_t^(i)    = W_upV^(i) · c_t        (value for head i, rebuilt from the latent)
```

逐项解读:`W_down` 是一个学出来的"下投影"矩阵,把 token 的完整隐藏状态映射成潜在向量 `c_t`,它远比所有头的键和值加起来短。每个头 `i` 各有自己学出来的"上投影"矩阵 `W_upK^(i)` 和 `W_upV^(i)`,把共享的潜在向量还原成该头自己的键和值——所以与 MQA/GQA 不同,每个头仍然拥有各不相同的键和值。让它在推理时变快的诀窍是:头 `i` 的注意力分数是 `q^(i) · k_t^(i) = q^(i) · (W_upK^(i) · c_t)`,而矩阵乘法满足结合律,因此可以把 `W_upK^(i)` 预先乘到查询一侧(称为"吸收"),让查询直接与缓存的潜在向量比较,完全不必显式生成完整的键。值的上投影同样可以被吸收进注意力的输出投影。

**一个麻烦——位置信息。** RoPE(旋转位置编码,见 [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md))会按与位置相关的角度旋转键,这破坏了"把上投影吸收进查询"的技巧(旋转夹在两个矩阵之间,且随位置变化)。MLA 的解决办法是**解耦的 RoPE 键**(decoupled RoPE key):为每个 token 额外存一个很小的键分量来承载位置旋转,各头共享,与潜在向量一起缓存。于是每个 token 每层的缓存就是"潜在向量 + 一个小的位置键"。

**每个 token 缓存了什么——示意图。** 下面是一个假想层的示意数字:32 个头,每头维度 128(这些*不是*任何具体模型的配置,只用来展示对比的形态):

```
 Numbers cached per token, per layer           (bar length ∝ cache size)

 MHA    32 heads × (K 128 + V 128)       ████████████████████████████████  8,192
 GQA-8   8 groups × (K 128 + V 128)      ████████                          2,048
 MLA    1 latent (512) + RoPE key (64)   ██▎                                 576
 MQA     1 shared × (K 128 + V 128)      █                                   256

 MHA / GQA / MQA: cache the actual keys and values (fewer distinct heads → smaller, but heads must share)
 MLA:             caches a compressed latent; every head still gets its own K/V, rebuilt on the fly
```

真正要比的不只是绝对大小——MQA 还更小——而是*在同等质量下*的大小。DeepSeek 报告称,MLA 的缓存远小于 GQA,质量却与标准多头注意力持平甚至更好,而 MQA 和 GQA 通常做不到这一点。

**为什么重要。** MLA 表明,KV cache 大小与注意力质量之间并不是一条只由"多少个头共享 K/V"控制的一维权衡曲线。它是 DeepSeek 系列模型据报道服务成本很低的核心要素之一,并被 DeepSeek 以外的模型采用——例如月之暗面(Moonshot)的 Kimi Linear(2025 年 10 月)在其全注意力层中使用 MLA。

**当前状态(2026 年底)。** 已在 DeepSeek 系列及其他一些开放权重模型中投入生产;总体而言 GQA 仍是更普遍的默认选择。凡是 KV cache 大小属于头等问题的场合,MLA 都是主要替代方案。

**优势与局限。**
- 优势:在不强迫各头共享键/值的前提下大幅缩小 KV cache;据报道质量与多头注意力持平或更好;可与分页、量化和稀疏注意力组合使用。
- 局限:高效实现更复杂(需要定制 kernel,以及 RoPE 解耦);潜在向量是一个低秩瓶颈,其大小是一个新的超参数;与 MQA/GQA 一样必须在训练时就内建(原则上可以改造已有模型,但并非免费的即插即用)。

## 滑动窗口注意力与局部/全局交替

**名称与定义。** 在**滑动窗口(局部)注意力**中,每个 token 只关注最近的 W 个 token,而不是全部历史。**局部/全局交替**则是在大量局部层之间穿插少量**全局**(全注意力)层。

**起源。** 固定的稀疏注意力模式见于 Child 等,"Generating Long Sequences with Sparse Transformers"(2019)和 Beltagy 等,"Longformer"(2020);Mistral 7B(Jiang 等,2023)把滑动窗口注意力带进了一个被广泛使用的开放 LLM。

**核心机制。** 滑动窗口层永远只需要最近 W 个 token 的键和值,因此它的 KV cache 增长到 W 个 token 就停止了(可以用环形缓冲区实现——一个固定大小的数组,最新条目覆盖最旧条目)。更早的信息并没有完全丢失:每叠加一层局部层,信息就能再向前"跳"W 个位置,所以有效视野随深度增长。穿插少量全局层则保证任何 token 仍能被直接访问到。

```
 Layer pattern (example, L = local window, G = global/full):   L L L G L L L G ...
 KV cache per layer:  L layers → capped at W tokens     G layers → grows with the full context
```

**当今的应用。** Gemma 2 和 Gemma 3 交替使用局部滑动窗口层和全局层;OpenAI 的 gpt-oss(2025 年 8 月)交替使用带状(窗口化)注意力层和稠密注意力层。

**一个陷阱:注意力汇聚点(attention sink)。** 训练好的模型常常把很大一部分注意力倾注到序列最开头的几个 token 上,把它们当作"汇聚点",用来吸收无处安放的注意力权重。如果窗口滑过了这些 token,质量可能会崩溃。Xiao 等,"Efficient Streaming Language Models with Attention Sinks"(StreamingLLM,2023)表明,只要永久保留最前面几个 token 的 KV 条目,就能解决大部分问题;一些较新的模型则改为学习一个显式的汇聚点,而 Qwen 团队的门控注意力(gated attention,2025)——在注意力输出上加一个可学习的逐元素 sigmoid 门——据报道从根本上大体消除了汇聚点。

**优势与局限。** 简单、便宜;局部层的缓存有上限。但局部层确实无法直接看到很久以前的 token,因此长距离检索依赖于全局层,而全局层的缓存仍随上下文增长。

## 稀疏注意力:DeepSeek 稀疏注意力(DSA)

**名称与定义。** 稀疏注意力让每个查询只关注过去 token 中*被选出的一个子集*,而不是全部。DeepSeek 稀疏注意力(DeepSeek Sparse Attention,DSA)*针对每个查询、依据内容*来挑选这个子集,所用的是一个名为**闪电索引器(lightning indexer)**的低成本打分模块。

**起源。** DeepSeek-V3.2-Exp(2025 年 9 月)和 DeepSeek-V3.2(2025 年 12 月)。更早的稀疏注意力工作(Sparse Transformer、Longformer、BigBird,2019–2020)大多采用*固定*模式(窗口、步长、少量全局 token),而不是基于内容的选择。

**核心机制。** 对每个新的查询 token,执行一个"先略读、再精读"的两阶段过程:

```
for the current query token q:
    scores  = lightning_indexer(q, all previous tokens)   # cheap, small scoring network
    chosen  = top_k(scores, k)                            # keep the k most relevant positions
    output  = attention(q, K[chosen], V[chosen])          # expensive attention over only k tokens
```

逐项解读:索引器仍要看过每一个先前的 token,所以从整条序列看,它在技术上仍是二次方的——但它的设计目标是每个 token 的开销远低于真正的注意力。昂贵的主注意力则每个查询只触及 `k` 个 token 而不是全部 `n` 个,所以在长上下文下,大部分成本从"昂贵 × n"变成了"便宜 × n + 昂贵 × k"。与固定窗口不同,被选中的 token 可以位于历史中的任何地方——索引器能从很远之前挑出一段相关的内容。

**一个重要的细节。** DSA 减少了计算量和每步*读取*的缓存量,但它本身并不减少*存储*的缓存量——任何过去的 token 以后都可能被选中,所以全部都得保留。(存储的缩减来自本文的其他技术,例如 MLA 或下文的序列压缩。)

**状态。** 截至 2026 年底已在 DeepSeek V3.2 系列中投入生产;基于内容的稀疏注意力作为一种通用技术仍很年轻,它在各类长上下文任务上能否始终表现良好,仍是一个活跃的研究问题。

## DeepSeek-V4 中的压缩注意力:CSA 与 HCA(据报道)

**它是什么。** DeepSeek-V4 技术报告(2026 年 4 月;模型包括 V4-Pro,总参数 1.6T / 激活 49B,以及 V4-Flash,284B / 激活 13B,均为混合专家模型,支持 1M token 上下文)描述了一种交替使用两类层的混合注意力设计。以下细节均来自 DeepSeek 的报告;独立分析目前仍然有限。

- **压缩稀疏注意力(Compressed Sparse Attention,CSA)。** 每隔几个连续 token 就压缩成一个 KV 条目。一个闪电索引器(与 DSA 中的相同思路)挑出最相关的 top-k 个*压缩*条目,查询关注这些条目,外加一个由最近未压缩 token 组成的**滑动窗口**。
- **重度压缩注意力(Heavily Compressed Attention,HCA)。** 压缩更激进得多(据报道 KV 条目减少约 128 倍),对所有压缩块做普通的稠密注意力,同样外加一个最近 token 的滑动窗口。

```
 Past context:   [t1 t2 t3 t4][t5 t6 t7 t8][ ... ][ ... ]  | recent tokens (uncompressed window)

 CSA layer:  [c1][c2][c3][c4] ...  ─ indexer picks top-k ─►  attend to {top-k compressed} ∪ {window}
 HCA layer:  [====== C1 ======][====== C2 ======] ...     ►  attend to {all heavy blocks} ∪ {window}
```

**直觉——沿另一个轴压缩。** MLA 压缩的是每个 token 的*特征*(每个 token 很多个数 → 每个 token 较少的数)。CSA 和 HCA 压缩的是*序列*(很多 token → 较少的缓存条目)。这一设计随后把本文的三种思路组合起来:用序列压缩缩小存储,用 top-k 稀疏选择(CSA)只读取相关部分,再用滑动窗口保证最近的上下文总能被精确看到。把一种更细粒度、有选择性的层(CSA)与一种更粗粒度、一次看全部的层(HCA)搭配,是同时获得精确检索和低成本全局概览的一种方式。

**报告的效果。** DeepSeek 报告称,在相同的长上下文长度下,V4 的单 token FLOPs 约为 V3.2 的 27%,KV cache 约为其 10%。这些是厂商自己给出的数字;CSA 的确切压缩比(除"每几个 token"之外)以及它在各类长上下文任务上的质量权衡,都应视为尚未经过独立验证。

## 线性注意力混合架构(指引)

缩小缓存的另一条路,是把大多数注意力层替换为**线性注意力**层或状态空间层,它们保存的是*固定大小*的循环状态,而不是不断增长的 KV cache——只有剩下的全注意力层才保留缓存。例如线性层与全注意力层按 3:1 配比时,只有大约四分之一的层需要按 token 存储 K/V。生产实例包括 Qwen3-Next(2025 年 9 月)和 Qwen3.5(2026 年初)——以约 3:1 的比例使用 Gated DeltaNet 层;月之暗面的 Kimi Linear(2025 年 10 月)——以约 3:1 的比例组合 Kimi Delta Attention 与 MLA 层;以及 NVIDIA 的 Nemotron-H(Mamba-2 + 注意力)。相关机制见 [state-space-models.md](../03-deep-learning-architectures/state-space-models.md)。

## 现状(2026 年底)

FlashAttention 式的 kernel 和分页式 KV cache 管理在所有正经的服务栈中都已普及。在架构层面则没有唯一的赢家;长上下文前沿模型越来越多地*组合*使用上述多根杠杆——用 GQA 或 MLA 缩小每个 token 的缓存,用滑动窗口或线性注意力层限制大多数层的缓存增长,再用基于内容的稀疏或压缩注意力(DSA、CSA/HCA)让剩下的全上下文层更便宜。哪种组合泛化得最好仍是开放问题,而且已发表的对比大多来自各方法的设计者本身。

## 对比表

| 技术 | 优化的对象 | 是否改变模型架构? | 是否改变注意力的输出? |
|---|---|---|---|
| KV cache(基础概念) | 避免各生成步骤之间的冗余重算 | 否 | 否 |
| FlashAttention | 注意力计算中的数据搬运/速度 | 否 | 否(精确,结果相同) |
| PagedAttention | 并发请求间 KV cache 内存的碎片/浪费 | 否(服务层) | 否 |
| MQA | KV cache 内存大小(所有头共享 K/V) | 是(必须按此方式训练/转换) | 近似(有一定质量代价) |
| GQA | KV cache 内存大小(组内共享 K/V) | 是(必须按此方式训练/转换) | 近似(质量代价小于 MQA) |
| MLA | KV cache 大小(缓存压缩后的潜在向量,每头 K/V 现场重建) | 是(按此方式训练) | 机制不同;据报道质量与 MHA 持平或更好 |
| 滑动窗口 / 局部-全局 | 限制局部层的缓存增长 | 是(层模式在训练时确定) | 是(局部层只能看到最近 W 个 token) |
| DeepSeek 稀疏注意力 | 注意力计算量与缓存读取量(每个查询只取 top-k 个 token) | 是(索引器与模型一起训练) | 是(只关注被选出的子集) |
| CSA / HCA(DeepSeek-V4,据报道) | 通过序列压缩 + 稀疏选择 + 窗口降低缓存大小与计算量 | 是 | 是(关注压缩后的条目) |
| 线性注意力混合层 | 用固定大小的状态取代大多数层不断增长的缓存 | 是 | 是(不同的层类型) |

## 与其他算法的关系

- KV cache 及其内存成本,正是量化对服务意义重大的直接动因(见 [quantization.md](quantization.md)),也是 [serving-and-batching.md](serving-and-batching.md) 中批处理、前缀缓存以及 prefill/decode 分离等决策背后的直接动因。
- MQA/GQA/MLA 以及局部/全局层模式,与 [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md) 中的位置编码和归一化选择处于同一设计阶段。
- 本节从不同角度应对的注意力二次方成本,正是状态空间模型和线性注意力混合架构(见 [state-space-models.md](../03-deep-learning-architectures/state-space-models.md))试图在架构上绕开、而非在其周围优化的同一个问题。
- DeepSeek 的 MLA、DSA 和 CSA/HCA 都内置于混合专家模型中(见 [mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md));MoE 削减的是前馈层的每 token 计算量,而这些技术削减的是注意力/缓存一侧的开销。
- 投机解码(见 [speculative-decoding.md](speculative-decoding.md))是一种互补技术,它从另一个角度提升生成速度(减少串行的模型调用次数),而不是降低每次注意力计算本身的成本。

## 来源

- Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019) [MQA]
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023)
- DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (2024) [MLA]
- Child, Gray, Radford, Sutskever, "Generating Long Sequences with Sparse Transformers" (2019)
- Beltagy, Peters, Cohan, "Longformer: The Long-Document Transformer" (2020)
- Jiang et al., "Mistral 7B" (2023) [滑动窗口注意力]
- Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (2023)
- Gemma Team, Gemma 2 (2024) 与 Gemma 3 (2025) 技术报告 [局部/全局交替]
- OpenAI, gpt-oss-120b / gpt-oss-20b 模型卡 (2025) [带状窗口与稠密注意力交替]
- Qwen Team, "Gated Attention for Large Language Models" (2025) [门控注意力;据报道可消除注意力汇聚点]
- DeepSeek-AI, DeepSeek-V3.2-Exp (2025 年 9 月) 与 DeepSeek-V3.2 (2025 年 12 月) [DeepSeek 稀疏注意力、闪电索引器]
- DeepSeek-AI, DeepSeek-V4 技术报告 (2026 年 4 月) [CSA/HCA;FLOPs 与 KV cache 数字为厂商报告值]
- Moonshot AI, Kimi Linear (2025);Qwen Team, Qwen3-Next (2025) 与 Qwen3.5 (2026) [线性注意力混合架构——见 state-space-models.md]
