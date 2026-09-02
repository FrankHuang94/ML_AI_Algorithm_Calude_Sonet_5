# 缓存和注意力优化

关注的次数成本在序列长度上 (见 [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)其他[state-space-models.md](../03-deep-learning-architectures/state-space-models.md)由于这种问题是由于在线学习的过程中,它可能会导致一个更重要的问题,即KV缓存.该文件将从零开始解释这个问题,然后涵盖用于管理的四种主要技术:FlashAttention,PagedAttention和MQA/GQA的建筑变化家族.

## 开车缓存,从零开始解释

**为什么反逆的代人需要它.**提醒您[autoregressive-generation.md](../04-generative-models/autoregressive-generation.md)通过在一个时间内生成一个文本词元,则意味着模型会重复运行,每次处理一个额外的词元,添加到增长序列.[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)计算在特定位置上的自注意力需要该位置的**查询**向量,以及**关键**其他**价值**变量*每一个*位置到包括它.

**简单的,浪费的方法.**没有缓存,生成每一个新的词元将需要重新计算关键和值向量*每一个*对于一个长度 n 的序列,这意味着 n 代步的总工作大约增加, n 代步的总工作大约增加, n 代步重新计算,冗余地,已经在之前的步骤中完成的工作.

**维修了KV缓存.**相反,每一个位置的关键和值向量在第一次计算时,只需在每一步计算后,只需要计算新查询,关键和值,以计算最新词元.这将冗余的,不断增长的重新计算转化为每一步的线性新工作量 (计算K/V一次,对所有缓存的K/V进行新查询) .

废物 (和固定) 最容易看到的步骤按步骤 大写 = 刚计算这个步骤,小写 = 从缓存中重复使用:

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

缓存将存储存储存换成计算:你永远不会以存储它们全部的代价重新计算词元的密钥/值.在计算上,长时间的对话中,交易是值得的,但存储的缓存本身就成为了新的瓶,这就是这份文件的其他内容.

**记忆成本是多少?**凯维缓存不自由,它必须存储在某个地方,其尺寸随着序列长度而线性增长,并且随着注意力头数,每个头的尺寸,模型中的层数和批量大小而扩展.对于长度的文本和/或大批量,凯维缓存可以成为总GPU内存使用量的很大一部分.[serving-and-batching.md](serving-and-batching.md).

##          

**起源.**达奥,福,埃尔蒙,鲁德拉,雷,"FlashAttention:快速和有效的记忆精确的注意力与IO-Awareness" (2022),后续版本 (FlashAttention-2,FlashAttention-3) 进一步改善硬件利用.

**它解决了问题.**在现代加速器上,在 GPU 的慢性主存储器 (HBM 高带宽内存,仍然比芯片的内载快速内存慢得多) 和其更快,但更小的芯片内存之间移动数据往往是注意计算的实际瓶,而不是原始算法本身.[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)) 在中间阶段计算和存储整个n×n注意力分数矩阵在缓慢的内存中,这涉及大量的数据前后移动,否则是无法避免的.

**核心机制 (概念性"I/O意识",而不是不同的数学结果).**闪电注意力计算了*完全相同*作为标准公式的注意力输出 这是一个关键的,容易被错过的点:它不是一个近似的 ,而是重组计算来处理输入的小块,这些块适合GPU的快速芯片内存,计算部分注意力结果块按块并结合它们,这样整个n×n分数矩阵永远不会完全实现缓慢内存.这是一个纯工程级重组的案例.*如何*计算计划,并且其中间结果在现场,数学结果变化为零,仅仅通过更好地将计算的内存访问模式与硬件实际性能特性相匹配,产生了实质性的现实世界加速.

**为什么这很重要.**闪存注意力几乎是标准注意力实现的默认,随时取代的,基本上所有严重的Transformer训练和推理堆,因为它是严格的更快,更有效的记忆,

## 页面注意  KV缓存操作系统页面类比

**起源.**和其他",用页面注意力服务的大型语言模型的有效内存管理" (2023) 广泛使用的vLLM服务系统的基础算法 (见 [serving-and-batching.md](serving-and-batching.md)).

**它解决了问题.**简单地说,服务系统可能会为每个请求的KV缓存保留一个大连续的内存块,以达到请求可能达到的最大序列长度的尺寸. 这会浪费大量的内存.大多数请求不会达到最大长度,而预留最坏的情况下,预先意味着内存在请求的持续时间内不会被使用 (但无法用于其他请求),严重限制在固定内存预算内可以同时服务的请求数量.

**核心机制 操作系统页面类比.**页面注意力管理KV缓存存储器,就像操作系统管理运行程序的虚拟内存一样:而不是每次请求的一个大连接配置,KV缓存被分解成小,固体大小**块**("页面",直接类似于OS内存页面),只分配给请求只因为它实际需要它们 (随着序列的成长),以及轻量级的搜索表 (类似于OS页面表) 追踪了哪些物理块属于每个请求序列中的逻辑位置. 块不需要在内存中物理连接.

**为什么这很重要.**这种存储管理技术,比任何单一的算法改变注意力本身,被广泛报告为大幅增加可实现的服务吞吐量 (每单位GPU内存服务的更多同步请求) LLM推理,其底层的区块/页面表设计已成为现代高吞吐量 LLM服务系统中的标准架构模式.

## 复杂查询注意 (MQA) 和组合查询注意 (GQA)

**这些问题是解决的问题.**标准多头注意力 (见[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)) 给每一个注意力头一个独立的键和值预测 这意味着KV缓存必须存储一个独立的键和值集 *按头*通过将KV缓存的内存成本乘以头数.

**复式查询注意力 (MQA).**起源:Shazeer, "快速Transformer解码:一个写头就是你需要的" (2019).MQA有每个关注头都分享一个*单身*按键和值预测的集合 (只需查询预测按头保持分别),这将KV缓存缩小大约为头数的一个因素大减少,因为键和值,实际上被缓存的东西,不再按头重复.

**集成问答注意 (GQA).**起源:Ainslie等人",GQA:从多头检查点训练通用多头Transformer模型" (2023年).GQA是标准多头注意力 (每个头都有自己的K/V) 和MQA之间的中间路线 (所有头都共享一个K/V):头被分成较小的组,组内的所有头都共享一个重点/值预测.这使得MQA的大部分KV缓存存存存存存,同时实验性地保留了比纯MQA的标准多头注意力质量更多 (拥有中等数量的独立K/V预测,而不是单个,似乎保留了多头的代表性好处).

**为什么这很重要.**随着KV缓存在长的文本长度和高的同时请求量成为越来越主导的成本,MQA/GQA成为最具影响力的成本之一.*建筑*对于控制长期服务成本的改变 (与FlashAttention/PagedAttention不同,这些都是基于推理服务的水平优化,不需要改变训练模型),这就是为什么GQA已经成为许多现代LLM中标准的建筑选择,到2026年,提供了MQA的大部分服务成本益处,但没有更大的质量妥协.

## 较量表

|技术|优化了什么?|改变模型架构?|改变了注意力输出?|
|---|---|---|---|
| KV缓存 (基线概念) |避免在生成阶段中冗余的重新计算|没有|没有|
|快速注意|记忆运动/计算注意力的速度|没有|没有 (确切,结果相同) |
|页面关注 |随时请求中的KV缓存存储存碎片化/废物|没有 (服务水平) |没有|
|商|KV缓存存储量 (所有头脑共享K/V) |是的 (必须以这种方式进行培训/转换) |约 (一些质量差距) |
|商|KV缓存存储量 (在头组中共享K/V) |是的 (必须以这种方式进行培训/转换) |约 (质量比MQA较小的权衡) |

## 与其他算法的关系

- 量化对服务的相关性背后的直接动机问题是KV缓存及其内存成本 (见[quantization.md](quantization.md)) 以及在批决定的背后[serving-and-batching.md](serving-and-batching.md).
- 建筑设计的决定与位置编码和规范化的选择相同.[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md).
- 关注的底层方位成本,所有这些部分都从不同的角度来处理,是相同的问题状态空间模型 (见[state-space-models.md](../03-deep-learning-architectures/state-space-models.md)) 试图在建筑上回避而不是优化周围.
- 投机解码 (见 [speculative-decoding.md](speculative-decoding.md)) 是一种补充技术,以不同的角度来处理生成速度 (减少序列模型调用数量),而不是每个个别的注意力计算成本.

## 来源

- 达,福,埃尔蒙,鲁德拉,雷, "闪光注意力:快速,有效的记忆精确注意力与IO意识" (2022)
- 昆等人",用页面注意力服务的大型语言模型的有效记忆管理" (2023)
- 莎泽, "快速Transformer解码:一个写头就是你需要的" (2019) [国际货币管理局]
- 艾恩斯利等人",GQA:从多头检查点培训通用多题Transformer模型" (2023)
