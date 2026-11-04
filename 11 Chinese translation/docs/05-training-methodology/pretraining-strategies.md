# 预训策略

预训是 (通常是巨大的) 初步培训阶段,[finetuning-and-peft.md](finetuning-and-peft.md)) 或调整工作 (见[rlhf-and-alignment.md](rlhf-and-alignment.md)根据"规模化法"的做法,决定要花多少计算时间用于模型大小与数据.

## 首先,一个先决条件:标记化

在下列任何目标都能实现之前,原始文本必须转化为**标记**模型实际使用 (这个术语在整个库中使用查看词典*成为*模型不读字符或整体词,它读出数万个字母的固定词汇.**字母**产品单位**标记器**.

主导的方法是**字节对编码 (BPE)**(和像WordPiece和Unigram/SentencePiece这样的亲戚).BPE通过简单的贪程序从训练组中学习:从单个字符 (或字节) 开始,然后反复找到最常见的邻近对并将其合并成一个新的单个词元,直到你建立了一个单个词元的词汇库.普通的字符最终变成单个词元 ("the", "ing"),稀有的字符被分成几个部分 ("词元化" → "词元" + "代码化"),真正新奇的字符串会回到字符/字节,这保证词元器可以表示*任何*没有"未知字"问题.

为什么这不仅仅是管道:词元器决定了下一个词元预测的"一步"意味着什么,并且它具有实际下游后果.词元计数驱动了文本窗口使用和API成本;词元器分成许多部分的语言 (通常不是英语或代码重文) 实际上使用了文本窗口的效率更低.词元化的奇特在许多令人惊的模型失败背后 (例如,难以计算字符等角色级任务,因为模型从来没有看到单个字母,只有字母小部分).**字节级别**它们可以在原始字节上操作,因此可以编码任何Unicode文本,情感符号或二元内容,而不会失败.

## 自主监督的学习,框架

**名称和定义**自主监督学习训练一个模型使用自动从输入数据中提取的标签,而不是要求人类手动注释示例.

**为什么这个局很重要.**整个现代预训练范式存在于自主监督使模型能够在有效无限量的原始,无标签数据 (从网上剪辑的文本,没有标题的图像等) 上训练.  避免了早期的纯粹监督深度学习的基本瓶,数据集的规模由人类可行的标签数据量限制. 下面是讨论语言的两个主导的自主监督目标;[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)本文件特别关注这些培训战略.

## 面具语言建模 (BERT式)

**核心机制.**随机隐藏 (掩盖) 一些词元的部分在输入序列中 (通常约15%) 并训练模型预测每个掩盖词元的原始身份,使用周围的文本从*两者*方向 (因为编码器的自注意力是无限制的/双向的见[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)).

**为什么它是自我监督.**任何人都不需要标签任何东西每个面具位置的"标签"只是原本存在的标志,

**什么是好的.**由于模型学习从某个位置前后使用背景,因此面具语言建模产生了适合*了解*任务 分类,提取一个答案问题的文本跨度,生成语义搜索的嵌入式 ,但自然不支持开放式生成 (见 仅用于解码器的讨论)[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)为什么?

## 原因/下一个标志预训 (GPT类型)

**核心机制.**训练模型以使用仅前的词元预测每个词元 (因果化伪装的自我注意),[autoregressive-generation.md](../04-generative-models/autoregressive-generation.md).

**为什么它是普遍目的的法学.**根据"[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)通过这种方式,我们可以将语言模拟的双向背景放弃,从而使理解和生成成为一个模型和一个训练信号.

## 抗训预训:CLIP

**名称和定义**对于真正相关的输入对 (例如图像及其标题) 的相对表示和对不相关的对的不同表示,采用InfoNCE类型的相对表示损失.[loss-functions.md](../01-foundations/loss-functions.md).

**起源.**拉德福德等人 (OpenAI), "从自然语言监督中学习可转移视觉模型" (2021) CLIP (对比语言图像预训练).

**核心机制.**CLIP训练两个独立的编码器,一个是图像,一个是文字,这样对于从网络中剪切的 (图像,字幕) 双组,对真正匹配的对的图像嵌入和文本嵌入具有高的共性相似性,而对不匹配的对的嵌入 (图像与该批次的随机其他字幕相对) 具有较低的相似性.这是通过InfoNCE损失训练,将"该批次的字幕实际上匹配这个图像"作为分类目标.

**为什么这很重要.**CLIP产生了生活在共享嵌入空间的图像和文本表示,使得零镜头图像分类 (通过将其嵌入与候选文本标签的嵌入进行比较,没有任何任务特定培训) 并且成为后来的多模式系统的基础构建块,包括作为几个文本到图像传播系统内的文本条件机制 (见[diffusion-models.md](../04-generative-models/diffusion-models.md)).

**现状.**CLIP型反向预训练仍然是构建多模拟嵌入模型和连接视觉和语言表示的标准技术,尽管最大的边界多模系统越来越多地通过单个大型模型共同训练视觉和语言,而不是仅通过单独的 CLIP型反向阶段.

## 实践中的规模化法:计算最佳训练

卡普兰等人 (2020) 和辛希拉 (霍夫曼等人,2022年) 的扩展法在建筑背景下引入了[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)专注于它们对预训练实践的具体意义.

**这些法律回答了实际的问题.**考虑到一个固定的计算预算 (一个真正的,艰难的限制训练计算成本是真正的钱,需要实际的墙钟时间),你应该如何将预算分为模型更大 (更多参数) 和训练更多数据 (更多词元) 之间?

**奇拉发现,重新定制为实践.**根据卡普兰等人的原始指导,**根据其规模的训练不足**对他们看到的数据量有太多参数.他们的计算最佳配方要求随着计算的增长,大约与模型大小相比例地扩大训练数据 (论文中常常引用的指纹规则是每个参数约有20个训练词元,尽管值得重复[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)这个特定的比率是从特定的实验设置中得到的经验性发现,而不是一个普遍的法律,不应被视为适用于每一个建筑,数据混合或培训目标的固定的常量.

**实际上,数据规模与参数规模的交易.**遵循奇拉风格的指导严格意味着训练比以前更大的模型,以获得比以前更大的数据,以获得真正的计算最佳结果但自2022年以来,真正的预训决策并没有仅仅仅是纯计算最佳的.一个将在长期的部署寿命内为大量用户提供服务的模型有强烈的激励力来训练.*更多*由于推理成本 (在每一个未来的请求中都会多次支付) 可以使额外的预训计算的一次性成本变得微小,并且达到类似质量的更小,更强训练的模型更便宜以运行.这是为什么几种有能力的,公众知名的模型已经在数据量上训练过了原始的辛奇拉最佳点,以其规模而言.[curriculum-and-data-strategies.md](curriculum-and-data-strategies.md)) 由于规模化法律一般认为具有合理高质量的,非冗余的数据,因此, 由于通过培训推进更多低质量的或高重复的词元, 无法可靠地复制规模化曲线预测对真正多样化的高质量的数据的相同收益.

## 2025-26年预训练实践的新进展

截至2026年末,上述目标仍是预训练的核心.2025-26年变化最大的是:算力花得有多 *省*、从每个序列中额外榨取了 *哪些训练信号*,以及训练过程 *如何分阶段*.本节只作简要梳理,具体机制见所链接的文件.

### 低精度预训练:先是FP8,然后是4位

**基本思路.** 神经网络训练早已使用 **混合精度(mixed precision)**——用较低精度的数值格式(16位)完成大部分矩阵乘法,同时保留一份较高精度的"主"权重副本,并让少数对数值敏感的运算保持32位.每个数用的位数越少,内存占用就越少、数据搬运就越少,而且在带有专用低精度单元的硬件上,每秒可完成的运算也更多.2025-26年的新闻在于:这一做法被大幅推进到了 *预训练本身*,而不只是推理(推理端缩小数值表示称为量化——见 [quantization.md](../06-inference-optimization/quantization.md)).

- **大规模FP8.** DeepSeek-V3(2024年12月;总参数671B、激活参数37B的混合专家模型)在前沿规模上使用FP8(8位浮点)混合精度完成训练,证明了8位训练在超大规模的生产级训练中是可行的,而不只是在小型实验中.
- **4位(NVFP4)预训练.** NVIDIA 的 "Pretraining LLMs with NVFP4"(arXiv 2509.25149,2025年9月;2026年3月修订)用4位 **NVFP4** 格式在10万亿(10T)词元上训练了一个12B的混合Mamba-Transformer模型,报告的MMLU-Pro准确率为62.58%,而FP8基线为62.62%——基本持平.4位浮点数只有16个可能取值,所以关键技巧是 **微缩放(microscaling)**:在NVFP4中,每16个相邻数值组成一个块,共享自己的FP8缩放因子(外加整个张量共享的一个FP32缩放因子),这样每个小块都能获得贴合自身数值的表示范围.(相关的 **MXFP4** 格式使用32元素的块和2的幂次缩放因子.)该论文还依赖若干稳定化手段:随机Hadamard变换(一种廉价的"混合"旋转,在舍入前把离群值分散开)、二维块缩放、对梯度使用 **随机舍入(stochastic rounding)**(按与上下两个可取值的接近程度随机向上或向下舍入,使舍入误差相互抵消而不是朝一个方向累积)、让少数敏感层保持较高精度,以及在训练后期切换到较高精度.

坦率地说:4位结果是硬件厂商在12B规模上报告的一次训练;FP8如今已在前沿规模上站稳脚跟,而在最大规模上进行4位预训练仍处于新兴阶段.

### 多词元预测(MTP)

**基本思路.** 标准的因果预训练在每个位置上只要求模型预测 *下一个* 词元.**多词元预测(Multi-Token Prediction,MTP)** 增加了额外的预测头(小型附加输出模块),从同一位置出发还要预测更远处的词元——下下个词元,以此类推.Meta 的研究者(Gloeckle 等人,2024)将其作为预训练目标进行了研究,DeepSeek-V3(2024)则在前沿规模模型中使用了MTP目标.人们声称它有两个好处:每个序列提供更密集的训练信号(每个位置监督多个未来词元),以及在推理时提供一个"免费"的加速器——额外的预测头可以复用为推测解码中的"草稿",提出接下来的词元,再由主模型验证(见 [speculative-decoding.md](../06-inference-optimization/speculative-decoding.md)).

### "中期训练(mid-training)":介于预训练与后训练之间的阶段

近期的技术报告越来越多地描述一个独立阶段:位于大规模预训练之后、后训练(微调与强化学习)之前,常被称为 **中期训练(mid-training)**(这个名称并不正式,各实验室对其边界的划分也不同).概括地说,它是在刻意调整过的数据配比和设置下,继续使用同样的下一个词元预测目标:

```
 pretraining (bulk)            →  mid-training                         →  post-training
 trillions of tokens, mostly      smaller, curated, higher-quality         SFT, RLHF/RL on
 general web/code/books,          mix: more math, code, reasoning          verifiable rewards
 shorter context windows          traces; long-context extension           (finetuning-and-peft.md,
                                  (training on much longer sequences)       rlhf-and-alignment.md)
```

- **长上下文扩展.** 注意力的成本随序列长度增长,因此大多数预训练词元都是在相对较短的上下文窗口中处理的;处理超长输入的能力是在后期加入的——通过在更长的序列上训练并调整位置编码(这一做法在2024年已可见,例如 Llama 3 在预训练接近尾声时分阶段扩展了上下文长度).
- **面向推理的数据.** 提高数学、代码和逐步推理文本的权重(包括模型生成的推理轨迹——见 [curriculum-and-data-strategies.md](curriculum-and-data-strategies.md)),能为随后的强化学习阶段提供更好的起点.

这实际上就是 [curriculum-and-data-strategies.md](curriculum-and-data-strategies.md) 中"数据配比调度"思想被单独命名、并拥有了自己的预算.

### 开源模型中的百万词元上下文

到2026年,开源权重模型宣传的上下文窗口已达到约一百万个词元.DeepSeek-V4(技术报告,2026年4月,据报道)就是一个例子:这是一个拥有1M词元上下文的MoE模型系列,它通过压缩注意力与稀疏注意力层的混合设计,使KV缓存和每个词元的注意力成本远低于同等长度下普通全注意力的水平,从而让如此长的上下文变得负担得起(机制见 [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md)).常见的提醒依然适用:宣传的上下文长度说明的是模型能 *接收* 多少文本,而不是它能多可靠地 *利用* 分布在全文各处的信息.

## 比较表

|目标|注意力模式|从中学到什么?|最适合的|
|---|---|---|---|
|面具语言建模|双向|预测来自周围的隐藏词元|了解,嵌入,分类 |
|原因/下一个标志的预测|原因 (伪装,仅向前) |预测每个词元从前文中|开放式生成,通用 LLM |
|异性 (CLIP式) |编码器 (N/A)|不同模式的匹配对比对不匹配对 |多型嵌入式,零射击分类|
|多词元预测(辅助目标)|因果|从每个位置预测多个未来词元|更密集的训练信号;内置的推测解码草稿头|

## 与其他算法的关系

- 面具和因果性预训目标是编码器与解码器架构讨论的培训战略对应.[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md).
- 培训目标是直接应用InfoNCE对比损失在[loss-functions.md](../01-foundations/loss-functions.md).
- 数据质量/保养,这里被引用为调节原始规模规律预测的因素,[curriculum-and-data-strategies.md](curriculum-and-data-strategies.md).
- 后训练阶段 (精细调节,RLHF) 建立在预训练的基础模型上[finetuning-and-peft.md](finetuning-and-peft.md)其他[rlhf-and-alignment.md](rlhf-and-alignment.md).
- 低精度数值格式(FP8、NVFP4、MXFP4)在 [quantization.md](../06-inference-optimization/quantization.md) 中从推理角度加以解释;它们对集群规模训练的影响见 [distributed-training.md](distributed-training.md).
- MTP预测头复用为草稿模型这一点与 [speculative-decoding.md](../06-inference-optimization/speculative-decoding.md) 相关.
- BERT式掩码语言建模是扩散语言模型所用掩码扩散目标的前身——见 [diffusion-models.md](../04-generative-models/diffusion-models.md).

## 来源

- 塞恩里希,哈多,伯奇, "含子词单元的罕见词的神经机器翻译" (2016) [BPE用于NLP]
- 德文,张,李,图塔诺瓦, "BERT:深度双向Transformer预训练语言理解" (2018)
- 拉德福德等人",通过生成式预训练提高语言理解" (2018) [GPT-1]
- 拉德福德等人",从自然语言监督中学习可转移视觉模型" (2021) [CLIP]
- 卡普兰等人",神经语言模型的扩展法则" (2020)
- 霍夫曼等人",训练计算最佳的大语言模型" (2022) [Chinchilla]
- Micikevicius 等人, "Mixed Precision Training" (2018);Micikevicius 等人, "FP8 Formats for Deep Learning" (2022)
- Gloeckle 等人(Meta), "Better & Faster Large Language Models via Multi-token Prediction" (2024)
- Llama Team(Meta), "The Llama 3 Herd of Models" (2024) [在预训练接近尾声时分阶段扩展长上下文]
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024) [大规模FP8训练;MTP目标]
- NVIDIA, "Pretraining LLMs with NVFP4" (arXiv 2509.25149, 2025;2026年3月v2) [4位预训练;报告的MMLU-Pro为62.58%,FP8为62.62%]
- DeepSeek-AI, DeepSeek-V4 技术报告(2026年4月,据报道) [1M词元上下文]
