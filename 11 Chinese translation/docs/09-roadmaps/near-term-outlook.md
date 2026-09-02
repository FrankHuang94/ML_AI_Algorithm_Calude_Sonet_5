# 短期前景 (6-18个月)

根据该库的方法 (见 报告) 报告,该文件涵盖了该库预计将在未来大约一年至18个月内持续或加速的趋势.[scope-and-methodology.md](../00-overview/scope-and-methodology.md)),每一个前性索赔都标记着`[Projection]`将这些视为来自公共研究趋势和广泛报道的行业方向的合理抽出,而不是确定的事实.

## 推进成本降低趋势

**根据目前的轨迹.**关于在 [06- 输入优化](../06-inference-optimization/)量化,GQA/MQA,页面关注,投机解码,连续批量均单独证明了大幅度的成本/延迟改善,并且越来越多地被结合而不是单独部署.

**`[Projection]`**未来6-18个月内,对特定质量水平的成本 (而不是单一重大突破) 持续增长,大致增长,[quantization.md](../06-inference-optimization/quantization.md)),更广泛地采用不需要单独的草案模型的投机解码变体 (Medusa式和Lookahead式方法见[speculative-decoding.md](../06-inference-optimization/speculative-decoding.md)),以及继续进行硬件软件联合设计 (新加速器代代与像FlashAttention继任者这样的内核级优化相结合).`[Projection]`预计这一趋势将是稳定的,复合的趋势,而不是单一的步骤变化,与过去几年观察到的模式一致.

## 持续的文本长度扩展

**根据目前的轨迹.**在过去几年中,文本窗口大幅增长 (见[timeline-2023-present.md](../08-history/timeline-2023-present.md)),由建筑变化 (GQA,更有效的定位编码外分) 和服务基础设施改进 (PagedAttention类型的内存管理) 结合.

**`[Projection]`**未来6至18个月,文本窗口可能会继续增长,*标题号*增长 (广告的最大文本长度) 较改进的模型实际情况更重要*使用*长文本 该领域广泛报告了"技术上可以接受N输入词元"和"可靠地关注并对所有这些N词元进行同样合理的处理"之间的持续差距,特别是在长文本中部位于,而不是开始或结束. `[Projection]`缩小这种可用性差距,而不是仅仅延长原始最大长度,

## 代理和工具使用培训方法

**根据目前的轨迹.**根据[timeline-2023-present.md](../08-history/timeline-2023-present.md)培训模式,以使用外部工具,在更长的任务视野,减少一步步监督成为主要的重点领域,建立在推理的RL发展中[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md).

**`[Projection]`**预计未来6-18个月将继续,对培训方法进行相当快速的投资,具体针对多步骤工具使用,更长的时间完成任务,以及更好的错误恢复 (一个模型在任务中发现和纠正自己的错误,而不是继续加剧) `[Projection]`诚实的预期是不均的,形的进展而不是一个平稳的能力曲线,与该领域历史上发展最快的能力领域实际上是如何的一致.[open-problems.md](open-problems.md)对于长远可靠性问题而言,

## 持续的MOE和稀疏性采用趋势

**根据目前的轨迹.**专家混合的采用 (见 [mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md)) 已经在多个边境实验室的最大模型中发展, 根据该文件所述的精度按计算FLOP逻辑.

**`[Projection]`**该库预计未来6-18个月将继续,可能会扩大,对最大规模模型的MoE采用,以及对更复杂的路由和负载平衡方案的持续研究,以缩小部分剩余的MoE差距 (培训稳定性,记忆足迹与计算差距见[mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md)). `[Projection]`密集模型不预计会消失 这个库项目继续使用密集架构,特别用于较小,延迟敏感或更简单的部署场景,其中MoE的增加复杂性没有理由,而不是完全转向MoE.

## 总结表

|趋势|信心基础|方向`[Projection]` |
|---|---|---|
|降低价成本|跨越多个独立技术的强电流轨迹|持续增长,复合改善|
|环境长度扩展|强的电流轨迹|持续增长;可用性差距比原始最大长度更重要 |
|代理/工具使用培训|强的电流轨迹,高可见性|快速但不均的进展;新的失败模式与新的能力相结合 |
|欧盟/零售政策的通过 |强的电流轨迹|继续扩大最大规模;较小/更简单的使用情况仍然存在密集型号|

## 与其他文件的关系

- 根据本文件的预测,[06- 输入优化](../06-inference-optimization/), [mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md)其他[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md).
- 更多的投机方向,见[long-term-outlook.md](long-term-outlook.md).
- 对于这里所引用的具体可靠性/强度差距 (代理错误恢复,长文本可用性),请参见 [open-problems.md](open-problems.md).

## 来源

- 这档案是`[Projection]`根据本文, 报告的作者对本文库中记录的公共研究和行业趋势进行了合理的抽出 (特别是[06- 输入优化](../06-inference-optimization/), [03-深度学习架构mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md)其他[历史timeline-2023-present.md](../08-history/timeline-2023-present.md)),而不是单独的引用.
