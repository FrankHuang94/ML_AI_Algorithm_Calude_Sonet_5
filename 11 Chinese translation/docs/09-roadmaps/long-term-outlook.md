# 长期展望 (2-5年)

这份文件涵盖了比2026年中期更具投机性,更长视野的方向 (大约2-5年).[near-term-outlook.md](near-term-outlook.md)预测越远,仓库对其的信心就越低.`[Projection]`这份文件更依赖于"这是一个可信的方向和为什么"而不是"这是什么会发生".

## 替代规模关注的架构

**开放的问题.**国家空间模型 (见 [state-space-models.md](../03-deep-learning-architectures/state-space-models.md)) 和相关的线性回复架构将从2026年中期开始,与基于注意力的Transformer相比,仍然是一个少数的方法,主要是对于非常长的背景长度而具有竞争力,其中注意力的平方成本是最具惩罚性的.

**`[Projection]`**在2-5年时间内,该库认为可以相信 虽然真正不确定 混合架构 (结合大多数高效的SSM型层和少量的全注意层) 在边界模型中变得更常见,而不是纯粹的SSM完全取代注意力或注意力,完全不受挑战.`[Projection]`由于对工具成熟度,培训食谱和最大规模的经验记录的巨大领先性, 作为边界一般用途模型的主导架构, 基于注意力转换器的完全取代在这个窗口内似乎不太可能,

## 持续/在线学习方法

**现在的限制.**现在的边界模型在很大程度上在大型,分离的预训/后训周期中训练,然后使用基本固定权重. 它们不会像人类从经验中学习的方式不断更新自己的权重.[open-problems.md](open-problems.md)们在未来的未来,将会有多种影响.

**`[Projection]`**本备忘录认为,实践中持续/在线学习方法的有意义的进展是可行的发展,在2-5年内,这被明确的商业和实践价值驱动的模型可以在没有完整的重训周期的情况下纳入新信息,但本备忘录明确表示,到2026年中旬,这仍然是一个真正未解决的研究问题,而不是一个新兴的工程实践,并对迫在眉的解决方案的自信要求进行了真正的怀疑. `[Projection]`现在的进展似乎更有可能逐步 (更好的细节调整/基于PEFT的更新战略见[finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md)结合采集方法,避免对某些类型的新信息完全更新权重) 而不是通过单一决定性突破.

## 较为激进的测试时间计算规模化

**现在的轨迹.**关于"理性培训"的发展[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)其他[timeline-2023-present.md](../08-history/timeline-2023-present.md)已经证明,让模型在推理时间中花更多的计算 (在最终答案之前生成更长的中间推理链) 可以改善某些任务的性能,补充了涉及到的预训时间扩展法[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md).

**`[Projection]`**预计未来几年将继续探索这种"测试时间计算"轴作为与原材料预训规模不同的杆.*多少钱*更多的计算一个给定的问题要求 (在难题上花更多的思考,在容易的问题上花少的时间) 而不是一个固定的额外的推理应用均. `[Projection]`如何使其成为像预训计算一样可靠和理解的扩展轴 (与其自己的类似于[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)) 是本文写到时,本库认为一个真正开放的实验问题尚未解决.

## 关于此文件不确定性的方法说明

根据本库的基本规则 (见[scope-and-methodology.md](../00-overview/scope-and-methodology.md)据报道,该文件的任务是为未来几年提供可靠的,有信息的形状,而不是假装为确定性, 或者坦率地说,任何在2026年中期公开写论该领域的人都会有.

## 较量表

|方向|现状 (2026年中期) | `[Projection]`两到五年.|
|---|---|---|
|注意替代品 (SSM,混合物) |少数群体的做法,最为强烈的例子|混合动力设计的增长可观;完全的替换被认为是不可能的,但不是不可能的 |
|持续/在线学习|基本上还没有解决的研究问题|增长进展可行的; 完全解决方案不予以预期|
|测试时间计算规模|新兴,积极研究|持续增长作为一个明显的扩展轴; 完全的成熟度/可靠性不确定|

## 与其他文件的关系

- 持续学习与遗忘问题直接联系在一起[open-problems.md](open-problems.md).
- 关注的替代品在当前的现状中被涵盖[state-space-models.md](../03-deep-learning-architectures/state-space-models.md)文件明确地建立了前进的基础.
- 测试时间计算规模是RL为推理方向的直接向前延长[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md).

## 来源

- 这档案是`[Projection]`根据本文的原始版本, 索赔是作者自己的合理抽象, 显然与源头的现有索赔区分开来.[state-space-models.md](../03-deep-learning-architectures/state-space-models.md), [finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md)其他[rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)这会激励他们.
