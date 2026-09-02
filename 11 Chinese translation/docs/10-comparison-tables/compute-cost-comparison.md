# 计算成本比较

本文件特别关注了本库所涵盖的技术的培训和推理成本交易 仅仅是合成,没有新的算法.[algorithm-comparison-master.md](algorithm-comparison-master.md)根据各类别的"计算资料"列,该文件深入了解具体的成本杆以及它们如何堆叠.

## 培训成本杆

|杆|交易所的价值|覆盖在|
|---|---|---|
|模型大小 (参数) |更多容量与每步的计算/记忆量| [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)根据"化法"的规定,|
|数据大小 (代码) |较好的通用化与较大的总计算 (参数 × 词元) | [pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md) |
|批量大小|提高硬件利用/平行性与优化器稳定性挑战 | [optimization-algorithms.md](../01-foundations/optimization-algorithms.md)对于这些问题,我们需要注意.|
|并行性战略 (数据/ensor/管道/ZeRO) |适合较大的模型/批量与通信通用费用| [distributed-training.md](../05-training-methodology/distributed-training.md) |
|完全调整与PEFT|任务性能最大,计算/记忆力显著下降| [finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md) |
|利与利|可重复使用的奖励模型 + 细粒度的RL控制与简单性/稳定性 | [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md) |
|摩埃与密集建筑|总容量较增加路由/服务复杂性| [mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md) |

**智尔时代的制 (见[transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)其他[pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md)):**对于固定训练计算预算,模型大小和数据大小之间的分比本身是一种成本决定,而不是自由变量 训练一个较小的模型在更多数据可以匹配或击败一个较大的模型训练在比例较少的数据,同样的总计算成本,并另外产生一个更便宜的模型*服务*因此,自2022年以来的现实世界预训决策往往将数据量推向了狭窄的计算最佳点,以更低的长期预训成本为代价.

## 推进成本杆

|杆|交易所的价值|覆盖在|
|---|---|---|
|量化 (位宽) |较低的内存/较快的推理与准确性损失 (较低的位宽度更大) | [quantization.md](../06-inference-optimization/quantization.md) |
|切割/蒸|较小/更快的模型与前期压缩成本+由原型/教师模型设置的能力上限 | [pruning-and-distillation.md](../06-inference-optimization/pruning-and-distillation.md) |
|管理KV缓存 (PagedAttention,GQA/MQA) |较高的可实现的批量同步性与某些质量交易 (GQA/MQA) 或工程复杂性 (页面化) | [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md) |
|投机解码|较低成本的连续模型调用与草案模型总费/复杂性| [speculative-decoding.md](../06-inference-optimization/speculative-decoding.md) |
|批量策略 (静态对连续) |吞吐量与单个请求延迟| [serving-and-batching.md](../06-inference-optimization/serving-and-batching.md) |
|文本长度|更多可用输入与方形注意力成本和更大的KV缓存| [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md), [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md) |

**值得清晰地提及的一个显著的不对称:**由于它们不寻常,它们在自己的文件中包含了两个显著的例外:闪存注意力 (完全相同的输出,只是更快/更有效的存储器查看[kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md)) 和投机解码 (完全相同的输出分布,只是更快查看[speculative-decoding.md](../06-inference-optimization/speculative-decoding.md)它们的应用与"免费午餐"相近,这正是为什么它们在技术方面得到了如此广泛,快速,无争议的采用,

## 实际上,这些杆是如何堆叠的

实际的生产LLM服务堆,[06- 输入优化](../06-inference-optimization/)总体上,它同时结合了几个,而不是只选择一个:一个由Chinchilla (或超越Chinchilla,服务成本意识) 的数据扩展训练的模型 → 定量化到4-8位用于部署 → 与GQA (在训练前的建筑选择) 服务 → 使用PagedAttention式KV缓存管理 → 连续批量 → 并且越来越多地,上面设置了投机解码.每个层都解决了总存储成本方程的不同部分 (模型大小,足迹,批量共存,连续调用数量),以及该仓库的覆盖范围是vLLM和TensorRT-LLM (见[serving-and-batching.md](../06-inference-optimization/serving-and-batching.md)它们是完全像这种结合式堆,而不是单一的银子弹技术.

## 培训成本与推理成本:真正的紧张性

**交易,明确表示.**在培训时间中花更多的计算 (更多数据,根据上述的奇拉时代逻辑,或像QAT这样的技术而不是更便宜的PTQ查看[quantization.md](../06-inference-optimization/quantization.md)由于部署的模型可以在长期内为大量用户提供服务,这种为服务的培训-节省交易价值可以相当的额外的提前培训计算[pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md)其他[quantization.md](../06-inference-optimization/quantization.md)经济现实,也就是在边界模式发展决策中,

## 与其他文件的关系

- [algorithm-comparison-master.md](algorithm-comparison-master.md)根据此文件的具体成本,
- [when-to-use-what.md](when-to-use-what.md)转化这些成本抵消成具体的决策指导.
