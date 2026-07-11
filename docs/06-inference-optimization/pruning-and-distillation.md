# Pruning and Distillation

Pruning and distillation are two more approaches — alongside quantization (see [quantization.md](quantization.md)) — to making a trained model cheaper to run, by directly reducing how much of it there is (pruning) or by training a smaller model to imitate a larger one (distillation).

## Magnitude pruning

**Name & definition.** Magnitude pruning removes weights whose absolute value is small, on the reasoning that small-magnitude weights contribute comparatively little to the network's output and can be zeroed out (removed) with minimal impact on accuracy.

**Core mechanism.** After (or during) training, rank all weights by absolute magnitude, and set some fraction of the smallest ones to exactly zero — commonly followed by a period of further fine-tuning to let the remaining weights adjust and recover some of the accuracy lost from the removed connections. This can be repeated iteratively (prune a bit, fine-tune, prune more, fine-tune again) to reach higher sparsity levels (fraction of weights removed) with less accuracy loss than pruning everything in one shot.

**Why it mattered.** Magnitude pruning demonstrated, across many architectures, that neural networks are often substantially **over-parameterized** for a given task — a large fraction of weights (in some widely-reported cases, well over half) can be removed with little to no accuracy loss, suggesting much of a trained network's raw parameter count isn't strictly necessary for its learned function.

## Structured vs. unstructured pruning

**Unstructured pruning** removes individual weights wherever they happen to be smallest, without regard to any larger structural pattern — this can achieve high sparsity with minimal accuracy loss, but the resulting pattern of zeroed weights is irregular and scattered, which is a poor fit for how modern accelerator hardware (GPUs/TPUs) actually achieves speed: they're built for large, regular, dense matrix multiplications, and standard hardware generally can't skip individual scattered zero values to actually save computation time, only specialized sparse-computation hardware/kernels can, and even then usually only for specific sparsity patterns.

**Structured pruning** instead removes entire structural units — whole neurons, whole channels (in a CNN — see [cnn-family.md](../03-deep-learning-architectures/cnn-family.md)), or whole attention heads (in a Transformer — see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)) — rather than individual scattered weights. This produces a smaller, but still fully dense, regularly-shaped network (e.g., a layer with fewer neurons, rather than the same number of neurons with some connections zeroed out), which delivers real speedups on standard hardware without needing specialized sparse-computation support, at the cost of generally being less surgically precise than unstructured pruning (removing a whole neuron/channel/head is a coarser, more disruptive change than removing individually-selected weights) and thus typically incurring somewhat more accuracy loss for a comparable reduction in parameter count.

**The practical tradeoff.** This is the central practical tension in pruning: unstructured pruning achieves higher accuracy-per-parameter-removed on paper, but structured pruning is far more likely to actually translate into real-world speedups on commonly available hardware — which is why structured pruning tends to be favored in production deployment contexts, despite being theoretically "less efficient" in raw parameter-removal terms.

## Knowledge distillation

**Name & definition.** Knowledge distillation trains a smaller "student" model to reproduce the behavior of a larger, already-trained "teacher" model, rather than (or in addition to) training the student directly on the original hard labels.

**Origin.** Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network" (2015), building on related earlier "model compression" ideas (Bucilă, Caruana, Niculescu-Mizil, 2006).

**Core mechanism — the teacher-student framing.** Rather than training the student purely to match hard, one-hot correct labels, distillation trains the student to match the teacher's full output probability distribution over all classes/tokens (the "soft labels" or "soft targets") — often using a raised **temperature** in the softmax (see [autoregressive-generation.md](../04-generative-models/autoregressive-generation.md) for temperature's role in a different context — here it's applied during training rather than at sampling time) to soften the teacher's distribution and expose more information about its relative confidence across all the non-top choices, not just which single answer it considers correct.

```
Loss = α · CrossEntropy(student, true_label) + (1 − α) · CrossEntropy(student, teacher_soft_labels)
```

Walkthrough: the teacher's full output distribution encodes more information than just its single top prediction — for instance, a teacher classifying an image might assign most of its probability to "dog" but also meaningfully more probability to "wolf" than to "airplane," and that relative-similarity information (sometimes informally called "dark knowledge") is itself a useful training signal the student can learn from, beyond just "the correct answer was dog." Training the student to match this fuller distribution (blended with, or instead of, the original hard-label loss, weighted by α) lets the smaller student learn a more nuanced approximation of what the larger teacher has learned, often producing a smaller model that performs better than one trained the same size purely on the original hard labels alone.

**Why it mattered.** Distillation provides a general, task-agnostic recipe for compressing a large model into a smaller, cheaper one for deployment, without needing model-specific compression techniques, and has been widely applied both for general model compression and — notably — as one of several techniques reportedly used to train smaller, faster models that approximate the capabilities of larger frontier models at a fraction of the inference cost.

## Current practical usage in production model families

As of 2026, all three families covered in this section (quantization, pruning, and distillation) are commonly used **together**, not as mutually exclusive alternatives — a typical production deployment pipeline might distill a large model down to a smaller architecture, then quantize the resulting smaller model further for deployment, applying structured pruning as an additional step where hardware/latency constraints justify the accuracy tradeoff. Distillation specifically has become an especially prominent, widely-reported technique for producing smaller "fast" or "mini" variants of frontier model families, aimed at latency- and cost-sensitive use cases where the full-scale flagship model's capabilities aren't fully needed.

## Strengths & limitations

- **Pruning** — Strengths: can achieve high sparsity with modest accuracy loss (especially structured pruning, when done carefully); reduces both memory and (for structured pruning) real compute cost. Limitations: unstructured pruning often doesn't translate into real hardware speedups without specialized support; iterative prune-and-fine-tune cycles add engineering and compute overhead relative to a one-shot compression method.
- **Distillation** — Strengths: general-purpose, architecture-agnostic compression recipe; can produce a genuinely smaller, faster model rather than just a sparser version of the same architecture; the resulting student model is often more accurate than one trained from scratch at the same size on hard labels alone. Limitations: requires access to run the teacher model (and, ideally, to sample many teacher outputs) as part of student training, adding compute cost during the compression process itself; the student's capabilities are fundamentally bounded by what the teacher can teach it, and won't spontaneously exceed a well-chosen teacher's own capabilities.

## Comparison table

| Technique | Reduces | Real hardware speedup without special support? | Requires retraining/fine-tuning? |
|---|---|---|---|
| Unstructured magnitude pruning | Parameter count (sparse) | Generally no | Yes (recovery fine-tuning) |
| Structured pruning | Parameter count (dense, smaller) | Yes | Yes (recovery fine-tuning) |
| Knowledge distillation | Model size (new smaller architecture) | Yes (smaller model entirely) | Yes (full student training process) |

## Relationship to other algorithms

- Pruning and distillation are commonly combined with quantization (see [quantization.md](quantization.md)) in production compression pipelines.
- Distillation's soft-label/temperature mechanism is closely related to, but distinct in purpose from, the temperature used in autoregressive sampling — see [autoregressive-generation.md](../04-generative-models/autoregressive-generation.md).
- Distillation is also used, in a related but distinct sense, as a data-generation strategy (self-distillation) rather than purely a compression technique — see [curriculum-and-data-strategies.md](../05-training-methodology/curriculum-and-data-strategies.md).

## Sources

- Bucilă, Caruana, Niculescu-Mizil, "Model Compression" (2006)
- Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network" (2015)
- Han, Pool, Tran, Dally, "Learning both Weights and Connections for Efficient Neural Networks" (2015) [magnitude pruning]
