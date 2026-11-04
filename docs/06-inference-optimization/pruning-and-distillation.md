# Pruning and Distillation

Pruning and distillation are two more approaches — alongside quantization (see [quantization.md](quantization.md)) — to making a trained model cheaper to run, by directly reducing how much of it there is (pruning) or by training a smaller model to imitate a larger one (distillation). For LLMs, distillation has also grown a second role beyond compression: **on-policy distillation**, covered below, has become a core post-training tool for merging the skills of several specialist models into one.

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

## On-policy distillation

**Name & definition.** On-policy distillation trains a student language model on text *the student itself generates*: the student samples its own responses, and a teacher model grades every token of those responses by supplying its own probability distribution for that position, which the student is trained to match.

**Origin.** Agarwal et al., "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes" (GKD, "Generalized Knowledge Distillation," 2023; ICLR 2024); Gu et al., "MiniLLM" (2023) also trained students with reverse KL on their own samples. A 2025 Thinking Machines Lab blog post popularized the technique and its framing as a cheap alternative to RL for post-training.

**The problem with off-policy distillation.** The classic way to distill an LLM is **off-policy**: have the teacher write lots of example outputs (or score a fixed dataset), and fine-tune the student to imitate them. A prominent example is DeepSeek-R1 (Jan 2025), whose reasoning ability was distilled into several smaller dense models by fine-tuning them on reasoning traces generated by R1. This works, but it has a structural flaw: the student only ever trains on the *teacher's* trajectories. At inference it generates its own, and as soon as it makes a mistake the teacher would never have made, it is in territory it has never seen in training and has no practice recovering from — so errors compound (the same "exposure bias" that afflicts any imitation of expert demonstrations).

**Core mechanism.** On-policy distillation flips who writes the training text:

```
repeat:
    prompt x  ← sample from training prompts
    y         ← student generates a full response to x              (on-policy: the student's own text)
    for each position t in y:
        teacher_probs_t ← teacher's next-token distribution given (x, y_<t)   (one teacher forward pass, no generation)
        student_probs_t ← student's next-token distribution given (x, y_<t)
    loss = Σ_t  KL( student_probs_t ‖ teacher_probs_t )               (reverse KL, per token)
    update student to reduce loss
```

Walkthrough: KL divergence measures how different two probability distributions are; **reverse KL** `KL(student ‖ teacher)` penalizes the student for putting probability on tokens the teacher considers unlikely. It is "mode-seeking": a small student that can't reproduce everything the teacher does is pushed to commit to the teacher's most likely behaviors rather than spreading itself thin — a good fit for compressing a big model into a small one. (Standard distillation uses forward KL, `KL(teacher ‖ student)`, which instead pushes the student to cover *all* of the teacher's options.) Because the student is graded on its own trajectories, it learns what the teacher would do *from the states the student actually gets itself into*, including after its own mistakes. GKD generalizes this, allowing a mix of student-generated and fixed data and a choice of divergences.

**The connection to RL.** This looks a lot like reinforcement learning (see [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)): the student samples its own outputs and is updated based on how good they are. The difference is the reward. RL with verifiable rewards gives one sparse signal per whole response ("was the final answer right?"); on-policy distillation gives a **dense, per-token** signal — essentially, at every step, "how much does the teacher agree with this token?" That makes each sample far more informative, which is why it is often much cheaper than RL for reaching a capability a teacher already has. The trade-off is that the student can't exceed the teacher in the way RL can exceed its starting point; and per-token targets require the teacher and student to share a tokenizer (or some mapping between the two).

**Multi-teacher on-policy distillation — the 2026 post-training consolidation pattern.** In 2026, a widely reported frontier recipe uses on-policy distillation not to shrink a model but to *merge* skills:

1. Train several **domain-specialist** models with RL — for example, one each for math, code, agentic tool use, and general chat.
2. Distill them all into one student: the student samples its own response to a prompt from some domain, and the matching specialist teacher supplies per-token targets (reverse KL on the student's own trajectories).

```
   specialist RL runs:   [math teacher]  [code teacher]  [agentic teacher]  [general teacher]
                                │               │                │                 │
   student answers a code prompt ──► code teacher scores every token ──► reverse-KL update
   student answers a math prompt ──► math teacher scores every token ──► reverse-KL update
                                     ...one student absorbs all specialties
```

The motivation is interference: putting every domain into a single mixed RL run tends to let capabilities trade off against each other, while separate specialist runs can each be tuned on their own. Reported examples include NVIDIA's Nemotron 3 Ultra, distilled from more than ten teachers, and staged pipelines such as GLM-5's (SFT → reasoning RL → agentic RL → general RL, with distillation across stages); Lambert & Timbers' June 2026 survey of post-training practice describes multi-teacher on-policy distillation as the frontier pattern. Note that here the teachers may be the same size as the student — distillation is acting as a *consolidation* tool rather than a compression tool. Details of these industrial pipelines come from the labs' own reports and have not been independently reproduced.

**Current status (late 2026).** Established in post-training of frontier and open-weight models; off-policy distillation (fine-tuning on teacher-generated data, as with the R1 distilled models) remains common for producing small models cheaply, because once the teacher's data has been generated, training the student needs no further teacher calls.

## Current practical usage in production model families

As of late 2026, all three families covered in this section (quantization, pruning, and distillation) are commonly used **together**, not as mutually exclusive alternatives — a typical production deployment pipeline might distill a large model down to a smaller architecture, then quantize the resulting smaller model further for deployment, applying structured pruning as an additional step where hardware/latency constraints justify the accuracy tradeoff. Distillation specifically has become an especially prominent, widely-reported technique for producing smaller "fast" or "mini" variants of frontier model families, aimed at latency- and cost-sensitive use cases where the full-scale flagship model's capabilities aren't fully needed.

## Strengths & limitations

- **Pruning** — Strengths: can achieve high sparsity with modest accuracy loss (especially structured pruning, when done carefully); reduces both memory and (for structured pruning) real compute cost. Limitations: unstructured pruning often doesn't translate into real hardware speedups without specialized support; iterative prune-and-fine-tune cycles add engineering and compute overhead relative to a one-shot compression method.
- **On-policy distillation** — Strengths: avoids the train/inference mismatch of imitating teacher-written text; dense per-token signal makes it much more sample-efficient than RL for reaching a teacher's capability; a clean way to merge several specialist models. Limitations: needs a teacher forward pass on every student sample during training; per-token targets need a shared (or mapped) tokenizer; the student is bounded by its teachers.
- **Distillation** — Strengths: general-purpose, architecture-agnostic compression recipe; can produce a genuinely smaller, faster model rather than just a sparser version of the same architecture; the resulting student model is often more accurate than one trained from scratch at the same size on hard labels alone. Limitations: requires access to run the teacher model (and, ideally, to sample many teacher outputs) as part of student training, adding compute cost during the compression process itself; the student's capabilities are fundamentally bounded by what the teacher can teach it, and won't spontaneously exceed a well-chosen teacher's own capabilities.

## Comparison table

| Technique | Reduces | Real hardware speedup without special support? | Requires retraining/fine-tuning? |
|---|---|---|---|
| Unstructured magnitude pruning | Parameter count (sparse) | Generally no | Yes (recovery fine-tuning) |
| Structured pruning | Parameter count (dense, smaller) | Yes | Yes (recovery fine-tuning) |
| Knowledge distillation | Model size (new smaller architecture) | Yes (smaller model entirely) | Yes (full student training process) |
| On-policy distillation (LLMs) | Model size, or number of separate specialist models (consolidation) | Yes (if the student is smaller) | Yes (student samples, teacher scores each token) |

## Relationship to other algorithms

- Pruning and distillation are commonly combined with quantization (see [quantization.md](quantization.md)) in production compression pipelines.
- Distillation's soft-label/temperature mechanism is closely related to, but distinct in purpose from, the temperature used in autoregressive sampling — see [autoregressive-generation.md](../04-generative-models/autoregressive-generation.md).
- Distillation is also used, in a related but distinct sense, as a data-generation strategy (self-distillation) rather than purely a compression technique — see [curriculum-and-data-strategies.md](../05-training-methodology/curriculum-and-data-strategies.md).
- On-policy distillation is closely related to RL for LLMs — same "sample from the student, then update" loop, but with a dense per-token teacher signal instead of a sparse reward — see [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md); the specialist teachers in multi-teacher distillation are themselves usually trained with RL, and the overall post-training pipeline is described in [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md).

## Sources

- Bucilă, Caruana, Niculescu-Mizil, "Model Compression" (2006)
- Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural Network" (2015)
- Han, Pool, Tran, Dally, "Learning both Weights and Connections for Efficient Neural Networks" (2015) [magnitude pruning]
- Kim, Rush, "Sequence-Level Knowledge Distillation" (2016) [distilling on teacher-generated sequences]
- Agarwal et al., "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes" (GKD; 2023, ICLR 2024)
- Gu et al., "MiniLLM: Knowledge Distillation of Large Language Models" (2023) [reverse-KL distillation on student samples]
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025) [distilled small dense models — an off-policy example]
- Thinking Machines Lab, "On-Policy Distillation" (blog post, 2025)
- Lambert & Timbers, survey of post-training practice (June 2026) [multi-teacher on-policy distillation as the 2026 frontier pattern]
- NVIDIA, Nemotron 3 Ultra (2026) and Zhipu AI, GLM-5 (2026) technical reports [multi-teacher / cross-stage distillation; reported]
