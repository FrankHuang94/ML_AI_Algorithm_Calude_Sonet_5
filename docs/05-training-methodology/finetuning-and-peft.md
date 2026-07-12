# Fine-Tuning and Parameter-Efficient Fine-Tuning

Fine-tuning adapts a pretrained model (see [pretraining-strategies.md](pretraining-strategies.md)) to a specific task, domain, or behavior, using a much smaller, more targeted dataset than pretraining used. This file covers full fine-tuning, why it became impractical at LLM scale, and the parameter-efficient fine-tuning (PEFT) methods — especially LoRA and QLoRA — that emerged to address that impracticality.

## Full fine-tuning

**Name & definition.** Full fine-tuning updates every parameter of a pretrained model, using the same kind of gradient-based training as pretraining (see [optimization-algorithms.md](../01-foundations/optimization-algorithms.md)), just on a smaller, task-specific dataset and usually for far fewer steps.

**Why it works.** A pretrained model already has good general representations from pretraining; fine-tuning nudges those representations and the model's output behavior toward a specific task without needing to relearn everything from scratch, since the pretrained weights are a much better starting point than random initialization.

**Why it became impractical at LLM scale.** Full fine-tuning requires storing and updating gradients and optimizer state (recall, from [optimization-algorithms.md](../01-foundations/optimization-algorithms.md), that Adam/AdamW keep two additional numbers — the first and second moment estimates — per parameter) for *every* parameter in the model. For a model with tens or hundreds of billions of parameters, this means the memory required for fine-tuning is a large multiple of the model's own size, often making full fine-tuning infeasible on anything short of the same large-scale hardware clusters used for pretraining itself. It also means that if you want many different fine-tuned variants of the same base model (e.g., one per customer, or one per task), you need to store a full separate copy of the entire model's weights for each variant — an increasingly untenable storage and serving cost as base models grow.

## The motivation for parameter-efficient fine-tuning

**The core idea.** Instead of updating all of a model's parameters, freeze the pretrained weights entirely and train only a small number of additional or modified parameters — a much smaller memory footprint for training, and (critically) each fine-tuned "variant" can be represented as just that small set of additional parameters, layered on top of one shared, frozen copy of the base model, rather than requiring its own full model copy.

## LoRA (Low-Rank Adaptation)

**Origin.** Hu et al. (Microsoft), "LoRA: Low-Rank Adaptation of Large Language Models" (2021).

**Core mechanism — low-rank decomposition, explained clearly.** For a given weight matrix W in the pretrained model (say, a d×d matrix inside an attention or feedforward layer), LoRA freezes W entirely and instead learns an additive update ΔW, but constrains ΔW to be expressible as the product of two much smaller matrices:

```
ΔW = B · A         where B is d×r, A is r×d, and r ≪ d
h = W·x + ΔW·x = W·x + B·(A·x)
```

```
   Full fine-tuning: update the whole      LoRA: freeze W, learn a thin B·A instead
   d×d matrix (d² parameters)              (2·d·r parameters, r ≪ d)

        ┌───────────────┐                     ┌───────────────┐    ┌─┐
        │               │                     │               │    │ │ B  (d×r,
        │   W  (train    │                     │   W  (FROZEN) │  + │ │     tall & thin)
        │   all of it)   │                     │               │    │ │
        │               │                     │               │    └─┘
        └───────────────┘                     └───────────────┘    ┌──────┐
         d² numbers to learn                  0 new here            └──────┘ A (r×d, short & wide)
         (e.g. 4096×4096                                            2·d·r numbers
          ≈ 16.7M per matrix)                                       (e.g. 4096×8×2 ≈ 66K — ~250× fewer)
```

Walkthrough: instead of learning a full d×d update matrix (which would have d² parameters), LoRA learns two much thinner matrices A and B, whose product approximates the update — this has only 2·d·r parameters, where r (the "rank" of the decomposition — typically a small number like 4, 8, or 16, versus d values that can be in the thousands) is chosen to be far smaller than d. The numbers in the sketch make the savings vivid: for a 4096×4096 matrix, full fine-tuning learns ~16.7M numbers; LoRA with rank 8 learns ~66K — roughly 250× fewer — per matrix. This is the "low-rank" in the name: it assumes the *useful* update to a pretrained weight matrix for a given task doesn't need to explore the full space of possible d×d changes, but can be well-approximated by a much lower-dimensional (rank-r) update — an assumption that has held up well empirically across a wide range of fine-tuning tasks. At inference time, B·A can even be computed once and added directly to W, meaning a LoRA-adapted model can run with zero extra latency compared to the original model, if desired — or B·A can be kept separate and swapped out per task/customer, using the same frozen base weights W for many different LoRA adapters simultaneously.

**Why it mattered.** LoRA reduces the number of trainable parameters (and the optimizer state needed for them) by orders of magnitude relative to full fine-tuning — often well under 1% of the base model's parameter count — while empirically matching full fine-tuning's task performance closely in a wide range of settings. This made fine-tuning large models practical on dramatically less hardware, and made it practical to maintain many different task/customer-specific fine-tunes cheaply (as small LoRA adapter files layered on one shared base model) rather than many full model copies.

## QLoRA (quantization + LoRA)

**Origin.** Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023).

**Core mechanism.** QLoRA combines LoRA with quantization (representing the frozen base model's weights using low-precision number formats — see [quantization.md](../06-inference-optimization/quantization.md) for the full mechanics — commonly 4-bit precision in QLoRA's case) — the frozen base weights are stored and used in quantized (compressed) form during fine-tuning, while the small LoRA adapter matrices A and B are still trained in normal higher precision. The paper also introduced supporting techniques (a specific 4-bit data type tuned for the typical distribution of neural network weights, and careful memory management for optimizer states) to keep this combination numerically stable despite the aggressive compression of the frozen base model.

**Why it mattered.** By quantizing the (much larger) frozen base model while keeping the (much smaller) trainable LoRA parameters at full precision, QLoRA cut the memory required to fine-tune large models even further than LoRA alone — the paper's headline demonstration was fine-tuning a large model on a single consumer-grade GPU, a scale of accessibility that would have been unthinkable with full fine-tuning.

### The broader LoRA family (DoRA and other refinements)

LoRA has spawned a family of refinements worth being aware of, though the core idea is unchanged. **DoRA** (Weight-Decomposed Low-Rank Adaptation, 2024) splits each weight into a magnitude and a direction and applies a LoRA-style low-rank update only to the direction, which has been reported to close some of the residual accuracy gap between LoRA and full fine-tuning at similar parameter cost. Other variants adjust *where* rank is spent (allocating more rank to layers that need it) or *how* the low-rank matrices are initialized. The practical takeaway for a reader: LoRA is not a single frozen technique but an actively refined family, and "use LoRA" in 2026 often means "use LoRA or one of its close descendants," with the plain original still a perfectly strong default.

## Adapters

**Name & definition.** Adapters insert small, newly-initialized neural network layers (typically a bottleneck: down-project to a small dimension, apply a nonlinearity, project back up) directly into a frozen pretrained model's architecture — commonly after the attention and feedforward sub-layers — training only these newly-inserted layers.

**Origin.** Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (2019) — predates LoRA and was one of the earliest PEFT approaches for Transformers.

**Tradeoffs relative to LoRA.** Adapters add extra sequential computation at inference time (the data has to pass through the new bottleneck layers, unlike LoRA's update, which can be merged directly into existing weight matrices with no added inference-time computation), which is the main reason LoRA-style approaches have become more popular for latency-sensitive deployment, even though adapters remain a reasonable, well-established PEFT choice.

## Prefix tuning and prompt tuning

**Prefix tuning** (Li, Liang, 2021) prepends a sequence of learned, continuous ("soft") vectors to the input at every layer of the model (not actual words — trainable vectors that don't correspond to any real token), training only these prefix vectors while keeping the rest of the model frozen; the model attends to these learned prefixes as extra context that steers its behavior toward the fine-tuned task.

**Prompt tuning** (Lester, Al-Rfou, Constant, 2021) is a simplification that adds learned soft-prompt vectors only at the input layer (not injected at every layer as in prefix tuning), trading a little expressiveness for even fewer trainable parameters and a simpler implementation.

**Tradeoffs.** Both methods are extremely parameter-efficient (often even more so than LoRA, in raw trainable-parameter count) but have generally shown somewhat more inconsistent performance across tasks and model scales compared to LoRA-style approaches, and — like adapters — add some sequential computation at inference time (extra "tokens" for the prefix/prompt vectors to attend over) rather than merging cleanly into existing weights.

## Comparison table

| Method | Trainable parameters (relative to full fine-tuning) | Adds inference latency? | Notes |
|---|---|---|---|
| Full fine-tuning | 100% | No | Highest memory/compute cost; best-case task performance ceiling |
| LoRA | Often <1% | No (can be merged into base weights) | Standard modern default; multiple adapters can share one frozen base model |
| QLoRA | Often <1% (plus quantized frozen base) | No (adapter mergeable; base model runs quantized) | Lowest fine-tuning memory footprint; enables consumer-hardware fine-tuning |
| Adapters | Small (a few %) | Yes (extra layers in the forward pass) | Early, well-established PEFT approach |
| Prefix tuning | Very small | Yes (extra context to attend over) | More expressive than prompt tuning; less consistent than LoRA |
| Prompt tuning | Smallest | Yes (extra input tokens) | Simplest; most parameter-efficient; more variable performance |

## Current status

LoRA (and QLoRA where fine-tuning memory is especially constrained) is the standard default parameter-efficient fine-tuning approach as of 2026, given its combination of strong empirical performance, zero added inference latency when merged, and the practical convenience of maintaining many lightweight task-specific adapters against one shared frozen base model. Full fine-tuning remains used when maximum task performance justifies its cost, or for the pretraining-adjacent large-scale post-training stages some labs run on their own base models before any PEFT-based customization layer is applied downstream.

## Relationship to other algorithms

- PEFT methods all build on top of a pretrained base model — see [pretraining-strategies.md](pretraining-strategies.md).
- QLoRA's quantization component is covered in full mechanical depth in [quantization.md](../06-inference-optimization/quantization.md).
- LoRA and its relatives are frequently used as the mechanism for applying RLHF/DPO-style post-training updates efficiently — see [rlhf-and-alignment.md](rlhf-and-alignment.md).
- Optimizer state memory costs, the underlying reason full fine-tuning is so expensive, are covered in [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).

## Sources

- Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (2019) [Adapters]
- Li, Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation" (2021)
- Lester, Al-Rfou, Constant, "The Power of Scale for Parameter-Efficient Prompt Tuning" (2021)
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Dettmers, Pagnoni, Holtzman, Zettlemoyer, "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
