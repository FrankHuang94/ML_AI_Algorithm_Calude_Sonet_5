# Quantization

Quantization reduces the numerical precision used to represent a model's weights (and sometimes activations), shrinking memory footprint and often speeding up computation — at some cost in accuracy. This file covers why this matters specifically for inference, the mechanics of low-bit representations, the two major approaches to applying quantization, and the two most widely-used practical methods (GPTQ, AWQ).

## Why quantization matters for inference cost

**The baseline.** Neural network weights are typically stored and computed in 16-bit or 32-bit floating-point formats during training (higher precision helps numerically stable gradient-based optimization — see [optimization-algorithms.md](../01-foundations/optimization-algorithms.md)). But a trained model's weights don't need to stay in that format forever — inference (using an already-trained model to make predictions) has different, often more forgiving, numerical requirements than training does.

**Why this matters so much at LLM scale.** A model's memory footprint is dominated by its parameter count times the number of bytes used per parameter. A 70-billion-parameter model stored in 16-bit precision requires roughly 140 GB just for the weights — before accounting for the KV cache (see [kv-cache-and-attention-optimization.md](kv-cache-and-attention-optimization.md)) or any other runtime memory. Cutting the per-parameter storage in half or to a quarter (via 8-bit or 4-bit quantization, below) directly and proportionally cuts memory requirements, which matters enormously for serving cost: it determines how many accelerators (and how expensive an accelerator) are needed to hold and serve a given model, how many requests can be batched together (see [serving-and-batching.md](serving-and-batching.md)) within a fixed memory budget, and — since many LLM inference workloads are memory-bandwidth-bound rather than pure-compute-bound (moving weights from memory to the compute units is often the bottleneck, not the arithmetic itself) — smaller weight representations can directly translate into faster inference, not just cheaper storage.

## INT8/INT4 mechanics

**The core idea.** Instead of representing each weight as a 16- or 32-bit floating-point number, represent it as a low-bit integer (commonly 8-bit or 4-bit), together with a small amount of extra shared information (a **scale factor**, and sometimes a **zero-point** offset) that lets you convert back and forth between the compact integer representation and an approximate real-valued number.

**Core mechanism.** For a group of weights (e.g., all the weights in one row of a matrix, or some other chosen grouping), find the range of values present, and define a scale factor s such that:

```
quantized_value = round(real_value / s)
real_value_approx = quantized_value · s
```

Picture it as snapping each continuous weight to the nearest "rung" on a ladder with only a few rungs. INT4 gives you just 16 rungs to represent the whole range of values:

```
   Original weights (continuous):   -0.9   -0.3    0.0   0.25   0.7   0.95
                                       │      │      │     │      │     │
                                       ▼      ▼      ▼     ▼      ▼     ▼
   INT4 grid (16 evenly-spaced rungs across the range, here −1 … +1):
     -1.0  -0.87 ... -0.33 -0.20  0.0  0.20  0.33 ...  0.73  0.87  1.0
       │           │      │            │     │              │
   snapped:      -0.87  -0.33   0.0   0.20  0.73  0.87   ← each weight rounds to its nearest rung
   error:        (−0.03)(+0.03) (0)  (−0.05)(+0.03)(−0.08)  ← small rounding error per weight
```

Walkthrough: s is chosen so that the full range of real-valued weights maps onto the available integer range (e.g., −128 to 127 for signed 8-bit integers, or −8 to 7 for 4-bit). Multiplying by s and rounding compresses each weight into a small integer; multiplying that integer back by s recovers an approximation of the original value — not exact, since rounding to the nearest available integer loses some precision, but close enough that, chosen carefully, the model's overall behavior is largely preserved. The reason it works at all: a large model has hundreds of billions of weights, and the individual rounding errors are small and roughly independent, so they tend to average out rather than compound — the model as a whole is far more robust to a little noise on every weight than intuition suggests. INT8 (256 rungs) is almost always nearly lossless; INT4 (16 rungs) is where careful method choice starts to matter; below that, errors stop averaging out cleanly and accuracy degrades faster. The smaller the bit-width, the fewer distinct values are available to represent the same range, and the coarser (less precise) this approximation becomes — which is why going from 8-bit to 4-bit saves more memory but risks more accuracy loss, and why the specific choice of grouping (how many weights share one scale factor) and calibration procedure (how the range/scale is determined) matters a great deal to how much accuracy is actually lost in practice.

## Post-training quantization (PTQ) vs. quantization-aware training (QAT)

**Post-training quantization (PTQ).** Quantize an already-fully-trained model's weights after the fact, typically using a small calibration dataset (run some representative inputs through the model to observe realistic activation/weight ranges, informing good choices of scale factors) but without any further gradient-based training. This is fast and cheap to apply (no retraining needed) and is the standard choice for adapting an existing pretrained model for cheaper inference.

**Quantization-aware training (QAT).** Simulate the effects of quantization *during* training itself (typically by rounding weights/activations to their quantized representation in the forward pass, while still computing gradients as if the values were full-precision, allowing the model to adapt its own weights to compensate for the quantization noise it will experience at inference time). This generally produces better accuracy at very low bit-widths than PTQ, since the model has a chance to actually adjust to the precision loss during training rather than only having it imposed after the fact, but is considerably more expensive — it requires a full (or substantial partial) training run, not just a fast calibration pass, and isn't practical to apply to a model whose original training data/pipeline isn't available.

**Current status.** PTQ is by far the more commonly used approach for deploying existing large pretrained models cheaply, precisely because it doesn't require retraining; QAT is used more selectively, in settings where squeezing out maximum accuracy at very aggressive quantization levels is worth the added training cost.

## GPTQ

**Origin.** Frantar, Ashkboos, Hoefler, Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022).

**Core contribution.** GPTQ is a PTQ method that quantizes a model's weights one layer at a time, and — critically — quantizes the weights within each layer in a specific, carefully-chosen order, using second-order information (an approximation related to the Hessian — see [optimization-algorithms.md](../01-foundations/optimization-algorithms.md)) to compensate for the error introduced by quantizing each weight by adjusting the *not-yet-quantized* remaining weights in that layer, so that errors don't simply accumulate independently and uncorrected across the layer.

**Why it mattered.** GPTQ demonstrated that large language models could be quantized down to very low bit-widths (commonly 4-bit, sometimes lower) with comparatively little accuracy degradation and without any retraining, using a relatively fast, one-shot calibration procedure — making aggressive PTQ practical for large pretrained models at a scale where QAT would have been prohibitively expensive.

## AWQ (Activation-aware Weight Quantization)

**Origin.** Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023).

**Core contribution.** AWQ's key insight is that not all weights are equally important to preserve precisely — specifically, weights that interact with unusually large-magnitude activations have an outsized effect on the model's output, and protecting the precision of *those* specific weights (by identifying them via a calibration pass, and treating them with proportionally less aggressive quantization, e.g., by rescaling before quantizing) preserves much more of the original model's accuracy than treating every weight uniformly.

**Why it mattered.** AWQ achieves accuracy competitive with (and in some reported comparisons, better than) GPTQ, while being simpler and faster to run (it doesn't require GPTQ's more involved per-layer, order-dependent error-compensation process), making it a popular practical choice for quantizing large models for deployment.

## Accuracy/latency/memory tradeoff table

| Precision | Relative memory (vs. 16-bit) | Typical accuracy impact | Notes |
|---|---|---|---|
| 16-bit (FP16/BF16) | 100% (baseline) | None (standard training/inference precision) | Default for most training; common inference baseline |
| 8-bit (INT8) | ~50% | Small, often near-negligible with good calibration | Widely used, mature tooling |
| 4-bit (INT4, via GPTQ/AWQ) | ~25% | Small to moderate, method- and model-dependent | Standard for cost-sensitive large-model deployment as of 2026 |
| Sub-4-bit (experimental) | <25% | Larger, more variable, more sensitive to method quality | Active research area; less consistently production-ready |

## Relationship to other algorithms

- Quantization is frequently combined with LoRA (as QLoRA) for memory-efficient fine-tuning — see [finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md).
- Quantization directly reduces the memory footprint relevant to KV cache management and serving/batching decisions — see [kv-cache-and-attention-optimization.md](kv-cache-and-attention-optimization.md) and [serving-and-batching.md](serving-and-batching.md).
- Quantization is one of several complementary compression techniques alongside pruning and distillation — see [pruning-and-distillation.md](pruning-and-distillation.md) — that are commonly stacked together in production deployments.

## Sources

- Frantar, Ashkboos, Hoefler, Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022)
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023)
