# Quantization

Quantization reduces the numerical precision used to represent a model's weights (and sometimes activations), shrinking memory footprint and often speeding up computation — at some cost in accuracy. This file covers why this matters specifically for inference, the mechanics of low-bit integer representations, the two major approaches to applying quantization, the two most widely-used practical integer methods (GPTQ, AWQ), and the low-precision *floating-point* formats (FP8, and the 4-bit microscaling formats MXFP4 and NVFP4) that moved during 2024–2026 from research into official model releases and even into pretraining.

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

## Low-precision floating-point formats: FP8, MXFP4, NVFP4

Everything above uses *integers* plus a scale. A second family keeps tiny *floating-point* numbers instead, and since 2024 it has become the main way frontier models are trained and shipped at low precision.

**Why floating point at low bit-widths.** A floating-point number spends some bits on an **exponent** (which power of two the value is near) and some on a **mantissa** (the fine position between consecutive powers of two), plus one sign bit. Formats are named by that split: **E4M3** means 4 exponent bits and 3 mantissa bits; **E5M2** trades a mantissa bit for more exponent (more range, less precision); **E2M1** is a 4-bit float. Unlike an integer grid, whose rungs are evenly spaced, a float grid is dense near zero and sparse far from it — which matches the bell-shaped, outlier-prone distributions of weights, activations, and gradients better: most values are small and get fine resolution, while the rare big values can still be represented.

```
 Magnitudes representable by a 4-bit format (positive half; sign bit gives the negatives):

 INT4 (evenly spaced):    0   1   2   3   4   5   6   7
 FP4 E2M1 (float):        0  0.5  1  1.5  2   3   4   6      ← fine steps near 0, coarse steps far out

 Either grid is then multiplied by a scale factor, exactly as in the INT8/INT4 section above.
```

### FP8 training (DeepSeek-V3)

**What it is.** Running the large matrix multiplications of *training* (not just inference) in 8-bit floating point (E4M3 and/or E5M2), while keeping the master copy of the weights, the optimizer state (see [optimization-algorithms.md](../01-foundations/optimization-algorithms.md)), and numerically sensitive operations in higher precision — "mixed-precision" training pushed one step further than the 16-bit BF16 standard.

**Origin.** The FP8 formats were proposed by Micikevicius et al., "FP8 Formats for Deep Learning" (2022), and NVIDIA's Hopper-generation GPUs added FP8 matrix hardware. DeepSeek-V3 (Dec 2024; 671B total / 37B active parameters, Mixture-of-Experts) was the landmark public demonstration that FP8 mixed-precision training works at frontier scale.

**Core mechanism (what makes it work).** The main danger is that 8 bits leave very little room, so one outlier value can force a coarse scale onto everything sharing its scale factor. DeepSeek-V3's recipe uses **fine-grained scaling** — separate scale factors for small tiles/blocks of each matrix rather than one per tensor, so an outlier only hurts its own small block — and accumulates matrix-multiply partial sums in higher precision so that rounding errors don't pile up across long dot products.

**Why it matters for inference.** A model trained natively in FP8 can be served in FP8 with no separate post-training quantization step, which blurs the PTQ/QAT distinction above: the model has, in effect, already adapted to low precision during training.

### Microscaling formats: MXFP4 vs. NVFP4

**The idea.** At 4 bits there is so little resolution that a single scale per tensor (or even per row) is far too coarse. **Microscaling** gives every small block of consecutive values its own shared scale factor — the same group-wise scaling idea used by GPTQ/AWQ above, but standardized as a number format that hardware understands directly. The Open Compute Project's Microscaling (MX) specification (2023), backed by several chip and cloud companies, defined the MX family; NVIDIA defined its own variant, NVFP4, supported natively by its Blackwell-generation GPUs.

```
 MXFP4 — block of 32 values, one power-of-two scale
   scale: E8M0  (8-bit exponent only → scale ∈ {…, 0.25, 0.5, 1, 2, 4, …})
   [v1][v2][v3] ... [v32]     each v = FP4 E2M1
   storage ≈ 4 + 8/32 = 4.25 bits per value

 NVFP4 — block of 16 values, one FP8 scale, plus one FP32 scale for the whole tensor
   scale: FP8 E4M3  (can be 1.375, 2.75, … — not just powers of two)
   [v1][v2] ... [v16]          each v = FP4 E2M1
   storage ≈ 4 + 8/16 = 4.5 bits per value  (+ one FP32 number per tensor, negligible)

 decoded value = v × block_scale            (MXFP4)
 decoded value = v × block_scale × tensor_scale   (NVFP4)
```

Walkthrough: both formats store each value as the same 4-bit float; they differ only in how the shared scale is chosen. MXFP4's scale must be a power of two, so the block's largest value generally can't be mapped exactly onto the top of the grid — it either falls short (leaving part of the already-tiny 4-bit grid unused) or overshoots and gets clipped. NVFP4's scale is itself a small float, so it can stretch the grid to fit each block more tightly; because an E4M3 scale has limited range on its own, a per-tensor FP32 scale handles the overall magnitude. NVFP4 also uses smaller blocks (16 vs. 32), so an outlier distorts fewer neighbors. The price is slightly more storage (4.5 vs. 4.25 bits per value) and a format tied to one vendor's hardware.

### gpt-oss: an official release shipped in MXFP4

OpenAI's open-weight gpt-oss models (Aug 2025; gpt-oss-120b and gpt-oss-20b, both Mixture-of-Experts) shipped with their MoE weights natively in MXFP4. Two things make this notable. First, the 4-bit weights *are* the official checkpoint — not a community quantization made after release. Second, it targets exactly where the memory is: in an MoE model the expert weights are the overwhelming majority of parameters (see [mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md)), so quantizing them captures most of the savings while the smaller remaining parts can stay at higher precision. Back-of-envelope: roughly 120 billion parameters at ~4.25 bits is about 64 GB, which is why the larger model can be served from a single 80 GB-class accelerator rather than several.

### Pretraining in 4 bits: NVFP4 (NVIDIA, 2025–2026)

**What was shown.** NVIDIA's "Pretraining Large Language Models with NVFP4" (arXiv 2509.25149, Sep 2025; revised Mar 2026) trained a 12B-parameter hybrid Mamba-Transformer model on 10 trillion tokens with NVFP4 matrix multiplications, reaching 62.58% on the MMLU-Pro benchmark versus 62.62% for an FP8-trained baseline — essentially matching it.

**The techniques that made it work**, each addressing a specific way 4-bit training goes wrong:
- **Random Hadamard transforms.** Before quantizing, multiply blocks of values by a randomized orthogonal matrix of ±1 entries (and undo it afterward — because the transform is orthogonal, it can be cancelled out inside the matrix multiplication). This spreads a single large outlier across many values, so no block's scale is dominated by one spike.
- **2-D block scaling.** Weights are scaled in square tiles rather than 1-D strips, so a weight gets the same quantized value whether the matrix is read row-wise (forward pass) or column-wise (the transposed use in the backward pass); otherwise the forward and backward passes would see slightly different weights.
- **Stochastic rounding on gradients.** Instead of always rounding to the nearest grid point, round up or down at random with probability proportional to closeness. Each rounding is noisier, but it is *unbiased* on average, so systematic rounding errors don't accumulate over millions of updates.
- **A few sensitive layers kept in higher precision**, and **switching to higher precision late in training**, when updates become small and precision matters most.

**Status.** A research result at 12B scale. It shows 4-bit training is *possible* with care; whether it becomes routine for frontier-scale pretraining had not been publicly established as of late 2026.

## Accuracy/latency/memory tradeoff table

| Precision | Relative memory (vs. 16-bit) | Typical accuracy impact | Notes |
|---|---|---|---|
| 16-bit (FP16/BF16) | 100% (baseline) | None (standard training/inference precision) | Default for most training; common inference baseline |
| 8-bit integer (INT8) | ~50% | Small, often near-negligible with good calibration | Widely used, mature tooling |
| 8-bit float (FP8, E4M3/E5M2) | ~50% | Small; used for training as well as inference | Mainstream on recent accelerators; DeepSeek-V3 showed FP8 training at frontier scale |
| 4-bit integer (INT4, via GPTQ/AWQ) | ~25% (plus small per-group scale overhead) | Small to moderate, method- and model-dependent | Standard for community/cost-sensitive deployment of open models as of late 2026 |
| MXFP4 (FP4, 32-value blocks, power-of-two scale) | ~27% (4.25 bits) | Small to moderate; best when the model was trained/post-trained for it | Open standard; gpt-oss ships its MoE weights in it |
| NVFP4 (FP4, 16-value blocks, FP8 scale + tensor scale) | ~28% (4.5 bits) | Designed for lower error than MXFP4 at the same bit-width | NVIDIA-specific; demonstrated for 12B-scale pretraining (2025) |
| Sub-4-bit (experimental) | <25% | Larger, more variable, more sensitive to method quality | Active research area; less consistently production-ready |

## Where things stand (late 2026)

- **16-bit (BF16) is no longer the floor for training at the frontier.** FP8 mixed precision, demonstrated at scale by DeepSeek-V3, is a widely reported choice for large training runs, and the resulting models can be served in FP8 directly.
- **4-bit floating point has moved into official releases.** gpt-oss shipped MXFP4 weights; hardware with native FP4 support makes 4-bit inference faster, not just smaller.
- **4-bit training is a research frontier**, with NVIDIA's NVFP4 result the main public data point.
- **Integer PTQ (GPTQ, AWQ) remains the everyday tool** for shrinking existing open models after release, especially on hardware without native FP4/FP8 support.

## Relationship to other algorithms

- Quantization is frequently combined with LoRA (as QLoRA) for memory-efficient fine-tuning — see [finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md).
- Quantization directly reduces the memory footprint relevant to KV cache management and serving/batching decisions — see [kv-cache-and-attention-optimization.md](kv-cache-and-attention-optimization.md) and [serving-and-batching.md](serving-and-batching.md). The KV cache itself can also be stored at reduced precision.
- Because LLM decoding is memory-bandwidth-bound, lower-precision weights speed up the decode phase in particular — see the prefill/decode discussion in [serving-and-batching.md](serving-and-batching.md).
- FP8 training sits alongside the distributed-training techniques in [distributed-training.md](../05-training-methodology/distributed-training.md) as a way to cut the memory and compute cost of large training runs.
- Quantization is one of several complementary compression techniques alongside pruning and distillation — see [pruning-and-distillation.md](pruning-and-distillation.md) — that are commonly stacked together in production deployments.

## Sources

- Frantar, Ashkboos, Hoefler, Alistarh, "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022)
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (2023)
- Micikevicius et al., "FP8 Formats for Deep Learning" (2022)
- Open Compute Project, "OCP Microscaling Formats (MX) Specification" v1.0 (2023); Rouhani et al., "Microscaling Data Formats for Deep Learning" (2023)
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024) [FP8 mixed-precision training at scale]
- OpenAI, gpt-oss-120b / gpt-oss-20b model card (Aug 2025) [MoE weights in MXFP4]
- NVIDIA, "Pretraining Large Language Models with NVFP4," arXiv 2509.25149 (Sep 2025; v2 Mar 2026) [12B model, 10T tokens; MMLU-Pro figures as reported]
