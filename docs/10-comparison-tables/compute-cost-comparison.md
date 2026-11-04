# Compute Cost Comparison

This file focuses specifically on training and inference cost tradeoffs across the techniques covered in this repository — synthesis only, no new algorithms. Where [algorithm-comparison-master.md](algorithm-comparison-master.md) gives a broad "Compute Profile" column per category, this file goes deeper on the specific cost levers and how they stack together.

## Training cost levers

| Lever | What it trades off | Covered in |
|---|---|---|
| Model size (parameters) | More capacity vs. more compute/memory per step | [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md) (scaling laws) |
| Data size (tokens) | Better generalization vs. more total compute (parameters × tokens) | [pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md) |
| Batch size | Better hardware utilization/parallelism vs. optimizer stability challenges | [optimization-algorithms.md](../01-foundations/optimization-algorithms.md) (LAMB/LARS) |
| Parallelism strategy (data/tensor/pipeline/ZeRO) | Fits larger models/batches vs. communication overhead | [distributed-training.md](../05-training-methodology/distributed-training.md) |
| Full fine-tuning vs. PEFT | Maximum task performance vs. dramatically lower compute/memory | [finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md) |
| RLHF vs. DPO | Reusable reward model + fine-grained RL control vs. simplicity/stability | [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md) |
| MoE vs. dense architecture | More total capacity vs. added routing/serving complexity | [mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md) |
| Training precision (BF16 → FP8 → FP4) | Less memory/bandwidth and faster matmuls vs. more numerical engineering (scaling, outlier handling) | [quantization.md](../06-inference-optimization/quantization.md), [pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md) |
| RL post-training vs. on-policy distillation | Discovering new behavior via exploration vs. a much denser, cheaper per-token signal from an existing teacher | [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md), [pruning-and-distillation.md](../06-inference-optimization/pruning-and-distillation.md) |

**The Chinchilla-era framing (see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md) and [pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md)):** for a fixed training compute budget, the split between model size and data size is itself a cost decision, not a free variable — training a smaller model on more data can match or beat a larger model trained on proportionally less data, at the same total compute cost, and additionally produces a model that's cheaper to *serve* afterward (see inference costs below) — which is why real-world pretraining decisions since 2022 have often pushed data quantity beyond the narrowly compute-optimal point, trading a bit more pretraining compute for meaningfully lower long-run inference cost.

## Inference cost levers

| Lever | What it trades off | Covered in |
|---|---|---|
| Quantization (bit-width) | Lower memory/faster inference vs. accuracy loss (larger at lower bit-widths) | [quantization.md](../06-inference-optimization/quantization.md) |
| Pruning/distillation | Smaller/faster model vs. upfront compression cost + capability ceiling set by original/teacher model | [pruning-and-distillation.md](../06-inference-optimization/pruning-and-distillation.md) |
| KV cache management (PagedAttention, GQA/MQA) | Higher achievable batch concurrency vs. some quality tradeoff (GQA/MQA) or engineering complexity (paging) | [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md) |
| Speculative decoding | Fewer expensive sequential model calls vs. draft model overhead/complexity | [speculative-decoding.md](../06-inference-optimization/speculative-decoding.md) |
| Batching strategy (static vs. continuous) | Throughput vs. individual request latency | [serving-and-batching.md](../06-inference-optimization/serving-and-batching.md) |
| Context length | More usable input vs. quadratic attention cost and larger KV cache | [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md), [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md) |
| Attention design (GQA → MLA → sparse/compressed → hybrid linear) | Smaller KV cache and sub-quadratic cost at long context vs. architecture-time commitment and some recall risk | [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md), [state-space-models.md](../03-deep-learning-architectures/state-space-models.md) |
| Reasoning length (test-time compute) | Higher accuracy on hard problems vs. many more decode tokens per request | [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md) |
| Prefill/decode disaggregation | Independent scaling and better utilization vs. KV-cache transfer cost and system complexity | [serving-and-batching.md](../06-inference-optimization/serving-and-batching.md) |

**A notable asymmetry worth naming plainly:** most inference-cost levers in the table above (quantization, pruning, distillation) trade some accuracy for speed/cost — a real tradeoff decision someone has to make deliberately. Two notable exceptions, covered in their own files precisely because they're unusual: FlashAttention (exact same output, just faster/more memory-efficient — see [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md)) and speculative decoding (exact same output distribution, just faster — see [speculative-decoding.md](../06-inference-optimization/speculative-decoding.md)) are close to "free lunches" in this repository's coverage, which is exactly why they've seen such broad, fast, uncontroversial adoption relative to techniques that require an explicit quality-for-speed tradeoff decision.

## How these levers stack in practice

A realistic production LLM serving stack, as described across [06-inference-optimization](../06-inference-optimization/), typically combines several of these simultaneously rather than choosing just one: a model trained with Chinchilla-informed (or beyond-Chinchilla, serving-cost-aware) data scaling → quantized to 4-8 bits for deployment (by late 2026, increasingly FP8 or block-scaled FP4 formats with native hardware support) → served with GQA or MLA, possibly with sparse attention or hybrid linear-attention layers (architectural choices made at pretraining time) → using PagedAttention-style KV cache management and prefix caching → with continuous batching, and at large scale with prefill and decode disaggregated onto separate GPU pools → and speculative decoding on top (often via the model's own MTP or EAGLE-style draft head). Each layer addresses a different part of the total cost equation (model size, memory footprint, batch concurrency, sequential call count), and this repository's coverage of vLLM and TensorRT-LLM (see [serving-and-batching.md](../06-inference-optimization/serving-and-batching.md)) reflects exactly this kind of combined stack, not a single silver-bullet technique.

## The reasoning-model shift: inference compute becomes a scaling axis

Since reasoning models (o1 in 2024, DeepSeek-R1 in 2025 — see [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)), a growing share of total compute is spent at inference time: a single hard request may generate thousands to tens of thousands of "thinking" tokens before answering. This changes which levers matter most. Decode-side optimizations — KV cache size (MLA, sparse attention, hybrids), speculative decoding, low-precision serving, and disaggregated serving — now carry more of the cost equation than they did when typical responses were short, and "how long should the model think?" has become a per-request cost/quality dial in its own right.

## Training cost vs. inference cost: a genuine tension

**The tradeoff, stated plainly.** Spending more compute at training time (more data, per the Chinchilla-era logic above, or techniques like QAT instead of cheaper PTQ — see [quantization.md](../06-inference-optimization/quantization.md)) can produce a model that's cheaper to run at inference time for a given quality bar. Given that a deployed model may be served to enormous numbers of users over a long lifetime, this training-for-serving-savings tradeoff can be worth substantial extra upfront training compute — this is a recurring theme across [pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md) and [quantization.md](../06-inference-optimization/quantization.md), and this repository treats it as one of the more important, if less flashy, economic realities shaping frontier model development decisions.

## Relationship to other files

- [algorithm-comparison-master.md](algorithm-comparison-master.md) gives the broader per-algorithm comparison this file drills into specifically on cost.
- [when-to-use-what.md](when-to-use-what.md) turns these cost tradeoffs into concrete decision guidance.
