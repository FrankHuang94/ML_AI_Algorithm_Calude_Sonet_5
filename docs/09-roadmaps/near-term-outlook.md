# Near-Term Outlook (6-18 Months)

This file covers trends this repository expects to continue or accelerate over roughly the next year to eighteen months, as of mid-2026. Per this repository's methodology (see [scope-and-methodology.md](../00-overview/scope-and-methodology.md)), every forward-looking claim here is labeled `[Projection]` — treat these as reasoned extrapolations from public research trends and widely-reported industry direction, not settled fact. Where a claim is grounded in already-published research or publicly stated lab direction rather than extrapolation, that's noted explicitly as such.

## Inference cost reduction trends

**Grounded in current trajectory.** The inference-optimization techniques covered in [06-inference-optimization](../06-inference-optimization/) — quantization, GQA/MQA, PagedAttention, speculative decoding, continuous batching — have each individually demonstrated substantial cost/latency improvements, and are increasingly deployed in combination rather than in isolation.

**`[Projection]`** Continued, roughly incremental improvement (rather than a single dramatic breakthrough) in cost-per-token for a given quality level over the next 6-18 months, driven by: further maturation of low-bit quantization (sub-4-bit methods moving from experimental to more production-ready — see [quantization.md](../06-inference-optimization/quantization.md)), wider adoption of speculative decoding variants that don't require a separate draft model (Medusa-style and lookahead-style approaches — see [speculative-decoding.md](../06-inference-optimization/speculative-decoding.md)), and continued hardware-software co-design (new accelerator generations paired with kernel-level optimizations like FlashAttention successors). `[Projection]` This is expected to be a steady, compounding trend rather than a single step-change, consistent with the pattern observed over the past several years.

## Continued context-length scaling

**Grounded in current trajectory.** Context windows have grown substantially over the past several years (see [timeline-2023-present.md](../08-history/timeline-2023-present.md)), driven by a combination of architectural changes (GQA, more efficient positional encoding extrapolation) and serving-infrastructure improvements (PagedAttention-style memory management).

**`[Projection]`** Context windows are likely to continue growing over the next 6-18 months, though this repository expects the rate of *headline number* growth (the maximum advertised context length) to matter less than improvements in how well models actually *use* long context — the field has widely reported a persistent gap between "can technically accept N tokens of input" and "reliably attends to and reasons over all of those N tokens equally well," particularly for information located in the middle of a long context rather than at the beginning or end. `[Projection]` Closing this usability gap, not just extending the raw maximum length, is a more meaningful near-term marker of progress than the maximum context number alone.

## Agentic and tool-use training approaches

**Grounded in current trajectory.** As covered in [timeline-2023-present.md](../08-history/timeline-2023-present.md), training models to use external tools and operate over longer task horizons with reduced step-by-step supervision has become a major focus area, building on the reasoning-focused RL developments covered in [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md).

**`[Projection]`** This repository expects continued, fairly rapid investment in training methodology specifically targeting multi-step tool use, longer-horizon task completion, and better error recovery (a model noticing and correcting its own mistake mid-task, rather than continuing to compound it) over the next 6-18 months. `[Projection]` This is likely to be an area of highly visible capability improvement paired with correspondingly visible new failure modes — the honest expectation is uneven, bumpy progress rather than a smooth capability curve, consistent with how most rapidly-developing capability areas in this field's history have actually played out (see [open-problems.md](open-problems.md) for the long-horizon reliability problem this connects to).

## Ongoing MoE and sparsity adoption trends

**Grounded in current trajectory.** Mixture-of-Experts adoption (see [mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md)) has grown across multiple frontier labs' largest models, following the accuracy-per-compute-FLOP logic covered in that file.

**`[Projection]`** This repository expects continued, and likely broadening, MoE adoption for the largest-scale models over the next 6-18 months, alongside continued research into more sophisticated routing and load-balancing schemes aimed at closing some of MoE's remaining gaps (training stability, memory-footprint-vs-compute tradeoffs — see [mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md)). `[Projection]` Dense models are not expected to disappear — this repository projects continued use of dense architectures specifically for smaller, latency-sensitive, or simpler deployment scenarios where MoE's added complexity isn't justified, rather than a full-field shift to MoE-only.

## Summary table

| Trend | Confidence basis | Direction of `[Projection]` |
|---|---|---|
| Inference cost reduction | Strong current trajectory across multiple independent techniques | Continued incremental, compounding improvement |
| Context-length scaling | Strong current trajectory | Continued growth; usability gap matters more than raw max length |
| Agentic/tool-use training | Strong current trajectory, high visibility | Rapid but uneven progress; new failure modes alongside new capabilities |
| MoE/sparsity adoption | Strong current trajectory | Continued broadening at the largest scale; dense models persist for smaller/simpler use cases |

## Relationship to other files

- This file's projections build directly on the current-state assessments in [06-inference-optimization](../06-inference-optimization/), [mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md), and [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md).
- For longer-horizon, more speculative directions, see [long-term-outlook.md](long-term-outlook.md).
- For the specific reliability/robustness gaps referenced here (agentic error recovery, long-context usability), see [open-problems.md](open-problems.md).

## Sources

- This file's `[Projection]` claims are the author's reasoned extrapolation from the public research and industry trends documented throughout this repository (particularly [06-inference-optimization](../06-inference-optimization/), [03-deep-learning-architectures/mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md), and [08-history/timeline-2023-present.md](../08-history/timeline-2023-present.md)), not a separate set of citations.
