# Long-Term Outlook (2-5 Years)

This file covers more speculative, longer-horizon directions (roughly 2-5 years out from mid-2026) than [near-term-outlook.md](near-term-outlook.md). The further out a prediction reaches, the less confidence this repository has in it — every claim here is explicitly `[Projection]`, and this file leans more heavily on "here's a plausible direction and why" than "here's what will happen."

## Alternative architectures to attention at scale

**The open question.** State space models (see [state-space-models.md](../03-deep-learning-architectures/state-space-models.md)) and related linear-recurrence architectures remain, as of mid-2026, a minority approach relative to attention-based Transformers, primarily competitive for very long context lengths where attention's quadratic cost is most punishing.

**`[Projection]`** Over a 2-5 year horizon, this repository considers it plausible — though genuinely uncertain — that hybrid architectures (combining a majority of efficient SSM-style layers with a smaller number of full attention layers) become more common in frontier models, rather than either pure SSMs fully replacing attention or attention remaining completely unchallenged. `[Projection]` A full replacement of attention-based Transformers as the dominant architecture for frontier general-purpose models within this window seems less likely than continued hybrid experimentation and coexistence, given attention's substantial head start in tooling maturity, training recipes, and empirical track record at the largest scales — but this repository holds this projection with real uncertainty, and flags it as one of the more genuinely open architectural questions in the field.

## Continual / online learning approaches

**The current limitation.** Frontier models today are, overwhelmingly, trained in large, discrete pretraining/post-training cycles and then deployed with largely fixed weights — they don't continuously update their own weights from ongoing deployment interactions the way a human continues learning from experience. This is directly related to the catastrophic forgetting problem covered in [open-problems.md](open-problems.md): naively continuing to train a deployed model on new data risks degrading its existing capabilities.

**`[Projection]`** This repository considers meaningful progress on practical continual/online learning methods a plausible development over a 2-5 year horizon, motivated by the clear commercial and practical value of models that can incorporate new information without a full retraining cycle — but this repository is explicit that this remains, as of mid-2026, a genuinely unsolved research problem rather than an emerging engineering practice, and treats confident claims of an imminent solution with real skepticism. `[Projection]` Progress here seems more likely to arrive incrementally (better fine-tuning/PEFT-based updating strategies — see [finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md) — combined with retrieval-based approaches that avoid needing to update weights at all for some categories of new information) than via a single decisive breakthrough.

## More aggressive test-time compute scaling

**The current trajectory.** The reasoning-focused training developments covered in [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md) and [timeline-2023-present.md](../08-history/timeline-2023-present.md) have already demonstrated that letting a model spend more computation at inference time (generating longer chains of intermediate reasoning before a final answer) can improve performance on certain tasks, complementing the pretraining-time scaling laws covered in [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md).

**`[Projection]`** This repository expects continued exploration of this "test-time compute" axis as a lever distinct from raw pretraining scale over the next several years — plausibly including more sophisticated methods for a model to decide *how much* extra computation a given problem warrants (spending more deliberation on hard problems, less on easy ones) rather than a fixed amount of extra reasoning applied uniformly. `[Projection]` Whether this becomes as reliable and well-understood a scaling axis as pretraining compute has been (with its own analogue of the scaling laws covered in [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)) is, as of this writing, a genuinely open empirical question this repository does not consider settled.

## A methodological note on this file's uncertainty

Per this repository's ground rules (see [scope-and-methodology.md](../00-overview/scope-and-methodology.md)), the further a claim reaches into the future, the less this repository asks you to trust it as anything beyond "a reasonable person's extrapolation from current public trends." This file's job is to give a plausible, informed shape to the next few years, not to pretend to a certainty this repository — or, frankly, anyone publicly writing about the field in mid-2026 — actually has.

## Comparison table

| Direction | Current status (mid-2026) | `[Projection]` for 2-5 years out |
|---|---|---|
| Attention alternatives (SSMs, hybrids) | Minority approach, strongest case for very long context | Plausible growth in hybrid designs; full replacement considered unlikely but not impossible |
| Continual/online learning | Largely unsolved research problem | Incremental progress plausible; full solution not confidently expected |
| Test-time compute scaling | Emerging, actively researched | Continued growth as a distinct scaling axis; full maturity/reliability uncertain |

## Relationship to other files

- Continual learning connects directly to the catastrophic forgetting problem in [open-problems.md](open-problems.md).
- Attention alternatives are covered in their current, present-day state in [state-space-models.md](../03-deep-learning-architectures/state-space-models.md), which this file explicitly builds forward from.
- Test-time compute scaling is the direct forward extension of the RL-for-reasoning direction in [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md).

## Sources

- This file's `[Projection]` claims are the author's own reasoned extrapolation, explicitly distinguished from the sourced, present-day claims in [state-space-models.md](../03-deep-learning-architectures/state-space-models.md), [finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md), and [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md) that motivate them.
