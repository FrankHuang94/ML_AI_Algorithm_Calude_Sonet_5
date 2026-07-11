# Open Problems

This file covers significant unsolved problems in ML/AI as of mid-2026 — framed honestly as open research problems, not issues on the verge of being solved. Unlike [near-term-outlook.md](near-term-outlook.md) and [long-term-outlook.md](long-term-outlook.md), this file makes fewer forward-looking predictions about *when* or *whether* these get solved, and focuses instead on precisely stating what's actually unresolved and why it's hard.

## Catastrophic forgetting

**The problem.** When a trained model is further trained on new data (whether via full fine-tuning or continued pretraining), it tends to lose some of its previously-learned capabilities in the process — the new training signal can overwrite representations that were important for earlier-learned tasks/knowledge, since nothing in standard gradient-based training inherently protects previously-useful weight configurations.

**Why it's hard.** There's a fundamental tension here: a model needs to be plastic enough to actually learn new things, but stable enough not to lose old things — and standard neural network training doesn't have a principled, general-purpose way to have both simultaneously. This is the central obstacle to the continual/online learning direction discussed as a `[Projection]` in [long-term-outlook.md](long-term-outlook.md) — until this is better addressed, continuously updating a deployed model's weights from live data remains risky rather than routine practice.

## Hallucination and factuality

**The problem.** Language models, including capable frontier ones, can generate fluent, confident-sounding statements that are factually incorrect — a failure mode broadly called "hallucination." This isn't a simple bug to patch; it's a fairly direct consequence of how these models are trained (next-token prediction over training data — see [autoregressive-generation.md](../04-generative-models/autoregressive-generation.md) — optimizes for plausible-sounding continuations, which is correlated with, but not identical to, factual accuracy) and how they're evaluated (a model that confidently guesses often scores better on many benchmarks than one that appropriately expresses uncertainty, unless the evaluation is specifically designed to penalize confident wrongness).

**Why it's hard.** A model has no built-in, reliable mechanism for distinguishing "things I actually know well" from "things I'm pattern-matching a plausible-sounding answer for" — its internal confidence (as reflected in output probabilities) doesn't always correlate cleanly with actual correctness (a calibration problem — see the glossary). Mitigations exist (retrieval-augmented approaches that ground answers in retrieved source documents, RLHF/training approaches that reward appropriate expressions of uncertainty, verifiable-reward RL for domains where correctness is checkable — see [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md)) but none of them constitute a full solution as of mid-2026, and this remains one of the most practically consequential open problems in deploying these systems for tasks where factual reliability matters.

## Interpretability

**The problem.** For a large neural network, it remains generally difficult to give a precise, reliable, human-understandable account of *why* the model produced a specific output, or what internal computation led to a given behavior — the model's reasoning is distributed across billions of parameters in a way that resists simple explanation, in sharp contrast to something like a decision tree (see [supervised-learning.md](../02-classical-ml/supervised-learning.md)), where you can literally read off the exact logic used.

**Why it's hard.** There's active, genuinely promising research (mechanistic interpretability approaches that attempt to reverse-engineer specific learned circuits or features within a network) but this work has, as of mid-2026, illuminated meaningful pieces of how specific narrow behaviors arise rather than providing a comprehensive, reliable account of frontier model behavior as a whole. This matters beyond pure scientific curiosity: better interpretability would directly help with debugging unwanted behaviors, verifying safety properties, and building justified trust in high-stakes deployments — which is exactly why its current limitations are a genuinely significant open problem, not just an academic curiosity.

## Long-horizon agentic reliability

**The problem.** As covered in [timeline-2023-present.md](../08-history/timeline-2023-present.md), models are increasingly trained and used for longer, multi-step, tool-using tasks with reduced step-by-step human supervision. Reliability over long task horizons remains a genuine, unresolved challenge — small per-step error rates compound over many steps, and a model may fail to notice or recover from its own earlier mistakes partway through a long task, an error-propagation problem that gets structurally worse as task length and step count grow, not better.

**Why it's hard.** This isn't simply a matter of "make each individual step more accurate" — even a model with a very low per-step error rate can have a high overall failure rate across a sufficiently long task, purely from compounding, unless the model also reliably self-corrects (notices something has gone wrong and adjusts) rather than continuing to build on a flawed intermediate state. Reliable self-correction over long horizons is itself an open research question, not a solved capability, as of mid-2026.

## Energy/compute cost trajectory

**The problem.** Training and serving frontier-scale models requires enormous amounts of compute and, correspondingly, energy — a cost trajectory that has grown substantially as model scale has grown, raising genuine questions about long-term sustainability, resource allocation, and who can realistically afford to train and deploy frontier-scale systems.

**Why it's hard.** This isn't purely a technical problem solvable by a single algorithmic trick — it involves a genuine tension between the scaling-law-driven benefits of larger models/more data/more compute (see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)) and real-world resource constraints (energy availability, hardware manufacturing capacity, environmental impact, and cost). The inference-optimization techniques in [06-inference-optimization](../06-inference-optimization/) meaningfully reduce cost-per-token, and this has been genuine, real progress, but they haven't eliminated the underlying tension between "bigger/more-trained models tend to perform better" and "bigger/more-trained models cost more to build and run" — this remains an active, structurally unresolved tension rather than a solved allocation problem.

## Evaluation validity

**The problem.** As models have become more capable, it's become genuinely harder to evaluate them meaningfully — many popular benchmarks have become saturated (models score near-perfectly, leaving little room to measure further improvement) or are suspected to have leaked into training data (a model may score well on a benchmark partly because it has effectively memorized answers related to that benchmark from its training data, rather than demonstrating the underlying capability the benchmark was designed to measure).

**Why it's hard.** Designing evaluations that reliably measure genuine capability (rather than benchmark-specific memorization or narrow overfitting to a test's particular format) — especially for open-ended, real-world-relevant tasks rather than narrow, easily-scored ones — is itself a substantial, unsolved methodological challenge, and the field's own tools for measuring progress are consequently less trustworthy than they ideally would be. This is a somewhat self-referential problem worth flagging plainly: this repository's own claims about "current state of the art" throughout its files are only as reliable as the evaluation methodology behind the results being cited, which is itself part of what's unsettled here.

## Comparison table

| Open problem | Core tension | Directly connects to |
|---|---|---|
| Catastrophic forgetting | Plasticity (learn new things) vs. stability (keep old things) | [long-term-outlook.md](long-term-outlook.md) (continual learning) |
| Hallucination/factuality | Plausible-sounding output vs. actually-correct output | [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md) (verifiable rewards) |
| Interpretability | Distributed, high-dimensional computation vs. human-understandable explanation | [supervised-learning.md](../02-classical-ml/supervised-learning.md) (contrast with interpretable classical models) |
| Long-horizon agentic reliability | Per-step error rate vs. compounding failure over many steps | [timeline-2023-present.md](../08-history/timeline-2023-present.md) (agentic training) |
| Energy/compute cost | Scaling-law benefits vs. real-world resource constraints | [06-inference-optimization](../06-inference-optimization/) (cost-reduction techniques) |
| Evaluation validity | Benchmark saturation/leakage vs. genuine capability measurement | Cuts across every "current status" claim in this repository |

## Relationship to other files

- These problems are referenced from, and directly motivate, several `[Projection]` claims in [near-term-outlook.md](near-term-outlook.md) and [long-term-outlook.md](long-term-outlook.md).
- Interpretability's contrast with classical, inherently-interpretable models is best understood alongside [supervised-learning.md](../02-classical-ml/supervised-learning.md) and [ensemble-methods.md](../02-classical-ml/ensemble-methods.md).
- Evaluation validity is a standing caveat relevant to essentially every "current status"/"still relevant (2026)" judgment made throughout this repository, and is worth keeping in mind as a background qualifier on those claims.

## Sources

- This file characterizes widely-discussed, actively-researched open problems in the field rather than summarizing a single paper per problem; where specific named techniques are referenced as partial mitigations, see the cross-linked files above for their individual sourcing.
