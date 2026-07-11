# RL for LLMs

This file is the dedicated bridge between the general reinforcement learning theory covered in [value-based-methods.md](value-based-methods.md), [policy-gradient-methods.md](policy-gradient-methods.md), and [model-based-rl.md](model-based-rl.md), and its specific application to training large language models. Rather than duplicating the full RLHF pipeline (covered end to end in [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)), this file's job is to explain *how the general RL concepts map onto the specific case of an LLM*, and to cover the RL-for-reasoning direction that has become one of the most significant recent methodological developments in the field.

## Mapping the general RL framing onto LLMs

Using the Markov Decision Process vocabulary from [value-based-methods.md](value-based-methods.md):

- **State** — the prompt plus whatever tokens have been generated so far in the current response. Each time the model generates a new token, the state grows by one token.
- **Action** — choosing which token to generate next, from the model's entire vocabulary. This is a huge action space (tens of thousands to over a hundred thousand possible tokens at every single step) compared to the small, fixed action spaces (a handful of possible moves) common in classic RL benchmarks like Atari games or board games.
- **Policy** — the language model itself: its next-token probability distribution, given the current state, is exactly a policy π(a|s; θ) in the sense covered in [policy-gradient-methods.md](policy-gradient-methods.md).
- **Reward** — for RLHF specifically, the score assigned by a trained reward model (see [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)) once a full response is complete; for the verifiable-reward approaches below, a reward computed by directly checking whether the final output is correct.
- **Episode** — one full prompt-to-response generation, from the first generated token to the end-of-response token.

## Why PPO fits this setting (and why it's a genuinely unusual RL setting)

**The fit.** PPO's clipped surrogate objective (see [policy-gradient-methods.md](policy-gradient-methods.md)) constrains how much the policy's behavior can change in a single update — this maps naturally onto the RLHF concern of not letting a language model's outputs drift too erratically from a well-validated starting point (the SFT model) in any single training step, which is exactly the role the KL penalty plays in the RLHF objective (see [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)).

**What makes this setting unusual, relative to classic RL benchmarks.** A few properties of the LLM-as-RL-environment setting are worth naming explicitly, since they differ meaningfully from the game-playing environments (Atari, Go, chess) where much of policy gradient theory was originally validated: the action space (vocabulary size) is enormous compared to classic RL benchmarks; a full "episode" (one generated response) can be very long (hundreds to thousands of tokens/actions) with reward typically only available at the very end (a **sparse reward** setting — feedback arrives once, at the end of a long sequence of decisions, rather than continuously); and the "environment" itself doesn't have independent dynamics the way a game does — the next state is fully and deterministically determined by appending whatever token the policy itself just chose, with no external randomness or opponent to react to. These properties are part of why RLHF in practice leans on the value-function-based **advantage** estimation techniques from [policy-gradient-methods.md](policy-gradient-methods.md) (typically a learned value function estimating expected total reward from any partial response, paired with GAE-style advantage estimation) to convert a single end-of-episode reward signal into a usable per-token training signal.

## RL on verifiable rewards — reasoning-focused training

**The idea.** Rather than relying on a learned reward model trained from human preference comparisons (which is itself an imperfect, learned approximation of human judgment — see the "reward hacking" discussion in [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)), some tasks have an objectively checkable correct answer — a math problem with a specific numerical answer, a coding problem with test cases that either pass or fail, a logic puzzle with a verifiable solution. For these tasks, the reward can be computed directly and automatically (did the final answer match the known correct answer? did the code pass the test suite?) rather than needing a separate, imperfect, human-preference-trained reward model at all.

**Why this matters as a distinct methodological shift.** This sidesteps reward-model imperfection and reward hacking entirely for the tasks where it's applicable — a verifiable, automatic reward signal can't be "gamed" the way a learned reward model's blind spots can be, since there's no learned proxy standing between the policy's behavior and the actual ground-truth objective. This has been widely reported as a major methodological direction in training models toward stronger multi-step reasoning behavior (working through a problem step by step before producing a final answer), since RL training on verifiable outcomes gives a direct, unhackable training signal specifically rewarding "did you reach the correct answer," which can encourage a model to develop and reinforce whatever intermediate reasoning strategies actually lead to correct answers more often, rather than optimizing toward "what does a reward model believe looks like good reasoning."

**Honest caveats.** This approach is naturally limited to domains where correctness is genuinely, automatically checkable — it doesn't directly extend to open-ended tasks like creative writing or nuanced advice, where "correctness" isn't a well-defined, checkable property, and those tasks still rely on preference-based reward modeling (RLHF/DPO-style methods, see [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)) or a combination of approaches. This repository treats RL-on-verifiable-rewards as a genuinely important, widely-reported recent methodological development (see [08-history/timeline-2023-present.md](../08-history/timeline-2023-present.md) for its place in the field's recent history) rather than a full replacement for preference-based alignment methods, which remain necessary for the large fraction of real-world tasks without a clean, automatic correctness check.

## Comparison table

| Reward source | Applicable to | Main strength | Main limitation |
|---|---|---|---|
| Learned reward model (classic RLHF) | Broad — any task with human preference judgments available | General-purpose, works for open-ended/subjective tasks | Imperfect proxy; vulnerable to reward hacking |
| Direct preference data (DPO) | Same scope as RLHF | Simpler, more stable training | Same underlying data-quality dependence as RLHF |
| Verifiable/automatic reward | Narrower — tasks with checkable correctness (math, code, logic) | Unhackable, direct ground-truth signal | Doesn't extend to open-ended/subjective tasks |

## Relationship to other algorithms

- The full RLHF training pipeline this file maps RL theory onto is covered end to end in [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md) — read that file for the pipeline, this file for the RL-theory framing underneath it.
- PPO's mechanics are covered in [policy-gradient-methods.md](policy-gradient-methods.md); the sparse-reward, single-episode-signal setting described here is why advantage estimation (GAE) from that file is especially relevant in this application.
- The reasoning-focused RL direction connects directly to the roadmap discussion of test-time compute scaling in [09-roadmaps/long-term-outlook.md](../09-roadmaps/long-term-outlook.md) and to the historical narrative in [08-history/timeline-2023-present.md](../08-history/timeline-2023-present.md).

## Sources

- Ouyang et al., "Training language models to follow instructions with human feedback" (2022) [InstructGPT — RLHF applied to LLMs]
- Schulman, Wolski, Dhariwal, Radford, Klimov, "Proximal Policy Optimization Algorithms" (2017) [cross-referenced from policy-gradient-methods.md]
- Widely reported industry and research direction (not attributed to a single foundational paper here): reinforcement learning on verifiable/automatically-checkable rewards for reasoning-focused post-training, an active and fast-moving area as of 2026
