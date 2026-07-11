# RLHF and Alignment

RLHF (Reinforcement Learning from Human Feedback) is the training methodology that turns a raw, next-token-predicting pretrained model (see [pretraining-strategies.md](pretraining-strategies.md)) into an assistant that follows instructions, avoids harmful outputs, and generally behaves the way its developers intend — a process broadly referred to as **alignment**. This file walks the full pipeline end to end, then covers the RL-free alternatives (DPO) and related techniques (RLAIF, Constitutional AI) that have reshaped the space since RLHF was first popularized. This file assumes familiarity with basic RL concepts covered in [07-reinforcement-learning](../07-reinforcement-learning/) — particularly policy gradient methods and PPO (see [policy-gradient-methods.md](../07-reinforcement-learning/policy-gradient-methods.md)) — and is cross-linked heavily with [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md), which serves as the RL-theory-to-practice bridge for this specific application.

## Why pretraining alone isn't enough

**The gap.** A model trained purely via next-token prediction on internet text (see [pretraining-strategies.md](pretraining-strategies.md)) learns to predict what token is *statistically likely* to come next in text resembling its training data — which is a different objective than "produce the response a user actually wants," "follow an instruction faithfully," or "refuse a harmful request." A raw pretrained model, prompted with a question, might just as easily continue with a plausible-sounding but unhelpful continuation (e.g., listing more similar questions, since that's a common pattern in web text) as it would with a direct, helpful answer. RLHF exists to close this gap: to shape the model's behavior toward what humans actually want from an assistant, using human judgments as the training signal, rather than relying purely on the statistical patterns of raw text.

## The RLHF pipeline, end to end

### Stage 1: Supervised fine-tuning (SFT)

**Mechanism.** Before any RL happens, the pretrained model is first fine-tuned (see [finetuning-and-peft.md](finetuning-and-peft.md)) on a curated dataset of high-quality example conversations — prompts paired with the kind of response a human demonstrator considers good (helpful, well-formatted, appropriately cautious). This is standard supervised learning: the model is trained to directly imitate these human-written or human-approved example responses, using the same next-token cross-entropy loss as pretraining (see [loss-functions.md](../01-foundations/loss-functions.md)), just on this much smaller, curated, instruction-and-response-shaped dataset.

**Why this stage exists.** SFT gets the model into roughly the right "mode" — responding to instructions in an assistant-like format, rather than continuing text the way a raw pretrained model would — providing a much better starting point for the RL stages that follow than the raw pretrained model would be.

### Stage 2: Reward model training

**Mechanism.** Collect many prompts, generate multiple candidate responses to each (usually from the SFT model), and have human labelers rank or compare these responses (commonly via pairwise comparisons: "which of these two responses is better?" — easier and more reliable for humans to judge consistently than assigning an absolute numeric score). Train a separate neural network — the **reward model** — to predict these human preference judgments: given a prompt and a response, output a scalar score such that responses humans preferred receive higher scores than responses they didn't.

**Why a separate model, and why scores rather than direct supervision.** Collecting a comparison judgment from a human for every possible response the policy might generate during RL training would be far too slow and expensive — RL training requires evaluating enormous numbers of generated responses over the course of training. The reward model is trained once, on a manageable quantity of real human comparisons, and then used as a fast, automated stand-in for a human judge throughout the (much larger) RL training process that follows — a proxy for human preference that can be queried instantly and as many times as needed.

### Stage 3: RL policy optimization (PPO-based)

**Mechanism.** The SFT model (now called the **policy** — the RL term for "the thing being trained to make decisions," here, decisions about which tokens to generate) is further trained using reinforcement learning: for a given prompt, the policy generates a response, the reward model scores that response, and the policy's parameters are updated (via PPO — Proximal Policy Optimization, covered in full mechanical depth in [policy-gradient-methods.md](../07-reinforcement-learning/policy-gradient-methods.md)) to make higher-reward responses more likely in the future.

**The KL penalty against drifting from the SFT model.** A critical detail: the RL objective doesn't purely maximize reward model score. It also includes a penalty term — a KL divergence (see [loss-functions.md](../01-foundations/loss-functions.md)) between the current policy's output distribution and the original SFT model's output distribution — that discourages the policy from drifting too far from its starting point:

```
objective = 𝔼[reward_model_score] − β · D_KL(policy ‖ SFT_model)
```

Walkthrough: without this penalty, the RL process has an incentive to exploit any imperfection in the (imperfect, learned-from-a-finite-sample) reward model — finding some narrow, possibly bizarre pattern of outputs that the reward model happens to score highly but that a human would not actually consider good (a failure mode generally called **reward hacking**, or more specifically here "reward model over-optimization"). The KL penalty acts as a leash, keeping the policy's behavior anchored reasonably close to the SFT model's more broadly-validated behavior, trading off some potential reward-model-score gains for keeping the policy's outputs recognizably reasonable.

**Why this whole pipeline mattered.** RLHF, as popularized by InstructGPT (Ouyang et al., 2022) and then ChatGPT, was the methodology that took capable-but-unwieldy pretrained language models and turned them into the helpful, instruction-following assistants that defined the LLM product category from 2022 onward — see [08-history/timeline-2017-2023.md](../08-history/timeline-2017-2023.md) for this moment in the field's broader history.

## Direct Preference Optimization (DPO) — an RL-free alternative

**Origin.** Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023).

**The problem DPO addresses.** The full RLHF pipeline above is complex and can be fragile in practice: it requires training and maintaining a separate reward model, running an actual RL training loop (PPO, with its own stability challenges — see [policy-gradient-methods.md](../07-reinforcement-learning/policy-gradient-methods.md)), and carefully tuning the KL penalty coefficient β, among other hyperparameters — considerably more moving parts, and more places for training to go wrong, than a standard supervised fine-tuning run.

**Core mechanism (conceptual).** DPO's key mathematical insight is that the RLHF objective above (reward maximization with a KL penalty against a reference policy) has a closed-form relationship between the optimal policy and an implicit reward function — which means you can skip training an explicit reward model and skip running RL entirely, and instead directly optimize the policy on human preference comparison data using a loss function that implicitly encodes "make preferred responses more likely relative to dispreferred ones, without straying too far from the reference model" — all via a single, ordinary supervised-learning-style training loop (not unlike standard fine-tuning) computed directly on pairs of (preferred, dispreferred) responses.

**Why it mattered.** DPO achieves results broadly comparable to full RLHF in many settings while being dramatically simpler to implement and more stable to train — no separate reward model, no RL training loop, no PPO-specific hyperparameter tuning — which made preference-based fine-tuning far more accessible to teams without deep RL infrastructure and expertise.

## RLAIF (RL from AI Feedback)

**Name & definition.** RLAIF replaces (or supplements) human preference labels with preference judgments generated by another AI model, reducing the amount of human labeling required.

**Why it emerged.** Collecting large-scale, high-quality human preference data is slow and expensive. If a capable enough AI model can produce preference judgments that correlate well with what humans would have said, that judgment can be used in place of (or alongside) human labels at a fraction of the cost and time, letting preference datasets scale further than human labeling budgets alone would allow.

**Tradeoffs.** RLAIF can meaningfully cut the cost and turnaround time of preference data collection, but it inherits whatever biases or blind spots the judging AI model has, and its judgments are only as trustworthy as that model's own alignment and reliability — a real, structurally unavoidable limitation worth stating plainly rather than glossing over.

## Constitutional AI

**Origin.** Bai et al. (Anthropic), "Constitutional AI: Harmlessness from AI Feedback" (2022).

**Core mechanism.** Rather than relying purely on human-labeled examples of "harmful" vs. "acceptable" outputs, Constitutional AI uses a written set of explicit principles (a "constitution") and has a model critique and revise its own outputs according to those principles, generating training data for both a supervised fine-tuning stage (self-critique-and-revision examples) and a preference-comparison stage feeding into an RLAIF-style process — reducing the amount of human labeling needed specifically for harm-avoidance behavior, while making the guiding principles behind that behavior more explicit and auditable than an implicit pattern learned purely from scattered human labels.

**Why it mattered.** This approach made the values guiding a model's harm-avoidance behavior more explicit and directly inspectable (you can read the constitution itself) compared to a purely implicit pattern inferred from thousands of individual human labeling decisions, and reduced the human labeling burden specifically for safety-related behavior.

## Tradeoffs between RLHF and DPO-style methods

| Dimension | Full RLHF (reward model + PPO) | DPO (and similar direct preference methods) |
|---|---|---|
| Complexity | High (reward model + RL loop + KL tuning) | Low (single supervised-style training loop) |
| Training stability | Can be fragile (RL-specific instabilities) | Generally more stable |
| Compute cost | Higher (reward model training + RL rollouts) | Lower |
| Flexibility to reuse reward signal for other purposes | Yes (reward model is a reusable artifact) | No explicit reward model produced |
| Empirical quality (as widely reported) | Strong, proven at frontier scale over a longer track record | Often comparable; increasingly widely adopted, though the full comparative picture across every setting is still an active area of study |

**Current status.** As of 2026, both approaches are in active use across the field — some organizations rely primarily on DPO-style methods for simplicity and stability, others continue to use full RLHF (or hybrid pipelines combining elements of both) particularly where a standalone, reusable reward model or finer-grained RL-based control over the training process is valuable. This is an area where practice continues to evolve, and this repository avoids overstating a settled consensus that doesn't yet exist.

## Relationship to other algorithms

- PPO, the RL algorithm underlying classic RLHF, is covered in full mechanical depth in [policy-gradient-methods.md](../07-reinforcement-learning/policy-gradient-methods.md); [rl-for-llms.md](../07-reinforcement-learning/rl-for-llms.md) is the dedicated bridge file connecting general RL theory to this specific application.
- The KL divergence penalty here is the same mathematical object covered in [loss-functions.md](../01-foundations/loss-functions.md).
- SFT (Stage 1) uses the same fine-tuning mechanics covered in [finetuning-and-peft.md](finetuning-and-peft.md), often applied via PEFT methods like LoRA for efficiency.
- RLHF/DPO-tuned models are frequently the target of the inference-optimization techniques in [06-inference-optimization](../06-inference-optimization/) once training is complete.

## Sources

- Christiano et al., "Deep Reinforcement Learning from Human Preferences" (2017) — foundational RLHF methodology, predating its LLM application
- Ouyang et al. (OpenAI), "Training language models to follow instructions with human feedback" (2022) [InstructGPT]
- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)
- Bai et al. (Anthropic), "Constitutional AI: Harmlessness from AI Feedback" (2022)
- Bai et al. (Anthropic), "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback" (2022)
