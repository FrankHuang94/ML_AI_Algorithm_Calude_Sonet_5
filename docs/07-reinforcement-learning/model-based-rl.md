# Model-Based RL

Model-based reinforcement learning is the counterpart to the model-free methods covered in [value-based-methods.md](value-based-methods.md) and [policy-gradient-methods.md](policy-gradient-methods.md): rather than learning purely from direct trial-and-error interaction with the real environment, a model-based agent learns (or is given) a model of how the environment behaves, and uses that model to plan or simulate ahead.

## World models

**Name & definition.** A world model is a learned (or given) representation of how an environment evolves — given the current state and an action, what state (and reward) comes next. Once you have such a model, an agent can "imagine" the likely consequences of different action sequences without actually having to execute them in the real environment, and use that imagined lookahead to plan better decisions or to generate additional synthetic training experience.

**Why this is valuable in principle.** Real-world interaction is often expensive, slow, or risky (a robot damaging itself while exploring randomly, a game taking real wall-clock time to play out, a real-world trial being simply impossible to run many times). A good world model lets an agent get much of the benefit of "trying things out" — seeing what would likely happen — without paying the real cost of doing so.

## MuZero

**Origin.** Schrittwieser et al. (DeepMind), "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (2019/2020).

**Core mechanism (brief but concrete).** MuZero learns three connected components, entirely from data, without ever being told the actual rules of the games it's mastering: a **representation function** (maps raw observations, like a board position, into an internal state representation), a **dynamics function** (predicts what internal state and reward come next, given a current internal state and an action — MuZero's learned world model, but notably operating in this internal, abstract representation space rather than trying to predict raw pixels or a literal board layout), and a **prediction function** (estimates both a value and a recommended policy from a given internal state). These three pieces are used together inside a tree-search planning procedure (a Monte Carlo Tree Search — a systematic way of exploring possible future action sequences by simulating many possible continuations and using the results to judge which immediate action looks best) to decide on each actual move, and are all trained jointly based on how well the resulting plans and predictions actually matched real outcomes.

**Why it mattered.** Earlier successes in this vein (like DeepMind's AlphaGo and AlphaZero systems) relied on being given the exact rules of the game as a hand-coded simulator to plan against. MuZero's headline result was matching or exceeding that level of performance across multiple different games (Atari, Go, chess, shogi) while learning its own internal model of the "rules" and dynamics purely from experience, without ever being explicitly told them — demonstrating that planning-based approaches don't strictly require a hand-given environment simulator, only a good enough learned substitute.

## Sample efficiency: the core motivation vs. model-free methods

**The core tradeoff.** Model-free methods (Q-learning, DQN, PPO, and the rest of [value-based-methods.md](value-based-methods.md) and [policy-gradient-methods.md](policy-gradient-methods.md)) generally require a very large number of real interactions with the environment to learn good behavior, since every bit of learning signal comes directly from actual trial and error. Model-based methods, once they have a reasonably accurate model, can generate large amounts of *additional*, "imagined" experience from that model — planning ahead, or training on simulated rollouts — substantially improving **sample efficiency** (how much real-world interaction is needed to reach a given level of performance).

**The catch.** This benefit is entirely conditional on the learned model actually being accurate enough to trust. An inaccurate world model can lead an agent to plan confidently around consequences that don't actually happen in the real environment — errors in the model can compound over a long imagined rollout, and the agent may end up "learning" a policy that's well-optimized for its flawed internal model rather than for reality. This is the central, honestly-stated tradeoff of the whole model-based approach: potentially much better sample efficiency, purchased at the cost of a genuinely harder learning problem (learning an accurate model in the first place) and a real risk of model-exploitation failure modes that model-free methods, by construction, don't have to worry about (since they never plan against an internal model that could be wrong).

## Current status

Model-based RL, and MuZero-style planning specifically, remain most prominent in domains with a clear, well-defined notion of state and action and where sample efficiency genuinely matters most (game-playing research being the most publicly visible showcase), rather than as a general-purpose default across all of reinforcement learning. For the specific application most relevant to the rest of this repository — RL applied to LLM training and alignment (see [rl-for-llms.md](rl-for-llms.md) and [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)) — model-free policy gradient methods (specifically PPO, and the RL-free DPO alternative) remain the standard approach as of 2026, rather than explicit world-model-based planning; the enormous, open-ended nature of language generation makes learning an explicit, trustworthy world model of "what happens next in this conversation" a substantially harder and less clearly-scoped problem than it is in a well-defined game environment.

## Comparison table

| Approach | Learns/uses a model? | Sample efficiency | Risk |
|---|---|---|---|
| Model-free (Q-learning, DQN, PPO) | No | Lower (needs more real interaction) | No model-exploitation risk |
| Model-based (world models, MuZero-style planning) | Yes | Higher, if the model is accurate | Compounding model errors can mislead planning |

## Relationship to other algorithms

- Model-based RL is the direct structural counterpart to the model-free methods in [value-based-methods.md](value-based-methods.md) and [policy-gradient-methods.md](policy-gradient-methods.md).
- MuZero's tree-search planning procedure builds on ideas from AlphaGo/AlphaZero (referenced in [08-history/timeline-2017-2023.md](../08-history/timeline-2017-2023.md) in broader historical context).
- The sample-efficiency motivation here parallels, at a conceptual level, why synthetic data generation (see [curriculum-and-data-strategies.md](../05-training-methodology/curriculum-and-data-strategies.md)) is valuable in supervised/self-supervised settings — both are strategies for getting more learning signal than raw, real, human/environment-provided data alone would offer, with an analogous risk (model-generated data or model-generated experience compounding a model's existing errors) that this repository flags in both places.

## Sources

- Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (2019, published 2020)
- Silver et al., "Mastering the game of Go without human knowledge" (2017) [AlphaGo Zero, direct predecessor context]
