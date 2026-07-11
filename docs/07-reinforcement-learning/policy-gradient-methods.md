# Policy Gradient Methods

Policy gradient methods are the second major RL paradigm, alongside the value-based methods covered in [value-based-methods.md](value-based-methods.md) — instead of learning a value function and deriving a policy from it indirectly (e.g., "pick whichever action has the highest Q-value"), policy gradient methods directly learn and optimize the policy itself. This file builds up from REINFORCE through actor-critic methods to PPO, given extra depth here since PPO is the RL algorithm underlying classic RLHF (see [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)).

## REINFORCE

**Name & definition.** REINFORCE directly parameterizes a policy π(a|s; θ) (the probability of taking action a in state s, given parameters θ — for an LLM, this is exactly the model's next-token probability distribution) and updates θ using gradient ascent, in the direction that makes actions which led to high reward more likely.

**Origin.** Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (1992).

**Core mechanism.**

```
∇θ J(θ) = 𝔼 [ ∇θ log π(a|s; θ) · R ]
```

Walkthrough: J(θ) is the expected total reward under the current policy — the thing we actually want to maximize. This formula (the **policy gradient theorem**, which this file states without deriving in full, since the derivation itself requires more probability theory than this repository's audience calibration assumes — see [scope-and-methodology.md](../00-overview/scope-and-methodology.md)) says: to improve the policy, increase the log-probability of actions in proportion to how much total reward R followed from taking them. If an action led to a high reward, increase its probability; if it led to a low (or negative) reward, decrease its probability. This is a genuinely elegant result — it lets you improve a policy using only samples of what happened (states, actions, and the rewards that followed), without needing a model of the environment's dynamics at all.

**Why it's rarely used directly today.** REINFORCE's raw form has a serious practical problem: its gradient estimates have very high **variance** (the specific numerical update computed from any single sampled trajectory can differ wildly from the "true" average update, even though it's correct on average over many samples), which makes training slow and unstable in practice. This directly motivates the actor-critic framing below.

## Actor-critic methods

**Name & definition.** Actor-critic methods reduce policy gradient variance by combining a policy (the **actor**, which decides what action to take) with a learned value function (the **critic**, which estimates how good a given state or action is) — the critic's value estimate is used to judge how much better or worse an action turned out than expected, rather than relying purely on the raw, noisy total reward that followed.

**Core mechanism (conceptual).** Instead of scaling the policy gradient update by the raw total reward R (as in REINFORCE), scale it by an **advantage** estimate A(s, a) = (actual outcome) − (critic's expected-value baseline for this state), which measures specifically how much better or worse this particular action did *relative to what was already expected* in this state, rather than its absolute reward. Walkthrough: if a state is already a great one to be in (high expected value regardless of action), even a modestly good action shouldn't get an enormous positive update — what matters is whether the action did *better than the critic already expected*, not its raw reward in isolation. Subtracting this expected-value baseline substantially reduces the variance of the gradient estimate (a mathematically-justified technique — the expected value of the baseline term's contribution to the gradient is exactly zero, so subtracting it doesn't bias the update, but it does reduce its noise) relative to using the raw, unadjusted reward.

## A3C (Asynchronous Advantage Actor-Critic)

**Origin.** Mnih et al. (DeepMind), "Asynchronous Methods for Deep Reinforcement Learning" (2016).

**Core contribution.** A3C runs many actor-critic agents in parallel, each interacting with its own independent copy of the environment and asynchronously sending gradient updates to a shared set of parameters, rather than running one single agent sequentially. This serves a role somewhat analogous to experience replay's decorrelation benefit in DQN (see [value-based-methods.md](value-based-methods.md)) — many parallel agents naturally generate more diverse, less temporally-correlated experience than a single agent would, which stabilizes training — while also providing a straightforward way to use more compute (many parallel workers) to speed up training.

**Current status.** A3C's core ideas (parallel actor-critic training, advantage-based updates) remain influential, though it has been largely superseded in practice by PPO (below) for most modern applications, including RLHF.

## Trust Region Policy Optimization (TRPO)

**Origin.** Schulman et al., "Trust Region Policy Optimization" (2015).

**The problem it addresses.** A naive policy gradient update can sometimes take a step that's too large — updating the policy so much in one step that its behavior changes drastically and, counterproductively, gets much worse rather than better (unlike supervised learning, a bad policy update in RL directly and immediately affects what data you collect *next*, since the policy determines the actions taken, which can compound instability rather than just causing one bad training step).

**Core mechanism (conceptual).** TRPO constrains each policy update to stay within a "trust region" — formally, a bound on the KL divergence (see [loss-functions.md](../01-foundations/loss-functions.md)) between the new policy and the old policy — so that the policy's behavior can't change too drastically in any single update, even if the raw, unconstrained gradient direction would suggest a large step. This makes training much more stable than naive policy gradients, at the cost of a more mathematically and computationally involved optimization procedure (solving a constrained optimization problem at every update, rather than a simple gradient step).

## Proximal Policy Optimization (PPO)

**Origin.** Schulman et al., "Proximal Policy Optimization Algorithms" (2017).

**The motivation.** PPO keeps TRPO's core goal — don't let the policy change too drastically in a single update — but replaces TRPO's expensive constrained-optimization machinery with a much simpler mechanism that can be implemented using ordinary gradient-based optimization (see [optimization-algorithms.md](../01-foundations/optimization-algorithms.md)), making it dramatically easier to implement and tune while achieving broadly comparable stability and performance in practice.

**Core mechanism — the clipped surrogate objective.**

```
r(θ) = π_new(a|s; θ) / π_old(a|s; θ)
L(θ) = 𝔼 [ min( r(θ) · A,  clip(r(θ), 1−ε, 1+ε) · A ) ]
```

Walkthrough, term by term: r(θ) is the ratio between the new policy's probability of taking a given action and the old policy's probability of the same action — a value near 1 means the policy hasn't changed much for this action, a value far from 1 means it has changed a lot. A is the advantage estimate (as in actor-critic methods above) — positive if the action turned out better than expected, negative if worse. The **clip** function restricts r(θ) to stay within a narrow band around 1 (e.g., [1−ε, 1+ε], with ε commonly around 0.2), and the **min** in the outer expression takes whichever is smaller: the unclipped objective, or the clipped one.

Why this specific construction works: when the advantage A is positive (the action was good, so we want to increase its probability), the min-of-clipped-and-unclipped construction means that once r(θ) grows past 1+ε — i.e., once the policy has already increased this action's probability by "enough" — the clipped term caps further reward from pushing r(θ) even higher, removing the incentive to keep increasing the same action's probability indefinitely in one update. When A is negative (the action was bad), a symmetric logic discourages the policy from decreasing that action's probability by more than the clip range allows in one step. The net effect, without requiring TRPO's explicit KL-divergence constraint machinery, is a "soft" version of the same core idea: don't let one training step change the policy's behavior on any given action too drastically, because reward for doing so is capped once you've moved far enough.

**Why it mattered.** PPO achieves stability roughly comparable to TRPO's, using an objective simple enough to optimize with standard gradient-based methods (no separate constrained-optimization solver needed), which made it dramatically easier to implement, tune, and scale — this combination of stability and simplicity is exactly why PPO became the default RL algorithm choice across a very wide range of applications, very much including the RL stage of classic RLHF (see [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)), where the "old policy" is the model's behavior before an update and the "new policy" is its behavior after, and the advantage estimate is derived from the reward model's score (adjusted by the KL penalty against the reference SFT model, as covered in that file).

## Advantage estimation: GAE (brief)

**Name & definition.** Generalized Advantage Estimation (Schulman et al., 2015 — the same broader research thread as TRPO) provides a principled way to compute the advantage estimate A(s, a) used above, balancing a tradeoff between using purely immediate reward signal (low bias, high variance) and relying more heavily on the critic's own value predictions further into the future (higher bias if the critic's estimates are imperfect, but lower variance) — a single tunable parameter (λ, lambda) smoothly interpolates between these two extremes. GAE is a standard component paired with PPO in most practical implementations, including in typical RLHF pipelines.

## Comparison table

| Method | Year | Key Innovation | Still Relevant (2026)? |
|---|---|---|---|
| REINFORCE | 1992 | Direct policy gradient via log-probability weighting | Foundational theory; high variance limits direct use |
| Actor-critic (general) | — | Critic-based advantage reduces gradient variance | Foundational framing, used throughout modern methods |
| A3C | 2016 | Parallel asynchronous actor-critic training | Influential; largely superseded by PPO in practice |
| TRPO | 2015 | Explicit KL trust-region constraint for stable updates | Foundational; largely superseded by PPO's simpler mechanism |
| PPO | 2017 | Clipped surrogate objective — TRPO-like stability via simple gradient steps | Yes — dominant RL algorithm, including for RLHF |

## Relationship to other algorithms

- PPO is the RL algorithm underlying classic RLHF's policy optimization stage — see [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md) for the full pipeline this fits into.
- The KL-divergence constraint in TRPO (and PPO's softer version of the same idea) uses the same mathematical object covered in [loss-functions.md](../01-foundations/loss-functions.md), and reappears as the drift penalty in RLHF.
- [rl-for-llms.md](rl-for-llms.md) is the dedicated bridge file connecting this general RL theory specifically to LLM training practice.
- Contrast with the value-based methods in [value-based-methods.md](value-based-methods.md) — policy gradient methods learn the policy directly rather than deriving it from a learned value function.

## Sources

- Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (1992)
- Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016) [A3C]
- Schulman, Levine, Abbeel, Jordan, Moritz, "Trust Region Policy Optimization" (2015)
- Schulman, Moritz, Levine, Jordan, Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015) [GAE]
- Schulman, Wolski, Dhariwal, Radford, Klimov, "Proximal Policy Optimization Algorithms" (2017)
