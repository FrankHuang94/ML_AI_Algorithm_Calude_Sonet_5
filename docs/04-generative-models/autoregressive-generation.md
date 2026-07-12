# Autoregressive Generation

Autoregressive generation is the training and sampling paradigm behind essentially every large language model in production today. This file covers the next-token-prediction framing, the major decoding/sampling strategies used at inference time, why this approach specifically dominates text generation, and the degeneration problems that come with it.

## Next-token prediction framing

**Name & definition.** An autoregressive model generates a sequence one element at a time, with each new element predicted based on all the elements generated so far — "autoregressive" meaning the model regresses (predicts) on its own previous outputs.

**Core mechanism.** For text, this means modeling the probability of an entire sequence of tokens (a token is a chunk of text — often a word or sub-word piece — that a language model treats as a single unit) as a product of conditional probabilities, via the chain rule of probability:

```
p(x₁, x₂, ..., x_n) = p(x₁) · p(x₂|x₁) · p(x₃|x₁,x₂) · ... · p(x_n|x₁,...,x_{n-1})
```

Walkthrough: rather than trying to model the probability of an entire sequence all at once (an enormously complex joint distribution), this factors the problem into a sequence of much simpler predictions — "given everything so far, what's the probability distribution over the next token?" — each of which is exactly the classification problem that cross-entropy loss (see [loss-functions.md](../01-foundations/loss-functions.md)) is designed for, with "class" being "which token in the vocabulary comes next." A decoder-only Transformer (see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)) is trained to do exactly this: given a prefix of tokens, output a probability distribution (via softmax over logits) for the next token, using causally-masked self-attention so that the prediction at each position only depends on tokens at or before that position.

**Why this training objective is so powerful.** Every token in every training document simultaneously provides a free training example (predict this token from everything before it) — there's no need for manual labeling, and the sheer quantity of available text data (essentially all of it) can be used directly. This is the training-side half of the story for why decoder-only, autoregressively-trained models became the dominant LLM architecture — see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md) for the fuller "why decoder-only won" discussion, and [pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md) for how this objective is used in practice at scale.

## Sampling strategies at inference time

Once a model can produce a probability distribution over the next token, there's a choice to make about how to actually pick a token from that distribution — this is a separate decision from training, made at inference (generation) time, and different choices produce noticeably different generation behavior from the exact same trained model.

The strategies are easiest to see as different ways of deciding *which slice of the probability distribution is eligible* to be sampled. Say the model, after the prompt "The weather today is", produces this distribution over next tokens:

```
   token:   "sunny" "warm" "cold" "nice" "mild" "rainy"  ... (thousands more, tiny probs)
   prob:     0.40    0.25   0.15   0.10   0.05   0.03     ...
            ████████ █████  ███    ██     █      ▌

   greedy      → takes only "sunny" (the single tallest bar). Deterministic.
   top-k=3     → eligible = {sunny, warm, cold}, then sample among those 3.
   top-p=0.80  → eligible = smallest set summing to ≥0.80 = {sunny, warm, cold} (0.40+0.25+0.15),
                 then sample among those. (If the distribution were flatter, top-p would
                 automatically include MORE tokens — that adaptivity is its whole advantage.)
   temperature → reshapes the bars BEFORE the above: T<1 makes tall bars taller
                 (more like greedy); T>1 flattens them (more surprising choices).
```

Read the whole family off this picture: greedy takes the single tallest bar; top-k keeps a fixed number of bars; top-p (nucleus) keeps however many bars are needed to cover a fixed *probability mass*; temperature rescales the bars' heights before any of that. The rest of this section just expands each of these.

### Greedy decoding

**Mechanism.** Always pick the single highest-probability token at each step.

**Tradeoffs.** Deterministic (same input always produces the same output) and simple, but prone to producing repetitive, generic text, and — because it only ever considers the locally best next token — can miss a better overall sequence that would have required a locally-slightly-worse token at some earlier step (a classic greedy-algorithm limitation, in the standard CS sense).

### Beam search

**Mechanism.** Instead of tracking only the single best partial sequence, maintain the k highest-probability partial sequences ("beams") at each step, expanding each by one token and keeping only the overall best k resulting sequences, repeated until completion.

**Tradeoffs.** Finds higher-probability *overall* sequences than pure greedy decoding by hedging against early locally-suboptimal choices, and remains standard for tasks with a fairly narrow space of "correct" outputs, like machine translation. For open-ended text generation, beam search has a well-documented failure mode: it tends to find sequences that are *too* high-probability in a degenerate way (bland, repetitive, generic text), because the most probable overall sequence according to the model isn't necessarily the most interesting or human-like one.

### Temperature

**Mechanism.** Before converting logits to probabilities via softmax, divide the logits by a temperature value τ: softmax(logits / τ). τ < 1 sharpens the distribution (makes the model more confident/deterministic, exaggerating the gap between likely and unlikely tokens); τ > 1 flattens it (makes the model's choices more random/diverse, since it depresses the relative gap between the most and least likely tokens); τ = 1 leaves the distribution unchanged.

**Tradeoffs.** Temperature is a dial on the diversity/coherence tradeoff, not a sampling strategy on its own — it's typically combined with one of the truncated-sampling strategies below, which restrict *which* tokens are even eligible before temperature is applied.

### Top-k sampling

**Mechanism.** Restrict sampling to only the k highest-probability tokens at each step (discarding the rest and renormalizing their probabilities to sum to 1), then sample randomly from that restricted set (optionally with temperature applied).

**Tradeoffs.** Prevents the model from ever sampling a wildly improbable token (which pure random sampling from the full distribution could occasionally do, producing incoherent output), while still allowing meaningful randomness/diversity among the plausible candidates. The fixed k can be a poor fit across different situations, though — sometimes there are only 2-3 genuinely plausible next tokens (and top-k=40 would let in implausible ones), and sometimes there are dozens of roughly equally good options (and a small fixed k would cut off legitimate variety).

### Nucleus (top-p) sampling

**Origin.** Holtzman et al., "The Curious Case of Neural Text Degeneration" (2019).

**Mechanism.** Instead of a fixed count k, restrict sampling to the smallest set of highest-probability tokens whose cumulative probability adds up to at least p (e.g., p = 0.9 means "keep adding the next most likely token until the kept set covers 90% of the total probability mass," then sample from that dynamically-sized set).

**Why it mattered.** Because the size of the eligible set adapts to how "peaked" or "flat" the model's distribution is at each specific step (a confident prediction with one dominant token gives a small eligible set; an uncertain prediction gives a larger one), nucleus sampling handles both the "only a few good options" and "many roughly-equal options" situations more gracefully than a fixed top-k, and — largely because of this — became, alongside temperature, the standard default sampling configuration for most deployed text-generation systems.

## Why autoregressive decoder-only models dominate text generation specifically

This is worth stating as its own point, distinct from the general "why decoder-only won" discussion in [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md): text is inherently sequential and discrete (a sentence has a definite left-to-right order, and there's a finite vocabulary of tokens to choose from at each position), which is an unusually good match for the autoregressive framing specifically — unlike, say, an image, which has no single "natural" generation order and is continuous-valued (pixel intensities), which is part of why diffusion (see [diffusion-models.md](diffusion-models.md)), not pure autoregression, became dominant for images (though autoregressive image/video generation, predicting one patch or frame at a time, remains an active, non-trivial alternative approach in some systems).

## Repetition and degeneration problems, and mitigations

**The problem.** Purely likelihood-maximizing decoding strategies (greedy decoding, and beam search especially) empirically tend to produce **degenerate** text — repetitive loops (the same phrase or sentence repeated verbatim), generic/bland phrasing, and a general lack of the variety and specificity that characterizes good human-written text. This isn't a bug in any single component; it's a structural consequence of the fact that "the sequence the model assigns highest probability to" and "the sequence a human would judge as best" are related but not identical targets, and greedy/beam-search decoding specifically optimizes for the former.

**Mitigations.**
- **Repetition/frequency penalties** — directly reduce the probability of tokens (or n-grams — sequences of n consecutive tokens) that have already appeared in the generated output, discouraging verbatim loops.
- **Nucleus/top-p sampling with temperature** (above) — by introducing controlled randomness among plausible candidates, these avoid the degenerate high-probability loops that purely greedy/beam-search decoding is prone to.
- **Post-training methods** — going beyond decoding-time fixes, techniques like RLHF (see [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)) directly train the model to prefer the kind of varied, high-quality output humans actually want, rather than only relying on inference-time sampling tricks to paper over a base model's degeneration tendencies.

**Current status.** In practice, essentially all deployed conversational LLMs as of 2026 use some combination of nucleus/top-p sampling (or closely related refinements), a moderate temperature setting, and a model that has already been substantially shaped by post-training (see [05-training-methodology](../05-training-methodology/)) to prefer varied, non-degenerate output — the decoding strategy and the training methodology work together, not as alternatives to each other.

## Comparison table

| Strategy | Deterministic? | Main strength | Main weakness |
|---|---|---|---|
| Greedy decoding | Yes | Simple, fast | Repetitive/generic; no lookahead |
| Beam search | Yes (for fixed beam width) | Finds higher-probability full sequences | Can over-optimize toward bland/degenerate text for open-ended generation |
| Top-k sampling | No | Bounds worst-case token choice | Fixed k doesn't adapt to distribution shape |
| Nucleus (top-p) sampling | No | Adapts eligible set to distribution confidence | Requires tuning p (and usually temperature) per use case |

## Relationship to other algorithms

- Next-token prediction is the training objective implemented by decoder-only Transformers — see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md) and [pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md).
- Cross-entropy loss and perplexity, covered in [loss-functions.md](../01-foundations/loss-functions.md), are the direct training-time and evaluation-time counterparts of this file's inference-time sampling discussion.
- Speculative decoding (see [speculative-decoding.md](../06-inference-optimization/speculative-decoding.md)) is a technique for speeding up autoregressive generation without changing its output distribution — a direct extension of the generation process described here.
- RLHF (see [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)) is the main post-training lever for addressing degeneration at the model level rather than the decoding-strategy level.

## Sources

- Holtzman, Buys, Du, Forbes, Choi, "The Curious Case of Neural Text Degeneration" (2019) [nucleus/top-p sampling]
