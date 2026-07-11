# Loss Functions

A loss function is the single number a model is trained to minimize — it quantifies "how wrong" the model's current predictions are, averaged over training examples, so that the optimization algorithms in [optimization-algorithms.md](optimization-algorithms.md) have something concrete to reduce. Choice of loss function determines what "good performance" even means for a given task, so getting it right (or wrong) shapes everything downstream. This file also briefly covers **entropy** and **KL divergence**, foundational information-theory concepts that recur constantly elsewhere in this repository.

## MSE / L2 loss

**Definition.** Mean Squared Error averages the squared difference between predicted and true values.

```
L = (1/N) · Σᵢ (yᵢ − ŷᵢ)²
```

Walkthrough: for every example i, take the difference between the true value yᵢ and the prediction ŷᵢ, square it (so errors in either direction count as positive, and larger errors are penalized disproportionately more than small ones), then average over all N examples. Squaring means a single large miss is penalized much more heavily than several small ones of the same total magnitude — MSE is sensitive to outliers.

**Where it's used.** The standard loss for regression tasks (predicting a continuous number) — house price prediction, forecasting, and as a per-pixel reconstruction loss in some image models. Rare as a primary loss for classification or language modeling.

## MAE / L1 loss

**Definition.** Mean Absolute Error averages the absolute (unsquared) difference between predicted and true values: L = (1/N)·Σᵢ|yᵢ − ŷᵢ|.

**Why it differs from MSE.** Because errors aren't squared, MAE penalizes a large single error proportionally to its size rather than its square — it's far less sensitive to outliers than MSE, at the cost of a less smooth gradient near zero error (its derivative is a constant ±1 regardless of error size, rather than proportional to the error). Used when robustness to outliers matters more than penalizing big misses heavily.

## Cross-entropy loss

**Definition.** Cross-entropy loss measures the difference between a predicted probability distribution and the true (target) distribution, and is the standard loss for classification.

**Background: entropy.** Before cross-entropy makes sense, it helps to have a one-sentence grip on **entropy** from information theory: entropy measures how much uncertainty (or, equivalently, how many bits of information) is inherent in a probability distribution. A coin that always lands heads has zero entropy (no uncertainty — you always know the outcome); a fair coin has maximum entropy for a two-outcome distribution (you learn the most from observing each flip because you were maximally uncertain beforehand). Formally, entropy H(p) = −Σₓ p(x)·log(p(x)).

**Binary cross-entropy** (two classes, e.g., "spam" vs. "not spam"):

```
L = −(1/N) · Σᵢ [yᵢ·log(ŷᵢ) + (1 − yᵢ)·log(1 − ŷᵢ)]
```

where yᵢ ∈ {0, 1} is the true label and ŷᵢ ∈ (0, 1) is the model's predicted probability of class 1. Walkthrough: for each example, only one of the two terms is "active" depending on the true label (if yᵢ = 1, the second term vanishes; if yᵢ = 0, the first term vanishes) — the loss is simply −log(predicted probability of the correct class). Since log of a number close to 1 is close to 0, and log of a number close to 0 goes to negative infinity, this loss is near zero when the model confidently predicts the right answer and grows explosively as the model confidently predicts the *wrong* answer. That asymmetric, unbounded penalty for confident wrongness is a deliberate design feature: it aggressively punishes overconfident mistakes.

**Categorical cross-entropy** (more than two classes) generalizes this using the **softmax function**, which converts a vector of raw scores (**logits**) into a valid probability distribution (all values positive, summing to 1):

```
softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ
L = −(1/N) · Σᵢ log(ŷᵢ,correct_class)
```

Walkthrough: softmax exponentiates every logit (making everything positive and exaggerating the gaps between large and small scores) and normalizes by the sum, producing a probability for each class. The loss is then just the negative log of the probability the model assigned to the correct class — the same "confidently right, near-zero loss; confidently wrong, huge loss" shape as the binary case, generalized to many classes.

**Why it mattered / current status.** Cross-entropy is the default loss for essentially all classification tasks, and — critically for this repository — it is also the loss used to train language models on next-token prediction (predicting which token comes next is a classification problem over the vocabulary; see [autoregressive-generation.md](../04-generative-models/autoregressive-generation.md)). It remains completely dominant as of 2026.

## Hinge loss (SVM)

**Definition.** Hinge loss is the loss function used to train Support Vector Machines (see [supervised-learning.md](../02-classical-ml/supervised-learning.md)): L = max(0, 1 − y·f(x)), where y ∈ {−1, +1} is the true label and f(x) is the model's raw (unbounded) output score.

Walkthrough: if the model's prediction has the correct sign *and* is confidently past a margin of 1 (i.e., y·f(x) ≥ 1), the loss is exactly zero — there's no further reward for being even more confident. If the prediction is within the margin or wrong, the loss grows linearly. This "flat once you're safely correct" shape is different from cross-entropy, which keeps rewarding confidence indefinitely — it directly encodes the SVM's "maximum margin" objective (find a decision boundary with as much separation as possible between classes, and don't bother optimizing further once that's achieved).

**Current status.** Central to the classical SVM story; rarely used to train deep networks today, where cross-entropy dominates classification.

## KL divergence

**Definition.** Kullback-Leibler (KL) divergence measures how different one probability distribution is from another — specifically, how much "extra surprise" you incur by using distribution Q to describe data that actually comes from distribution P.

```
D_KL(P ‖ Q) = Σₓ P(x) · log(P(x) / Q(x))
```

Walkthrough: for each possible outcome x, weight the log-ratio of the two distributions' probabilities by how likely that outcome actually is under the true distribution P. If P and Q are identical, every log-ratio term is log(1) = 0, so KL divergence is exactly zero. The more Q's probabilities diverge from P's (especially by assigning low probability to outcomes P considers likely), the larger the value. KL divergence is **not symmetric** — D_KL(P‖Q) ≠ D_KL(Q‖P) in general — which matters when choosing which distribution is the "reference" (P) and which is the one being fit (Q).

**Why this recurs so much elsewhere in this repository.** KL divergence shows up repeatedly: as the regularization term in the VAE's ELBO objective (see [vaes.md](../04-generative-models/vaes.md)), as the constraint in TRPO and the implicit constraint in PPO (see [policy-gradient-methods.md](../07-reinforcement-learning/policy-gradient-methods.md)), and as the term keeping an RLHF-tuned model's outputs from drifting too far from its pretrained base model (see [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md)). It's worth internalizing once here rather than re-deriving in every file: KL divergence is the standard way to penalize "distribution Q has drifted too far from reference distribution P."

**Relationship to cross-entropy.** Cross-entropy H(P, Q) = H(P) + D_KL(P‖Q) — cross-entropy is entropy of the true distribution plus the KL divergence between true and predicted. Since H(P) is fixed (it doesn't depend on the model), minimizing cross-entropy with respect to the model's predictions is mathematically equivalent to minimizing KL divergence between the true and predicted distributions. This is why cross-entropy loss can be understood either as "penalize wrong predictions directly" or as "make the predicted distribution match the true distribution as closely as possible" — they're the same objective.

## Contrastive losses: InfoNCE and triplet loss

**Definition.** Contrastive losses train a model to produce representations (**embeddings** — dense numerical vectors that represent an input in a learned space where geometric distance corresponds to semantic similarity) that are close together for "similar" pairs of inputs and far apart for "dissimilar" pairs, rather than directly predicting a label.

**Triplet loss** (Schroff et al., 2015, FaceNet) takes an anchor example, a positive example (same class as anchor), and a negative example (different class), and pushes the anchor-positive distance down while pushing the anchor-negative distance up, with a margin: L = max(0, d(anchor, positive) − d(anchor, negative) + margin).

**InfoNCE** (van den Oord et al., 2018, in the context of Contrastive Predictive Coding) generalizes this to many negatives at once, using a softmax-like formulation over one positive pair against a batch of negative pairs:

```
L = −log[ exp(sim(a, p)/τ) / Σⱼ exp(sim(a, nⱼ)/τ) ]
```

where sim is a similarity function (commonly cosine similarity) and τ (tau) is a temperature parameter controlling how sharply the softmax distinguishes close calls. Walkthrough: this is exactly the categorical cross-entropy formula, but with "classes" replaced by "which of these candidate examples is the true match" — the model is trained as if it were solving a multiple-choice problem where the right answer is the true positive pair and the distractors are the negatives in the batch.

**Where it's used today.** InfoNCE-style contrastive losses are the training objective behind CLIP-style image-text alignment (see [pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md)) and many modern embedding models used for search and retrieval.

## Focal loss

**Definition.** Focal loss is a modification of cross-entropy designed for severe class imbalance (when one class vastly outnumbers another in the training data, e.g., in object detection where "background, no object here" vastly outnumbers "object present").

**Origin.** Lin et al., "Focal Loss for Dense Object Detection" (2017, introduced alongside the RetinaNet detector).

**Core mechanism.** Focal loss multiplies the standard cross-entropy term by a factor that down-weights easy, already-well-classified examples: L = −(1 − ŷ)^γ · log(ŷ), where ŷ is the predicted probability of the true class and γ (gamma, typically ~2) controls how aggressively easy examples are down-weighted.

Walkthrough: when ŷ is already close to 1 (an easy, correctly-classified example), (1 − ŷ)^γ is close to zero, so that example contributes almost nothing to the loss even though standard cross-entropy would still assign it some (small) loss. This redirects the training signal toward the harder, misclassified, or minority-class examples that actually need it, preventing the overwhelming volume of easy majority-class examples from drowning out the learning signal from rare but important ones.

**Current status.** Standard in object detection pipelines with severe foreground/background imbalance; less commonly needed elsewhere.

## Perplexity

**Definition.** Perplexity is a standard way of reporting a language model's cross-entropy loss in a more human-interpretable form: perplexity = e^(cross-entropy loss) (using the natural-log version of cross-entropy).

Walkthrough: cross-entropy loss for a language model is measured in "nats" (or bits, if using log base 2) per token — an abstract unit that's hard to build intuition for. Perplexity converts this into "the model is, on average, as uncertain as if it were choosing uniformly among this many equally likely next tokens." A perplexity of 1 means the model is perfectly certain of every next token (impossible in practice for real language); a perplexity of, say, 20 means the model's average uncertainty over the next token is comparable to picking uniformly among 20 options at each step. Lower perplexity means a better-fitting language model, all else equal.

**Current status.** Still widely reported as a language-model quality metric, though the field has increasingly supplemented (and for many purposes displaced) perplexity with downstream task benchmarks and human/model-judged evaluations, since low perplexity doesn't automatically mean a model is good at instruction-following, reasoning, or the specific behaviors users care about.

## Summary table

| Loss | Task type | Sensitive to outliers? | Typical use |
|---|---|---|---|
| MSE / L2 | Regression | Yes (squares errors) | Continuous value prediction |
| MAE / L1 | Regression | No | Robust regression |
| Binary/categorical cross-entropy | Classification | N/A (probability-based) | Classification, language modeling |
| Hinge loss | Classification (margin-based) | Moderate | SVMs |
| KL divergence | Distribution matching | N/A | VAEs, RLHF/PPO drift constraints, distillation |
| InfoNCE / triplet loss | Representation learning | N/A | Embeddings, contrastive pretraining (e.g., CLIP) |
| Focal loss | Classification (imbalanced) | Down-weights easy examples by design | Object detection |

## Relationship to other algorithms

- Cross-entropy is the loss minimized by the optimizers in [optimization-algorithms.md](optimization-algorithms.md).
- KL divergence appears again in [vaes.md](../04-generative-models/vaes.md) (ELBO), [policy-gradient-methods.md](../07-reinforcement-learning/policy-gradient-methods.md) (TRPO/PPO), and [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md) (drift penalty).
- Label smoothing (see [regularization-techniques.md](regularization-techniques.md)) directly modifies the targets fed into cross-entropy loss.
- Perplexity connects directly to autoregressive language model training — see [autoregressive-generation.md](../04-generative-models/autoregressive-generation.md).

## Sources

- Schroff, Kalenichenko, Philbin, "FaceNet: A Unified Embedding for Face Recognition and Clustering" (2015)
- van den Oord, Li, Vinyals, "Representation Learning with Contrastive Predictive Coding" (2018)
- Lin, Goyal, Girshick, He, Dollár, "Focal Loss for Dense Object Detection" (2017)
- Kullback, Leibler, "On Information and Sufficiency" (1951)
