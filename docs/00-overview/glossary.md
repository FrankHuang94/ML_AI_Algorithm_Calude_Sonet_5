# Glossary

A living, alphabetized list of every jargon term used in this repository. Each entry gets a one-line definition and a link to the file where it's covered in depth (if any single file "owns" it). This file is appended to throughout the build — if you find a term used in a doc file that isn't listed here, that's a bug; please add it.

## How to use this

Files in this repository define jargon inline on first use with a short parenthetical, per the audience calibration in [scope-and-methodology.md](scope-and-methodology.md). This glossary exists so you don't have to hunt through files to re-find a definition, and so terminology stays consistent repo-wide (the same concept should never have two different names in two different files).

---

## A

- **Ablation** — an experiment where you remove or disable one component of a system (a layer, a loss term, a training trick) and re-measure performance, to find out how much that component actually contributed. If removing it doesn't hurt performance, it wasn't pulling its weight. Covered in: [scope-and-methodology.md](scope-and-methodology.md).

## B

- **Benchmark** — a standardized dataset/task pair (plus a scoring rule) used to compare models against each other under identical conditions. "SOTA on benchmark X" means "best published score on that specific standardized test," which is narrower than "best model overall." Covered in: [scope-and-methodology.md](scope-and-methodology.md).
- **Bias-variance tradeoff** — the tension between systematic error from a model's assumptions being wrong (bias) and sensitivity to the particular training sample used (variance); ensembles like bagging primarily reduce variance. Covered in: [ensemble-methods.md](../02-classical-ml/ensemble-methods.md).

## C

- **Calibration** — how well a model's stated confidence (predicted probability) matches its actual empirical accuracy. A well-calibrated model that says "80% confident" is right about 80% of the time. Covered in: [regularization-techniques.md](../01-foundations/regularization-techniques.md).
- **Cross-entropy** — a loss function measuring the gap between a predicted probability distribution and the true target distribution; equals the true distribution's entropy plus the KL divergence between true and predicted. The standard loss for classification and language modeling. Covered in: [loss-functions.md](../01-foundations/loss-functions.md).

## D

- **Data leakage** — when information about the target variable inadvertently leaks into features during preprocessing, inflating training/validation performance in a way that doesn't generalize. Covered in: [ensemble-methods.md](../02-classical-ml/ensemble-methods.md).

## E

- **Embedding** — a dense numerical vector representing an input (a word, image, user, etc.) in a learned space where geometric distance corresponds to semantic similarity. Covered in: [loss-functions.md](../01-foundations/loss-functions.md) (contrastive losses); see also [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md).
- **Entropy** — a measure of the inherent uncertainty (or information content) in a probability distribution; zero for a fully predictable outcome, maximal when all outcomes are equally likely. Covered in: [loss-functions.md](../01-foundations/loss-functions.md).

## F

- **Few-shot (learning/prompting)** — giving a model a handful (typically 1-100) of example input/output pairs at inference time (in the prompt, not via weight updates) and expecting it to generalize the pattern to a new input. Contrast with zero-shot. Covered in: [scope-and-methodology.md](scope-and-methodology.md).

## G

- **Generalization** — how well a model performs on data it did not train on, as opposed to how well it merely memorized its training set. The entire point of regularization is to improve this. Covered in: [regularization-techniques.md](../01-foundations/regularization-techniques.md).
- **Gradient** — the vector of partial derivatives of the loss function with respect to every model parameter; points in the direction of steepest increase of the loss, so training steps move opposite to it. Covered in: [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).
- **Gradient clipping** — rescaling the gradient vector so its norm never exceeds a fixed threshold, to prevent a single abnormally large gradient from destabilizing training. Covered in: [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).
- **Gradient explosion** — a failure mode where gradients grow multiplicatively as they're backpropagated through many layers or time steps, producing enormous, destabilizing updates. Contrast with vanishing gradients. Covered in: [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).
- **Gradient vanishing** — the opposite failure mode from gradient explosion: gradients shrink toward zero as they propagate backward through many layers/time steps, so early layers stop learning. Covered in: [rnn-lstm-gru.md](../03-deep-learning-architectures/rnn-lstm-gru.md).

## H

- **Hessian** — the matrix of second derivatives of a function; describes the local curvature of the loss landscape. Too large (parameters²) to compute directly for deep networks, which is why most training uses only first-derivative (gradient) information. Covered in: [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).

## I

- **Inductive bias** — a built-in assumption baked into a model architecture (rather than learned from data) that makes certain patterns easier to learn; e.g., convolution's assumption that nearby pixels are related. Covered in: [cnn-family.md](../03-deep-learning-architectures/cnn-family.md).

## K

- **Kernel trick** — a mathematical shortcut letting an algorithm (classically, SVMs) operate as if data had been mapped into a much higher-dimensional space, without ever computing that mapping explicitly, by computing dot products directly via a kernel function. Covered in: [supervised-learning.md](../02-classical-ml/supervised-learning.md).
- **KL divergence (Kullback-Leibler divergence)** — a measure of how different one probability distribution is from another (not symmetric); the standard tool for penalizing a distribution for drifting away from a reference distribution. Covered in: [loss-functions.md](../01-foundations/loss-functions.md).

## L

- **Latent space / latent representation** — a learned, compressed vector representation of data (e.g., an autoencoder's bottleneck layer) capturing its important underlying factors of variation, as opposed to its raw input form. Covered in: [unsupervised-learning.md](../02-classical-ml/unsupervised-learning.md); see also [vaes.md](../04-generative-models/vaes.md).
- **Learning rate (η)** — the scalar step size controlling how far a single optimization step moves the parameters along the (negative) gradient direction. Covered in: [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).
- **Logits** — the raw, unnormalized scores a classifier produces before they're converted into probabilities (typically via softmax). Covered in: [loss-functions.md](../01-foundations/loss-functions.md).
- **Loss landscape** — the high-dimensional surface formed by plotting the loss function's value against every possible setting of the model's parameters; training is the process of finding a low point on this surface. Covered in: [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).

## M

- **Markov property** — the assumption that the future depends on the past only through the present state, not the full history; the defining assumption of Markov chains, HMMs, and MDPs. Covered in: [probabilistic-models.md](../02-classical-ml/probabilistic-models.md); see also [value-based-methods.md](../07-reinforcement-learning/value-based-methods.md).
- **Momentum** — an optimization technique that accumulates a running average of past gradients and steps in that averaged direction rather than the raw current gradient, smoothing the trajectory. Covered in: [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).

## O

- **Overfitting** — when a model fits the training data (including its noise/quirks) so closely that it performs worse on new, unseen data than a less-fitted model would. Contrast with underfitting. Covered in: [regularization-techniques.md](../01-foundations/regularization-techniques.md).

## P

- **Perplexity** — a human-interpretable transform of a language model's cross-entropy loss (e^loss); roughly, "the model is as uncertain as if choosing uniformly among this many options per token." Lower is better. Covered in: [loss-functions.md](../01-foundations/loss-functions.md).

## R

- **Receptive field** — the region of the original input that a given neuron's output is effectively influenced by; grows larger in deeper layers of a CNN. Covered in: [cnn-family.md](../03-deep-learning-architectures/cnn-family.md).
- **ReLU (Rectified Linear Unit)** — the activation function max(0, x); outputs zero for negative inputs and passes positive inputs through unchanged. Standard in CNNs and widely used elsewhere because its gradient doesn't shrink for positive inputs. Covered in: [cnn-family.md](../03-deep-learning-architectures/cnn-family.md).

## S

- **SOTA (State of the Art)** — the best publicly reported result on a given benchmark at a given point in time. A moving target, not a fixed method — "SOTA" describes a leaderboard position, not any one algorithm. Covered in: [scope-and-methodology.md](scope-and-methodology.md).
- **Softmax** — a function converting a vector of raw scores (logits) into a valid probability distribution (all positive, summing to 1) by exponentiating and normalizing. Covered in: [loss-functions.md](../01-foundations/loss-functions.md).

## U

- **Underfitting** — when a model is too simple or undertrained to capture the real patterns in the training data, performing poorly on both training and new data. Contrast with overfitting. Covered in: [regularization-techniques.md](../01-foundations/regularization-techniques.md).

## Z

- **Zero-shot (learning/prompting)** — asking a model to perform a task it was never given explicit examples of at inference time, relying entirely on what it learned during training/pretraining. Contrast with few-shot. Covered in: [scope-and-methodology.md](scope-and-methodology.md).

---

*(Additional terms are appended here as each subsequent file in the build is written.)*
