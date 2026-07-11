# Glossary

A living, alphabetized list of every jargon term used in this repository. Each entry gets a one-line definition and a link to the file where it's covered in depth (if any single file "owns" it). This file is appended to throughout the build — if you find a term used in a doc file that isn't listed here, that's a bug; please add it.

## How to use this

Files in this repository define jargon inline on first use with a short parenthetical, per the audience calibration in [scope-and-methodology.md](scope-and-methodology.md). This glossary exists so you don't have to hunt through files to re-find a definition, and so terminology stays consistent repo-wide (the same concept should never have two different names in two different files).

---

## A

- **Ablation** — an experiment where you remove or disable one component of a system (a layer, a loss term, a training trick) and re-measure performance, to find out how much that component actually contributed. If removing it doesn't hurt performance, it wasn't pulling its weight. Covered in: [scope-and-methodology.md](scope-and-methodology.md).
- **Alignment** — training a model to behave in accordance with human intentions and values (helpful, honest, harmless), as opposed to merely predicting statistically likely text. Covered in: [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md).
- **All-reduce** — a collective communication operation where every device in a distributed training job ends up with the combined (e.g., summed/averaged) value of some quantity (typically gradients) computed across all devices. Covered in: [distributed-training.md](../05-training-methodology/distributed-training.md).
- **Attention** — a mechanism that computes relevance scores between a query and a set of candidate items, converts them to weights (typically via softmax), and takes a weighted sum of the candidates; originated in seq2seq translation (Bahdanau/Luong attention) and generalized into Transformer self-attention. Covered in: [rnn-lstm-gru.md](../03-deep-learning-architectures/rnn-lstm-gru.md); see also [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md).

## B

- **Benchmark** — a standardized dataset/task pair (plus a scoring rule) used to compare models against each other under identical conditions. "SOTA on benchmark X" means "best published score on that specific standardized test," which is narrower than "best model overall." Covered in: [scope-and-methodology.md](scope-and-methodology.md).
- **Bias-variance tradeoff** — the tension between systematic error from a model's assumptions being wrong (bias) and sensitivity to the particular training sample used (variance); ensembles like bagging primarily reduce variance. Covered in: [ensemble-methods.md](../02-classical-ml/ensemble-methods.md).

## C

- **Calibration** — how well a model's stated confidence (predicted probability) matches its actual empirical accuracy. A well-calibrated model that says "80% confident" is right about 80% of the time. Covered in: [regularization-techniques.md](../01-foundations/regularization-techniques.md).
- **Causal masking** — in self-attention, blocking each position from attending to later positions (by setting their scores to negative infinity before softmax) so a model can't "see the future" it's supposed to predict. Covered in: [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md).
- **Compute-optimal** — the allocation of a fixed training compute budget between model size and data size that minimizes loss; the Chinchilla paper's central finding. Covered in: [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md).
- **Cross-attention** — an attention variant where queries come from one sequence (e.g., a decoder) and keys/values come from another (e.g., an encoder's output), letting one sequence look back at another. Covered in: [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md).
- **Cross-entropy** — a loss function measuring the gap between a predicted probability distribution and the true target distribution; equals the true distribution's entropy plus the KL divergence between true and predicted. The standard loss for classification and language modeling. Covered in: [loss-functions.md](../01-foundations/loss-functions.md).

## D

- **Data leakage** — when information about the target variable inadvertently leaks into features during preprocessing, inflating training/validation performance in a way that doesn't generalize. Covered in: [ensemble-methods.md](../02-classical-ml/ensemble-methods.md).
- **Degeneration** — a failure mode of likelihood-maximizing text decoding (greedy/beam search) producing repetitive, generic, or looping output rather than varied, human-like text. Covered in: [autoregressive-generation.md](../04-generative-models/autoregressive-generation.md).

## E

- **ELBO (Evidence Lower BOund)** — a computable lower bound on a generative model's (otherwise intractable) data likelihood, consisting of a reconstruction term and a KL-divergence regularization term; the training objective for VAEs. Covered in: [vaes.md](../04-generative-models/vaes.md).
- **𝔼 (expectation)** — the average value of a quantity, weighted by how likely each outcome is; standard notation in ML papers for "averaged over the data" or "averaged over samples from a distribution." Covered in: [gans.md](../04-generative-models/gans.md).
- **Embedding** — a dense numerical vector representing an input (a word, image, user, etc.) in a learned space where geometric distance corresponds to semantic similarity. Covered in: [loss-functions.md](../01-foundations/loss-functions.md) (contrastive losses); see also [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md).
- **Entropy** — a measure of the inherent uncertainty (or information content) in a probability distribution; zero for a fully predictable outcome, maximal when all outcomes are equally likely. Covered in: [loss-functions.md](../01-foundations/loss-functions.md).

## F

- **Few-shot (learning/prompting)** — giving a model a handful (typically 1-100) of example input/output pairs at inference time (in the prompt, not via weight updates) and expecting it to generalize the pattern to a new input. Contrast with zero-shot. Covered in: [scope-and-methodology.md](scope-and-methodology.md).
- **FLOPs (floating-point operations)** — a measure of computational cost; used to quantify how much compute a model requires to process a token, train, or run inference. Covered in: [mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md).

## G

- **Generalization** — how well a model performs on data it did not train on, as opposed to how well it merely memorized its training set. The entire point of regularization is to improve this. Covered in: [regularization-techniques.md](../01-foundations/regularization-techniques.md).
- **Guidance scale** — in diffusion models, a tunable strength parameter controlling how strongly generation is pushed toward a conditioning signal (e.g., a text prompt) via classifier-free guidance. Covered in: [diffusion-models.md](../04-generative-models/diffusion-models.md).
- **Gradient** — the vector of partial derivatives of the loss function with respect to every model parameter; points in the direction of steepest increase of the loss, so training steps move opposite to it. Covered in: [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).
- **Gradient clipping** — rescaling the gradient vector so its norm never exceeds a fixed threshold, to prevent a single abnormally large gradient from destabilizing training. Covered in: [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).
- **Gradient explosion** — a failure mode where gradients grow multiplicatively as they're backpropagated through many layers or time steps, producing enormous, destabilizing updates. Contrast with vanishing gradients. Covered in: [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).
- **Gradient vanishing** — the opposite failure mode from gradient explosion: gradients shrink toward zero as they propagate backward through many layers/time steps, so early layers stop learning. Covered in: [rnn-lstm-gru.md](../03-deep-learning-architectures/rnn-lstm-gru.md).

## H

- **Hidden state** — a vector maintained and updated by a recurrent network at each time step, summarizing everything relevant the network has seen so far in a sequence. Covered in: [rnn-lstm-gru.md](../03-deep-learning-architectures/rnn-lstm-gru.md).
- **Hessian** — the matrix of second derivatives of a function; describes the local curvature of the loss landscape. Too large (parameters²) to compute directly for deep networks, which is why most training uses only first-derivative (gradient) information. Covered in: [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).

## I

- **Inductive vs. transductive learning** — inductive models generalize to new, unseen nodes/graphs/examples; transductive models are defined only over the fixed data seen during training and don't naturally extend beyond it. Covered in: [graph-neural-networks.md](../03-deep-learning-architectures/graph-neural-networks.md).
- **Inductive bias** — a built-in assumption baked into a model architecture (rather than learned from data) that makes certain patterns easier to learn; e.g., convolution's assumption that nearby pixels are related. Covered in: [cnn-family.md](../03-deep-learning-architectures/cnn-family.md).

## K

- **Kernel trick** — a mathematical shortcut letting an algorithm (classically, SVMs) operate as if data had been mapped into a much higher-dimensional space, without ever computing that mapping explicitly, by computing dot products directly via a kernel function. Covered in: [supervised-learning.md](../02-classical-ml/supervised-learning.md).
- **KL divergence (Kullback-Leibler divergence)** — a measure of how different one probability distribution is from another (not symmetric); the standard tool for penalizing a distribution for drifting away from a reference distribution. Covered in: [loss-functions.md](../01-foundations/loss-functions.md).

## L

- **Latent space / latent representation** — a learned, compressed vector representation of data (e.g., an autoencoder's bottleneck layer) capturing its important underlying factors of variation, as opposed to its raw input form. Covered in: [unsupervised-learning.md](../02-classical-ml/unsupervised-learning.md); see also [vaes.md](../04-generative-models/vaes.md).
- **Learning rate (η)** — the scalar step size controlling how far a single optimization step moves the parameters along the (negative) gradient direction. Covered in: [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).
- **Logits** — the raw, unnormalized scores a classifier produces before they're converted into probabilities (typically via softmax). Covered in: [loss-functions.md](../01-foundations/loss-functions.md).
- **Loss landscape** — the high-dimensional surface formed by plotting the loss function's value against every possible setting of the model's parameters; training is the process of finding a low point on this surface. Covered in: [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).
- **Low-rank decomposition** — approximating a large matrix update as the product of two much smaller (thin) matrices, drastically reducing trainable parameters; the mechanism behind LoRA. Covered in: [finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md).

## M

- **Markov property** — the assumption that the future depends on the past only through the present state, not the full history; the defining assumption of Markov chains, HMMs, and MDPs. Covered in: [probabilistic-models.md](../02-classical-ml/probabilistic-models.md); see also [value-based-methods.md](../07-reinforcement-learning/value-based-methods.md).
- **Minimax game/objective** — an optimization setup where two parties have directly opposing goals (one maximizes, one minimizes the same expression); the training framework behind GANs. Covered in: [gans.md](../04-generative-models/gans.md).
- **Mode collapse** — a GAN failure mode where the generator produces only a small, non-diverse set of outputs that happen to fool the current discriminator, rather than covering the full diversity of the true data distribution. Covered in: [gans.md](../04-generative-models/gans.md).
- **Mixture of Experts (MoE) / router** — an architectural pattern where many parallel sub-networks ("experts") exist, but a small router network selects only a few to process each token, decoupling total parameter count from per-token compute cost. Covered in: [mixture-of-experts.md](../03-deep-learning-architectures/mixture-of-experts.md).
- **Momentum** — an optimization technique that accumulates a running average of past gradients and steps in that averaged direction rather than the raw current gradient, smoothing the trajectory. Covered in: [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).

## O

- **Overfitting** — when a model fits the training data (including its noise/quirks) so closely that it performs worse on new, unseen data than a less-fitted model would. Contrast with underfitting. Covered in: [regularization-techniques.md](../01-foundations/regularization-techniques.md).

## P

- **PEFT (Parameter-Efficient Fine-Tuning)** — an umbrella term for fine-tuning methods (LoRA, adapters, prefix/prompt tuning) that freeze most of a pretrained model's weights and train only a small number of additional/modified parameters. Covered in: [finetuning-and-peft.md](../05-training-methodology/finetuning-and-peft.md).
- **Perplexity** — a human-interpretable transform of a language model's cross-entropy loss (e^loss); roughly, "the model is as uncertain as if choosing uniformly among this many options per token." Lower is better. Covered in: [loss-functions.md](../01-foundations/loss-functions.md).
- **Policy** — in reinforcement learning, the model/function being trained to choose actions (e.g., which token to generate); RLHF's RL stage optimizes the language model as a policy. Covered in: [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md); see also [value-based-methods.md](../07-reinforcement-learning/value-based-methods.md).
- **Prior** — an assumed probability distribution over a variable before observing data (e.g., a VAE's assumed standard-normal distribution over latent vectors), used as a reference/target that inference is regularized toward. Covered in: [vaes.md](../04-generative-models/vaes.md).

## Q

- **Query, Key, Value (Q/K/V)** — the three learned projections of each position's input in self-attention: the query represents what a position is looking for, the key represents what a position advertises, and the value represents the information a position offers if attended to. Covered in: [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md).

## R

- **Receptive field** — the region of the original input that a given neuron's output is effectively influenced by; grows larger in deeper layers of a CNN. Covered in: [cnn-family.md](../03-deep-learning-architectures/cnn-family.md).
- **ReLU (Rectified Linear Unit)** — the activation function max(0, x); outputs zero for negative inputs and passes positive inputs through unchanged. Standard in CNNs and widely used elsewhere because its gradient doesn't shrink for positive inputs. Covered in: [cnn-family.md](../03-deep-learning-architectures/cnn-family.md).
- **Reparameterization trick** — rewriting a random sample as a deterministic function of learned parameters plus fixed external randomness, so gradients can backpropagate through an otherwise non-differentiable sampling step. Covered in: [vaes.md](../04-generative-models/vaes.md).
- **Reward hacking** — when a policy exploits imperfections in a learned reward model to score highly without actually producing what a human would judge as good output. Covered in: [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md).
- **Reward model** — a model trained to predict human preference judgments, used as a fast automated proxy for human feedback during RL training. Covered in: [rlhf-and-alignment.md](../05-training-methodology/rlhf-and-alignment.md).

## S

- **Scaling laws** — empirical power-law relationships between a language model's loss and its parameter count, dataset size, and training compute, used to plan large training runs. Covered in: [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md).
- **Score function** — the gradient of the log-probability of a data distribution with respect to the data itself; learning this at multiple noise levels is the basis of score-based generative modeling, closely related to DDPM's noise-prediction objective. Covered in: [diffusion-models.md](../04-generative-models/diffusion-models.md).
- **Self-supervised learning** — training with labels automatically derived from the input data itself (e.g., predicting a masked or next token), rather than requiring human annotation. Covered in: [pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md).
- **Sharding** — splitting a large object (model parameters, gradients, optimizer state) into pieces distributed across multiple devices, so no single device needs to hold the whole thing. Covered in: [distributed-training.md](../05-training-methodology/distributed-training.md).
- **SOTA (State of the Art)** — the best publicly reported result on a given benchmark at a given point in time. A moving target, not a fixed method — "SOTA" describes a leaderboard position, not any one algorithm. Covered in: [scope-and-methodology.md](scope-and-methodology.md).
- **Softmax** — a function converting a vector of raw scores (logits) into a valid probability distribution (all positive, summing to 1) by exponentiating and normalizing. Covered in: [loss-functions.md](../01-foundations/loss-functions.md).

## T

- **Temperature** — a scalar dividing logits before softmax during sampling; below 1 sharpens (more deterministic) a model's output distribution, above 1 flattens it (more random/diverse). Covered in: [autoregressive-generation.md](../04-generative-models/autoregressive-generation.md).
- **Token** — a chunk of text (often a word or sub-word piece) that a language model treats as a single unit of input/output. Covered in: [autoregressive-generation.md](../04-generative-models/autoregressive-generation.md).

## U

- **Underfitting** — when a model is too simple or undertrained to capture the real patterns in the training data, performing poorly on both training and new data. Contrast with overfitting. Covered in: [regularization-techniques.md](../01-foundations/regularization-techniques.md).

## Z

- **Zero-shot (learning/prompting)** — asking a model to perform a task it was never given explicit examples of at inference time, relying entirely on what it learned during training/pretraining. Contrast with few-shot. Covered in: [scope-and-methodology.md](scope-and-methodology.md).

---

*(Additional terms are appended here as each subsequent file in the build is written.)*
