# Diffusion Models

Diffusion models are, as of 2026, the dominant approach to image, video, and audio generation. This file covers the forward/reverse process at the core of the method, the DDPM formulation that made it practical, its connection to score-based generative modeling, the latent diffusion architecture that powers most deployed systems, and the classifier-guidance techniques used to steer generation.

## The forward (noising) and reverse (denoising) process

**The core idea.** Take a real image and gradually corrupt it with a small amount of random (Gaussian) noise, repeated over many steps, until it becomes pure noise indistinguishable from random static. Then train a neural network to reverse this process — starting from pure noise, gradually remove a small amount of noise at a time, over many steps, until a realistic image emerges. If the network has genuinely learned to reverse the noising process step by step, running it starting from freshly-sampled random noise (rather than noise derived from a real image) should produce a brand-new, realistic image.

**Why frame generation this way.** Directly training a network to map random noise straight to a realistic image in one shot is a very hard function to learn (a GAN, see [gans.md](gans.md), attempts something like this, with the training difficulties discussed there). Breaking the problem into many small, easy steps — "remove a little bit of noise from a slightly-noisy image" — is a much easier prediction task for a network to learn well at each individual step, even though it takes many steps (often dozens to a few hundred) to complete a full generation.

## DDPM (Denoising Diffusion Probabilistic Models)

**Origin.** Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models" (2020), building on earlier diffusion-based generative modeling theory (Sohl-Dickstein et al., 2015).

**Core mechanism — the forward process.** Define a fixed (not learned) sequence of T steps, where at each step t, a small amount of Gaussian noise is added to the current (increasingly noisy) image xₜ₋₁ to produce xₜ:

```
xₜ = √(1 − βₜ) · x_{t-1} + √βₜ · ε,     ε ~ N(0, 1)
```

βₜ (a small value, following a fixed schedule across steps) controls how much noise is added at step t. Critically, this process is designed so that x_T (after all T steps) is, for practical purposes, indistinguishable from pure random noise, and — a useful mathematical convenience — xₜ at any step t can be computed directly from the original image x₀ in a single formula, without needing to actually simulate all the intermediate steps one by one during training.

**Core mechanism — the reverse process (what's actually learned).** A neural network εθ (parameterized by learnable weights θ) is trained to predict the noise ε that was added at a given step, given the noisy image xₜ and the step number t:

```
Loss = 𝔼_{x₀, ε, t} [ ‖ε − εθ(xₜ, t)‖² ]
```

Walkthrough: for training, take a real image x₀, pick a random step t, add the corresponding amount of noise to get xₜ (using the single-formula shortcut mentioned above — no need to simulate every intermediate step), and train the network to predict exactly what noise was added, given only the noisy result and the step number. This is simply a regression problem (MSE loss — see [loss-functions.md](../01-foundations/loss-functions.md)) with a clear, directly-computable target — much more stable to train than a GAN's adversarial objective (see [gans.md](gans.md)), since there's no second, simultaneously-changing network to chase. Once trained, generation works by starting with pure random noise x_T and repeatedly using the trained network's noise prediction to compute a slightly-less-noisy xₜ₋₁, all the way down to a clean x₀.

**Why it mattered.** DDPM demonstrated that this noise-prediction framing could produce image samples competitive with (and, over the next couple of years, better than) GANs, using a training procedure that was dramatically more stable and easier to scale.

## Score-based generative modeling (brief connection)

**Name & definition.** Score-based generative models (Song, Ermon, and colleagues, 2019-2021) frame generation in terms of learning the **score function** — the gradient of the log-probability of the data distribution with respect to the data itself (∇ₓ log p(x)) — at many different noise levels, and using that learned score to guide samples from pure noise toward high-probability (realistic) regions of the data distribution via a sampling procedure related to Langevin dynamics (a physics-inspired stochastic sampling method).

**The connection to DDPM.** It turns out that DDPM's noise-prediction objective and the score-based approach's score-matching objective are, mathematically, very closely related — predicting the noise added to a sample turns out to be essentially equivalent to estimating the score function at that noise level, up to a known scaling factor. This connection, formalized in later work (notably Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations," 2021), unified what had initially looked like two separate research directions into a single underlying theoretical framework, and much of the diffusion model literature since draws on both framings interchangeably.

## Latent diffusion (Stable Diffusion architecture)

**Origin.** Rombach, Blattmann, Lorenz, Esser, Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models" (2021/2022) — the paper underlying Stable Diffusion.

**The problem it solves.** Running the full noising/denoising diffusion process directly on raw, high-resolution pixel images is computationally expensive — every one of the many denoising steps has to process the full pixel grid.

**Core mechanism.** Instead of diffusing in pixel space, first compress the image into a much smaller latent representation using a pretrained VAE-style encoder (see [vaes.md](vaes.md) — this is a direct, literal reuse of the VAE architecture as a component, not just a conceptual parallel), run the entire noising/denoising diffusion process in that smaller latent space, and only decode back to full-resolution pixels once, at the very end, using the paired VAE decoder.

Walkthrough: since the latent space is much lower-dimensional than raw pixels but still captures the perceptually important structure of the image (that's what the VAE was trained to preserve), running dozens-to-hundreds of denoising steps in that smaller space is dramatically cheaper than doing the same in full pixel space, while producing final images (after the one decoding step back to pixels) of comparable quality. This architectural choice — diffuse in a compressed latent space, decode once at the end — is what made high-resolution diffusion-based image generation computationally practical enough for widespread deployment, and is the architecture underlying Stable Diffusion and closely related systems.

**Text conditioning.** To turn this into a text-to-image system, a text encoder (producing embeddings from a text prompt — see the glossary for "embedding") is used to condition the denoising network at every step (typically via cross-attention — see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md) — between the image-latent representation and the text embedding), so the denoising process is steered toward producing an image consistent with the prompt rather than an arbitrary realistic image.

## Classifier guidance vs. classifier-free guidance

**Classifier guidance** (Dhariwal, Nichol, 2021). Use a separately-trained classifier that predicts a class label (or other conditioning signal) from a noisy image xₜ, and use the *gradient* of that classifier's confidence with respect to xₜ to nudge the denoising process toward images the classifier would confidently label as the desired class/condition. This works, but requires training and maintaining a separate classifier model specifically for this purpose, which adds complexity and only supports whatever conditioning signal that separate classifier was trained on.

**Classifier-free guidance** (Ho, Salimans, 2022). Instead of a separate classifier, train a single diffusion model that can operate in both a "conditioned" mode (given the text prompt/condition) and an "unconditioned" mode (given no condition — the condition input replaced with a fixed placeholder during some fraction of training). At generation time, run the model both ways, and extrapolate *away* from the unconditioned prediction and *toward* the conditioned one, by more than a 1:1 ratio:

```
guided_prediction = uncond_pred + w · (cond_pred − uncond_pred)
```

where w > 1 is the guidance scale (how strongly to push toward the condition). Walkthrough: the difference between the conditioned and unconditioned predictions represents "what specifically the condition adds"; amplifying that difference (rather than just using the conditioned prediction directly) pushes generation to more strongly reflect the prompt/condition than a simple conditioned prediction alone would, at some cost to overall sample diversity if the guidance scale is pushed too high.

**Why classifier-free guidance won.** It requires no separate classifier model, works with whatever conditioning signal (text, class label, image) the main diffusion model was trained on rather than being limited to a specific classifier's outputs, and has empirically produced better results — this is why essentially all deployed text-to-image diffusion systems as of 2026 use classifier-free guidance rather than classifier guidance.

## Why diffusion became dominant for image/video/audio generation

Tying together threads from this file and from [gans.md](gans.md): stable, simple, likelihood-adjacent training (no adversarial game to destabilize); strong empirical sample diversity (mode-collapse-type failures, endemic to GANs, are not a comparable concern for diffusion training); a natural, well-understood mechanism (classifier-free guidance) for conditioning generation on text/class/other signals with a controllable strength; and — via latent diffusion specifically — a practical path to high-resolution generation at a computationally reasonable cost. The same core noising/denoising framework has generalized well beyond static images to video (extending the noising process across a temporal dimension as well as spatial ones) and audio generation, which is part of why diffusion has become the default generative approach across multiple modalities, not just images.

## Current frontier status

Diffusion models remain the dominant approach for image, video, and audio generation as of 2026. The main practical drawback relative to some alternatives (particularly GANs, and to a lesser extent some newer few-step/distilled diffusion variants) is inference speed — the classic multi-step denoising process requires many sequential network evaluations per generated sample, which is markedly slower than a GAN's single forward pass or an autoregressive model's per-token generation. Active research directions (consistency models, distillation of many-step diffusion models into few-step or single-step samplers) are aimed specifically at closing this speed gap while retaining diffusion's quality and stability advantages; the field has made real, reported progress on this front, though many-step diffusion remains the default for maximum-quality generation as of this writing.

## Comparison table

| System | Year | Key Innovation | Still Relevant (2026)? |
|---|---|---|---|
| DDPM | 2020 | Noise-prediction training objective, stable and simple | Yes — foundational formulation, still directly used |
| Score-based generative modeling | 2019-2021 | Score-function framing, unifies with DDPM theoretically | Yes — theoretical backbone, used interchangeably with DDPM framing |
| Latent diffusion (Stable Diffusion) | 2021/2022 | Diffuse in compressed VAE latent space, not raw pixels | Yes — dominant practical architecture |
| Classifier-free guidance | 2022 | Amplify conditioned-vs-unconditioned prediction difference | Yes — standard conditioning technique |

## Relationship to other algorithms

- Latent diffusion uses a VAE (see [vaes.md](vaes.md)) as a literal architectural component for compression.
- Diffusion's stability advantage is best understood in direct contrast to GAN training instability — see [gans.md](gans.md).
- Text conditioning via cross-attention connects directly to the attention mechanism in [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md).
- The noise-prediction regression objective is an application of MSE loss — see [loss-functions.md](../01-foundations/loss-functions.md).

## Sources

- Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli, "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (2015)
- Ho, Jain, Abbeel, "Denoising Diffusion Probabilistic Models" (2020)
- Song, Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution" (2019)
- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" (2021)
- Rombach, Blattmann, Lorenz, Esser, Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models" (2021, published 2022) [Stable Diffusion]
- Dhariwal, Nichol, "Diffusion Models Beat GANs on Image Synthesis" (2021) [classifier guidance]
- Ho, Salimans, "Classifier-Free Diffusion Guidance" (2022)
