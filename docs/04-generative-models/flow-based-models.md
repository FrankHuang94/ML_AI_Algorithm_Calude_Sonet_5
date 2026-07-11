# Flow-Based Models

Flow-based models are the fifth and, in terms of real-world adoption, smallest of the major generative model families covered in this section. This file is deliberately brief and direct about that: normalizing flows are a mathematically elegant idea that has stayed niche in practice, and this file won't inflate their importance to match the space given to GANs, VAEs, or diffusion models.

## Normalizing flows: the concept

**Name & definition.** A normalizing flow models a complex data distribution by starting from a simple, well-understood distribution (like a standard Gaussian) and applying a sequence of **invertible** transformations to it — meaning each transformation can be run forward (simple distribution → data) or backward (data → simple distribution) exactly, with no information loss in either direction.

**Why invertibility matters — the change of variables formula.** Because every transformation in the flow is exactly invertible, and because there's a precise mathematical formula (the **change of variables formula** from probability theory) relating the probability density of a variable to the probability density of a transformed version of it, a normalizing flow can compute the *exact* likelihood of a given data point under the model — not an approximation or a lower bound (contrast this directly with the VAE's ELBO, which is only a lower bound on likelihood — see [vaes.md](vaes.md)), and not an implicit, non-probabilistic training signal (contrast with a GAN's adversarial objective — see [gans.md](gans.md)). This exact-likelihood property is the main mathematical selling point of the entire flow-based family.

**Core mechanism (conceptual).** Each layer of the flow applies a learned, invertible function (often something restricted in structure — like transforming only half the input dimensions at a time, conditioned on the other half — specifically so that the transformation's invertibility and the change-of-variables calculation stay computationally tractable, since a fully general invertible transformation would be far too expensive to work with). Stacking many such layers lets the overall flow represent a complex, flexible transformation from the simple base distribution to something that matches the true data distribution well, while every individual layer remains cheap enough to invert and compute a likelihood through.

## RealNVP and Glow (brief)

**RealNVP** ("Real-valued Non-Volume Preserving," Dinh, Sohl-Dickstein, Bengio, 2016) introduced the specific "transform half the dimensions, conditioned on the other half" layer design (a **coupling layer**) that made deep normalizing flows practical to train, by keeping the change-of-variables computation cheap (specifically, keeping the relevant determinant calculation — a byproduct of the change-of-variables formula — reducible to a simple product rather than an expensive general matrix computation).

**Glow** (Kingma, Dhariwal, 2018) extended this design with additional architectural refinements (notably, invertible 1×1 convolutions, generalizing RealNVP's fixed dimension-splitting into a learned mixing of channels) and demonstrated higher-quality image generation and smoother latent-space interpolation than RealNVP, at the time of its release.

## Why this family stayed niche

Being honest, as this repository's quality bar requires, about why flows never became a mainstream generative modeling default: the invertibility requirement is architecturally restrictive — every layer must be exactly invertible and have a cheaply-computable determinant, which rules out many of the most expressive, unrestricted neural network designs that GANs, VAEs, and diffusion models are free to use without any such constraint. This structural limitation has empirically translated into flow-based models needing considerably more layers/parameters to match the sample quality of GANs or diffusion models on complex, high-dimensional data like natural images, and they have not demonstrated the same scaling behavior that made diffusion models (see [diffusion-models.md](diffusion-models.md)) dominant for large-scale image/video/audio generation. Flow-based models remain most valuable specifically in settings where *exact* likelihood computation is itself the point (rather than just sample quality) — some scientific and density-estimation applications, and specific research contexts studying probabilistic modeling — rather than as a general-purpose route to high-fidelity generation.

## Current status

**Strengths & limitations.**
- Strengths: exact likelihood computation (not a bound or an implicit signal); exact invertibility gives clean encode/decode in both directions; theoretically elegant, well-grounded framework.
- Limitations: architectural constraints (invertibility, tractable determinants) limit expressiveness relative to unconstrained generative architectures; has not matched GAN/diffusion-model sample quality on complex high-dimensional data; comparatively little large-scale industrial adoption relative to the other four generative model families in this section.

**Current status.** Niche as of 2026 — a smaller, more specialized research area than GANs, VAEs, diffusion models, or autoregressive generation, used primarily where exact likelihoods matter or in specific scientific/density-estimation contexts, rather than as a general-purpose competitor for mainstream image/video/audio/text generation.

## Comparison table

| System | Year | Key Innovation | Still Relevant (2026)? |
|---|---|---|---|
| RealNVP | 2016 | Coupling layers make deep invertible flows practical | Niche — foundational design, limited mainstream use |
| Glow | 2018 | Invertible 1×1 convolutions, improved sample quality | Niche — same general status as RealNVP |

## Relationship to other algorithms

- Flow-based models' exact likelihood is a direct contrast to the VAE's approximate ELBO lower bound — see [vaes.md](vaes.md).
- Their invertibility-driven training stability is a contrast to GAN adversarial instability — see [gans.md](gans.md).
- As a smaller-scale, more specialized family, flows are best understood in this file's comparative context alongside the other four generative model families in this section, rather than in isolation.

## Sources

- Dinh, Sohl-Dickstein, Bengio, "Density estimation using Real NVP" (2016)
- Kingma, Dhariwal, "Glow: Generative Flow with Invertible 1x1 Convolutions" (2018)
