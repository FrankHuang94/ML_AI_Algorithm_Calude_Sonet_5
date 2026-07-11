# Variational Autoencoders

Variational Autoencoders (VAEs) turn the basic autoencoder (see [unsupervised-learning.md](../02-classical-ml/unsupervised-learning.md)) into a genuine generative model — one that can produce new, realistic samples, not just compress and reconstruct existing ones. VAEs are also historically and conceptually important as the model family whose "latent space" framing carries directly into how diffusion models are built (see [diffusion-models.md](diffusion-models.md)).

## From autoencoders to a generative model

**The problem with a plain autoencoder.** A basic autoencoder (see [unsupervised-learning.md](../02-classical-ml/unsupervised-learning.md)) learns to map inputs to points in a latent space (a compressed vector representation) and back, but it gives no guarantee about *what* those latent points look like as a whole — they might be scattered in an irregular, disconnected way that leaves large "gaps" with no meaning. If you tried to generate a new example by picking a random point in that latent space and decoding it, you'd have no way to know whether that point lands somewhere the decoder actually knows how to turn into something realistic.

**The VAE fix, conceptually.** A VAE forces the latent space to have a specific, well-behaved overall shape — specifically, it encourages the encoder to map inputs to points distributed roughly like a standard normal (Gaussian) distribution, rather than to arbitrary, unconstrained points. If the whole latent space is shaped this way, then randomly sampling from a standard normal distribution and feeding that sample into the decoder should reliably produce something realistic — because that's exactly the kind of input the decoder was trained on.

**Origin.** Kingma, Welling, "Auto-Encoding Variational Bayes" (2013/2014).

## Core mechanism: encoder outputs a distribution, not a point

**The key structural change from a plain autoencoder.** Instead of the encoder mapping an input x directly to a single latent vector z, a VAE's encoder outputs the *parameters of a probability distribution* over z — specifically, a mean vector μ and a standard-deviation vector σ, defining a Gaussian distribution. The actual latent vector z used by the decoder is then a random sample drawn from that Gaussian: z ~ N(μ, σ²).

Walkthrough: rather than saying "this input maps to exactly this one point in latent space," the VAE's encoder says "this input maps to this *region* of latent space" (centered at μ, with spread σ), and a specific point in that region is randomly sampled each time. This is what gives the latent space its smooth, well-behaved structure — nearby inputs get encoded to overlapping regions, and because sampling is involved, the decoder is forced to produce reasonable outputs for a whole neighborhood around μ, not just for one exact point, which is what makes interpolating between points in latent space (and sampling new random points) produce sensible results.

## The reparameterization trick

**The problem it solves.** Training requires backpropagating gradients through the entire encoder-decoder pipeline (see [optimization-algorithms.md](../01-foundations/optimization-algorithms.md)). But "randomly sample z from a distribution defined by μ and σ" is not a differentiable operation in the ordinary sense — you can't directly ask "how would the loss change if μ changed slightly," because the *randomness* itself sits between μ/σ and z, blocking the gradient from flowing through in the usual way.

**The trick.** Instead of sampling z directly from N(μ, σ²), sample a fixed auxiliary random value ε ~ N(0, 1) (standard normal, with no learned parameters at all) and compute z deterministically from μ, σ, and ε:

```
z = μ + σ · ε
```

Walkthrough: this is mathematically equivalent to sampling z ~ N(μ, σ²) directly (multiplying a standard normal sample by σ and shifting it by μ produces exactly a N(μ, σ²)-distributed value), but it restructures the computation so that all the *randomness* is isolated in ε — a value with no learned parameters, sampled once per forward pass and then held fixed for the purposes of computing gradients. Because z is now a deterministic, differentiable function of μ, σ, and ε, gradients can flow backward through z to μ and σ (and from there, back into the encoder's weights) exactly as they would through any other layer. This is a genuinely clever, non-obvious piece of engineering — moving the "randomness" outside the part of the computation graph that needs a gradient — and it's the specific mechanical trick that makes training a VAE with standard backpropagation possible at all.

## The ELBO, explained before the formula

**In plain English first.** A VAE would ideally like to directly maximize the likelihood of the training data under the model — roughly, "how probable does my model consider the real training examples to be" (higher is better). But computing this exact likelihood requires accounting for every possible latent vector z that could have produced a given input x, which is generally intractable (there's no efficient way to sum/integrate over all of them for a complex model). Instead, VAEs maximize a *lower bound* on that likelihood — a quantity that's provably always less than or equal to the true (intractable) likelihood, but which is actually possible to compute. This lower bound is called the **Evidence Lower BOund**, or **ELBO**. The logic is: since you can't directly maximize the real target, maximize the best computable stand-in for it instead — and because it's a genuine lower bound (not just a loose approximation), pushing it up is guaranteed to be pushing the true likelihood up too (or at least not letting it go down), even though you can't measure the true likelihood directly.

**The formula.**

```
ELBO = 𝔼_{z~q(z|x)}[log p(x|z)]  −  D_KL(q(z|x) ‖ p(z))
```

Walkthrough of each term: q(z|x) is the encoder's distribution over z given input x (the μ, σ from above); p(x|z) is the decoder's distribution over reconstructions given a latent z; p(z) is the **prior** — the well-behaved target distribution we want the latent space to look like overall (standard normal, N(0,1)). The first term, 𝔼_{z~q(z|x)}[log p(x|z)], is the **reconstruction term**: it rewards the model for reconstructing x accurately from latent samples drawn from the encoder's distribution — this is essentially the same reconstruction objective a plain autoencoder uses. The second term, the KL divergence (see [loss-functions.md](../01-foundations/loss-functions.md)) between the encoder's distribution q(z|x) and the prior p(z), is the **regularization term**: it penalizes the encoder for producing a distribution that strays too far from the well-behaved standard normal shape we want the overall latent space to have. Training maximizes the ELBO (equivalently, minimizes its negative as a loss), which means balancing two competing pressures: reconstruct the input well, but don't let the latent distribution drift too far from the nice, samplable standard normal shape. This tension is exactly what gives a trained VAE both a latent space that's well-organized enough to sample from, and a decoder that's good enough to make those samples look realistic.

## Relationship to autoencoders and diffusion models

**Vs. plain autoencoders.** A plain autoencoder (see [unsupervised-learning.md](../02-classical-ml/unsupervised-learning.md)) is deterministic (fixed input → fixed latent point) and has no mechanism encouraging a well-organized latent space — it's a compression tool, not a generative model. A VAE's probabilistic encoder and KL-regularization term are exactly what convert "compression" into "generation."

**Vs. diffusion models.** Diffusion models (see [diffusion-models.md](diffusion-models.md)) can be viewed, at a conceptual level, as pushing the VAE's core idea to an extreme: instead of one encoding step compressing an input into a single latent distribution, diffusion uses a long sequence of many small, gradual noising steps, and instead of one decoding step, uses a long sequence of many small denoising steps. Latent diffusion models (a specific, widely-used variant covered in [diffusion-models.md](diffusion-models.md)) additionally use an actual VAE explicitly as a first-stage compressor — running the diffusion process itself in a VAE's smaller latent space rather than directly on raw pixels — a direct, literal architectural reuse of the VAE mechanism covered in this file, not just a conceptual echo of it.

## Current status

**Strengths & limitations.**
- Strengths: principled probabilistic framework with a well-organized, interpretable, smoothly-interpolatable latent space; stable, straightforward training (a single well-defined loss to minimize, unlike GANs' adversarial game — see [gans.md](gans.md)); fast single-pass generation once trained.
- Limitations: VAE-generated samples have historically tended to look somewhat blurrier/less sharp than GAN or diffusion-model outputs (a widely-observed, if not perfectly theoretically settled, empirical tendency, often attributed to the reconstruction term's behavior under commonly-used simplifying assumptions); balancing the reconstruction and KL terms well in practice can require careful tuning.

**Current status.** VAEs as a *standalone* end-to-end image generator have been substantially outpaced in sample quality by diffusion models (see [diffusion-models.md](diffusion-models.md)) and, in their image-generation heyday, by GANs (see [gans.md](gans.md)). However, the VAE mechanism itself remains very much alive and load-bearing as a *component* inside modern systems — most prominently, as the compression stage inside latent diffusion architectures like Stable Diffusion (see [diffusion-models.md](diffusion-models.md)) — which makes VAEs simultaneously "superseded as a standalone generator" and "quietly essential infrastructure inside the current state of the art," a distinction worth being precise about rather than writing VAEs off entirely.

## Comparison table

| Property | Plain autoencoder | VAE |
|---|---|---|
| Encoder output | Single deterministic latent vector | Distribution (μ, σ) over latent vectors |
| Latent space structure | No guarantee of good organization | Encouraged toward a smooth, standard-normal shape |
| Can generate new samples? | Not reliably | Yes — sample from prior, decode |
| Training objective | Reconstruction loss only | ELBO (reconstruction + KL regularization) |

## Relationship to other algorithms

- VAEs are a direct probabilistic extension of the plain autoencoders covered in [unsupervised-learning.md](../02-classical-ml/unsupervised-learning.md).
- The KL divergence term in the ELBO is the same quantity covered generally in [loss-functions.md](../01-foundations/loss-functions.md).
- The alternating "infer the hidden structure, then optimize given that structure" flavor of variational inference echoes the EM algorithm covered in [probabilistic-models.md](../02-classical-ml/probabilistic-models.md), though VAEs optimize via gradient descent/backprop rather than EM's exact alternating steps.
- VAEs are used as a literal architectural component inside latent diffusion models — see [diffusion-models.md](diffusion-models.md).

## Sources

- Kingma, Welling, "Auto-Encoding Variational Bayes" (2013, published 2014)
- Rezende, Mohamed, Wierstra, "Stochastic Backpropagation and Approximate Inference in Deep Generative Models" (2014) — independently introduced closely related ideas around the same time
