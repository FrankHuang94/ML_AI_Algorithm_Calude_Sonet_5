# CNN Family

Convolutional Neural Networks (CNNs) are the architecture family that made deep learning work for images, and their core building block — the convolution — is still used throughout modern vision systems even as pure CNNs have partly given ground to vision transformers for the largest-scale models. This file builds up convolution and pooling from scratch, then walks the lineage from LeNet through ResNet and EfficientNet, and briefly covers detection architectures as an applied use case.

## Convolution and pooling mechanics, from scratch

**The problem with fully-connected layers on images.** A standard fully-connected layer connects every input value to every output neuron with its own independent weight. For an image, this is a bad fit for two reasons: the parameter count explodes (a 256×256 RGB image has ~196,000 input values; a single fully-connected layer to another 196,000 values would need on the order of 10^10 weights), and it discards the spatial structure of images entirely — the layer has no built-in notion that nearby pixels are related, or that a pattern of pixels representing an edge in the top-left of an image is the same kind of pattern as the same edge in the bottom-right.

**Convolution.** A convolutional layer instead applies a small learned **filter** (or **kernel** — a small grid of weights, e.g., 3×3 or 5×5) that slides across the entire image, computing a dot product between the filter and the local patch of pixels underneath it at every position. Formally, for a 2D input and filter:

```
output[i, j] = Σ_m Σ_n  input[i+m, j+n] · filter[m, n]
```

Walkthrough: at each position (i, j), you multiply the filter's weights element-wise against the corresponding patch of the input and sum the result into a single output value, then slide the filter over by one (or more) pixels and repeat. Here it is concretely — a 3×3 "vertical edge detector" filter sliding over one patch of an image:

```
   Image patch          Filter (vertical         Element-wise
   (pixel values)        edge detector)           multiply & sum
   ┌───┬───┬───┐         ┌────┬───┬────┐
   │ 0 │ 0 │ 9 │         │ -1 │ 0 │ +1 │          (0·-1)+(0·0)+(9·+1)
   ├───┼───┼───┤    ⊛    ├────┼───┼────┤    =     +(0·-1)+(0·0)+(9·+1)   =  +27
   │ 0 │ 0 │ 9 │         │ -1 │ 0 │ +1 │          +(0·-1)+(0·0)+(9·+1)
   ├───┼───┼───┤         ├────┼───┼────┤
   │ 0 │ 0 │ 9 │         │ -1 │ 0 │ +1 │          → large positive = "edge found here"
   └───┴───┴───┘         └────┴───┴────┘
   (dark left,           (fires on left-to-        A flat patch (all same value)
    bright right)         right brightness jumps)   would sum to ≈ 0 — no edge.
```

The filter outputs a big number wherever the image locally matches its pattern (here, a dark-to-bright vertical transition) and near-zero where it doesn't. Slide this same filter across the whole image and you get a "map" of everywhere that edge appears. Critically, the *same* filter weights are reused at every position — this is called **weight sharing**, and it's the key idea that fixes both problems above: the parameter count for one filter is just its size (e.g., 9 weights for a 3×3 filter) regardless of image size, and because the same filter is applied everywhere, a pattern the filter learns to detect (like a vertical edge) is detected wherever it appears in the image, not just in one fixed location. A convolutional layer typically learns many filters in parallel (e.g., 64 different 3×3 filters), each producing its own output "channel," so the layer can detect many different local patterns simultaneously. And crucially, these filters are *learned*, not hand-designed — the edge detector above is illustrative, but in a trained network the filters emerge from data (early layers reliably learn edge- and color-detectors like this one; later layers learn detectors for textures, object parts, and eventually whole objects).

**Pooling.** A pooling layer downsamples its input by summarizing small regions into a single value — most commonly **max pooling** (take the maximum value in each small region, e.g., each 2×2 block) or **average pooling** (take the mean). Pooling has no learned weights; it's a fixed summarization operation. Walkthrough: pooling reduces the spatial resolution of the data flowing through the network (e.g., a 2×2 max-pool halves both height and width), which reduces compute for subsequent layers and adds a degree of **translation invariance** (a small shift in where a feature appears in the input has less effect on the pooled output, since the pooling operation summarizes over a local neighborhood rather than reading one exact pixel location).

**Stacking layers.** A typical CNN alternates convolution (often followed by a nonlinearity like ReLU) and pooling layers, so that early layers detect simple local patterns (edges, color blobs) and later layers — operating on the already-abstracted output of earlier layers, and covering progressively larger effective regions of the original input (a growing **receptive field**) — detect increasingly complex, composite patterns (textures, object parts, whole objects).

## LeNet

**Origin.** LeCun et al., "Gradient-Based Learning Applied to Document Recognition" (1998), building on earlier work by the same group in the late 1980s-early 1990s.

**Why it mattered.** LeNet-5 was the first CNN to demonstrate, on a real practical task (handwritten digit recognition for check processing), that the convolution-plus-pooling architecture, trained end-to-end via backpropagation, could beat hand-engineered feature-extraction pipelines. It's the direct architectural ancestor of every CNN below.

**Current status.** Purely historical/educational at this point — its scale (a few thousand parameters) is trivial by modern standards — but its structure (convolution → pooling → convolution → pooling → fully-connected classifier head) is still the basic template every later CNN elaborates on.

## AlexNet

**Origin.** Krizhevsky, Sutskever, Hinton, "ImageNet Classification with Deep Convolutional Neural Networks" (2012).

**Why it mattered.** AlexNet's dramatic win in the 2012 ImageNet Large Scale Visual Recognition Challenge (a large-scale image classification benchmark) — beating the next-best (non-deep-learning) approach by a wide margin — is widely regarded as the moment that convinced the broader ML community that deep learning, given enough data (ImageNet's ~1 million labeled images) and enough compute (AlexNet was trained on GPUs, itself a significant practical choice at the time), could substantially outperform the hand-engineered computer vision pipelines that had dominated for the prior decade. It used ReLU activations (faster to train than the sigmoid/tanh activations common before it, because ReLU's gradient doesn't shrink for large positive inputs), dropout (see [regularization-techniques.md](../01-foundations/regularization-techniques.md)), and data augmentation — several of which are still standard practice.

**Current status.** Historical — its specific architecture is not used today — but this is the paper this repository's history section (see [08-history](../08-history/)) treats as the field-defining "ImageNet moment."

## VGG

**Origin.** Simonyan, Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition" (2014).

**Core contribution.** VGG demonstrated that stacking many small (3×3) convolutional filters — rather than using fewer, larger filters as AlexNet had — could achieve better accuracy, because stacking two 3×3 convolutions gives the same effective receptive field as one 5×5 convolution while using fewer parameters and adding an extra nonlinearity in between (allowing the network to learn a more expressive function for the same receptive field size). VGG's architecture is also notably simple and uniform (just 3×3 convolutions and 2×2 max pooling, repeated), which made it easy to analyze and reuse as a feature extractor for other tasks.

**Current status.** Rarely used as-is today (its 130+ million parameters are large and inefficient by modern standards for its accuracy level), but VGG-style feature extractors remain used in some specific applications like perceptual loss functions for image generation (measuring how similar two images "look" to a pretrained VGG network's internal features, rather than comparing raw pixels).

## ResNet — skip connections and solving vanishing gradients at depth

**Origin.** He, Zhang, Ren, Sun, "Deep Residual Learning for Image Recognition" (2015).

**The problem it solved.** Before ResNet, simply stacking more layers did not reliably improve accuracy past a certain depth — and, surprisingly, *training* accuracy (not just generalization) got *worse* for very deep plain networks, indicating an optimization problem, not overfitting. The core cause is the vanishing gradient problem (see [optimization-algorithms.md](../01-foundations/optimization-algorithms.md) and [rnn-lstm-gru.md](rnn-lstm-gru.md)): as the loss's gradient is backpropagated through many stacked layers, it's repeatedly multiplied by the local derivative of each layer, and if those derivatives tend to be less than 1 (common for layers with saturating nonlinearities, and even a persistent issue with ReLU-based networks at extreme depth), the gradient reaching early layers can shrink toward zero, leaving those layers essentially untrained.

**Core mechanism.** A **residual (skip) connection** adds a layer block's input directly to its output: instead of a block learning a full transformation H(x), it learns only the *residual* F(x) = H(x) − x, and the block's actual output is F(x) + x.

```
output = F(x) + x
```

Walkthrough: the "+x" term provides a direct, unimpeded path for the gradient to flow backward through the network — the gradient of "output" with respect to "x" always includes a clean "1" term from the identity shortcut, in addition to whatever the F(x) branch contributes, so the gradient can't vanish to zero purely from passing through that block, no matter how poorly-scaled F(x)'s own gradient might be. This also means that if a given block's transformation isn't useful, the network can effectively learn to make F(x) ≈ 0, letting the block default to passing its input through unchanged — much easier to learn than an equivalent identity mapping would be in a plain (non-residual) block.

**Why it mattered.** ResNet made it practical to train networks with over 100 layers (the original paper trained versions up to 152 layers) that reliably improved with depth, whereas prior plain architectures had stalled out around 20-30 layers. This was a foundational unlock — and residual connections went on to become an essentially universal architectural element, used in the Transformer (see [transformer-architecture.md](transformer-architecture.md)) and nearly every deep architecture since (also discussed as an implicit regularizer in [regularization-techniques.md](../01-foundations/regularization-techniques.md)).

**Current status.** ResNet variants remain in active production use for many vision tasks (particularly where compute/latency budgets favor CNNs over vision transformers, or where the additional inductive bias of convolution — see below — helps on smaller datasets), and skip connections specifically are essentially universal in modern deep learning, well beyond vision.

## Inception / GoogLeNet

**Origin.** Szegedy et al., "Going Deeper with Convolutions" (2014).

**Core contribution.** The Inception module runs several different filter sizes (e.g., 1×1, 3×3, 5×5) and a pooling operation *in parallel* on the same input and concatenates their outputs, rather than choosing one filter size per layer. This lets the network capture patterns at multiple spatial scales simultaneously at each stage. It also popularized 1×1 convolutions as a cheap way to reduce the number of channels before an expensive larger convolution, cutting compute substantially.

**Current status.** Its multi-scale-parallel-branch idea influenced later architecture design broadly, but GoogLeNet/Inception itself is not commonly deployed today; EfficientNet and ResNet-family architectures are more common practical choices, and vision transformers dominate the largest-scale regime.

## EfficientNet — compound scaling

**Origin.** Tan, Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (2019).

**Core contribution.** Prior work typically scaled up CNNs along a single dimension at a time — more layers (depth), more channels per layer (width), or higher input image resolution. EfficientNet's key insight is that these three dimensions interact, and scaling them together, in a fixed ratio determined by a small grid search, gives much better accuracy-per-unit-of-compute than scaling any single dimension alone. This **compound scaling** approach, combined with a carefully-designed small base architecture found via neural architecture search (an automated search process for good architectures, rather than hand design), produced a family of models (EfficientNet-B0 through B7) with substantially better accuracy-per-parameter and accuracy-per-FLOP than prior hand-designed CNNs at the time of release.

**Current status.** EfficientNet and its descendants remain a strong, efficiency-focused choice for vision tasks with real compute/latency constraints (e.g., on-device or edge deployment), though the very largest-scale vision workloads have increasingly moved to vision transformer variants.

## Detection architectures (applied use case, brief)

Object detection — localizing and classifying multiple objects within an image — is a major applied use of CNN features, covered here briefly as a use case rather than in full architectural depth.

- **The R-CNN family** (R-CNN, Fast R-CNN, Faster R-CNN — Girshick et al., 2013-2016) works by proposing candidate regions of an image likely to contain an object, then classifying and refining each proposed region using CNN features. This "propose, then classify" two-stage approach is accurate but comparatively slow.
- **YOLO** ("You Only Look Once" — Redmon et al., 2016) reframes detection as a single-pass regression problem: one forward pass through the network directly predicts bounding boxes and class probabilities across the whole image, without a separate region-proposal stage. This is substantially faster, making YOLO-family models (which have gone through many iterations since) the standard choice for real-time detection applications.

**Current status.** Both families remain in production use, chosen based on the accuracy/speed tradeoff a given application needs — two-stage detectors when maximum accuracy matters more than latency, single-pass detectors (YOLO family) for real-time constraints.

## CNNs vs. Vision Transformers — where things stand in 2026

CNNs' core architectural bias — that nearby pixels matter more than far-apart ones, encoded directly into the convolution operation (an **inductive bias**, meaning a built-in assumption baked into the architecture rather than something the model has to learn from data) — makes them data-efficient: they can achieve good accuracy on comparatively small/medium datasets because they don't have to learn "nearby pixels are related" from scratch. Vision Transformers (which apply the Transformer architecture from [transformer-architecture.md](transformer-architecture.md) to image patches) drop this built-in bias, which lets them potentially learn more flexible, less locally-constrained patterns — but only pays off with very large training datasets, since the model has to learn spatial structure from data rather than getting it for free. As of 2026, CNNs remain the more practical, more data-efficient choice for many mid-scale and resource-constrained vision applications, while vision transformers (and hybrid CNN-transformer designs) tend to be preferred for the largest-scale vision and multimodal systems where enormous training datasets are available.

## Comparison table

| Architecture | Year | Key Innovation | Still Relevant (2026)? |
|---|---|---|---|
| LeNet | 1998 | First practical trained CNN (convolution + pooling + backprop) | Historical/educational |
| AlexNet | 2012 | GPU-trained deep CNN, ReLU, dropout; ImageNet breakthrough | Historical (the "ImageNet moment") |
| VGG | 2014 | Stacked small (3×3) filters, uniform simple design | Feature extractor niches (e.g., perceptual loss) |
| Inception/GoogLeNet | 2014 | Parallel multi-scale filters, 1×1 conv for cheap channel reduction | Design ideas persist; rarely deployed as-is |
| ResNet | 2015 | Skip connections solve vanishing gradients at depth | Yes — widely deployed, and residuals are near-universal |
| EfficientNet | 2019 | Compound scaling (depth/width/resolution together) | Yes — strong efficiency-focused choice |

## Relationship to other algorithms

- Skip connections, introduced here via ResNet, are foundational to the Transformer block — see [transformer-architecture.md](transformer-architecture.md).
- The vanishing gradient problem is explained in its other major historical context (recurrent networks) in [rnn-lstm-gru.md](rnn-lstm-gru.md).
- CNN feature hierarchies are conceptually related to how self-attention builds contextual representations, though the mechanisms (local, shared filters vs. global, content-based attention) differ substantially — see [transformer-architecture.md](transformer-architecture.md).
- Batch Normalization, commonly used in CNN architectures, is covered in [regularization-techniques.md](../01-foundations/regularization-techniques.md).

## Sources

- LeCun, Bottou, Bengio, Haffner, "Gradient-Based Learning Applied to Document Recognition" (1998)
- Krizhevsky, Sutskever, Hinton, "ImageNet Classification with Deep Convolutional Neural Networks" (2012)
- Simonyan, Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition" (2014)
- Szegedy et al., "Going Deeper with Convolutions" (2014)
- He, Zhang, Ren, Sun, "Deep Residual Learning for Image Recognition" (2015)
- Tan, Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" (2019)
- Girshick, Donahue, Darrell, Malik, "Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation" (2013) [R-CNN]; Girshick, "Fast R-CNN" (2015); Ren, He, Girshick, Sun, "Faster R-CNN" (2016)
- Redmon, Divvala, Girshick, Farhadi, "You Only Look Once: Unified, Real-Time Object Detection" (2016)
