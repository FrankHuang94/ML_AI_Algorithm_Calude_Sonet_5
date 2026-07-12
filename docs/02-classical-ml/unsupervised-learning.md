# Unsupervised Learning

Unsupervised learning finds structure in data that has no labels — no "correct answer" is provided per example. This file covers clustering (grouping similar examples together), dimensionality reduction (compressing data into fewer dimensions while preserving important structure), and basic autoencoders, which bridge into the generative models covered in [04-generative-models](../04-generative-models/).

The clustering algorithms below differ mainly in *what shape of cluster they can find*, and the cleanest way to see this is a dataset where the "right" answer is two concentric rings — an inner blob surrounded by an outer ring. k-means, which assumes clusters are round blobs around a center point, gets it badly wrong; DBSCAN, which follows density, gets it right:

```
   The true structure          k-means result             DBSCAN result
   (2 concentric rings)         (splits by nearest         (follows density,
                                 center — WRONG)            recovers rings — RIGHT)
      x x x x x                   x x A A A                   x x x x x
    x         x                 x         A                 x         x
   x   o o o   x               B   o o A   A               x   o o o   x
   x   o   o   x               B   o   A   A               x   o   o   x
   x   o o o   x               B   B o A   A               x   o o o   x
    x         x                 B         A                 x         x
      x x x x x                   B B B A A                   x x x x x
                              (cuts a straight line       (inner blob = one cluster,
                               through both rings)          outer ring = another)
```

This one picture explains most of the "when to use what" for clustering: if your clusters are roughly round and similarly-sized, k-means is fast and fine; if they have arbitrary shapes or you don't know how many there are, you need a density- or hierarchy-based method. Keep the rings in mind as you read.

## k-means clustering

**Name & definition.** k-means partitions data into k clusters by iteratively assigning each point to its nearest cluster center and then recomputing each center as the mean of its assigned points.

**Origin.** The algorithm has multiple independent originators; Lloyd's 1957 Bell Labs technical report (published 1982) and MacQueen's 1967 paper are the most commonly cited.

**Core mechanism.** 1) Initialize k cluster centers (**centroids**), often randomly or via smarter seeding (e.g., k-means++). 2) Assign every data point to its nearest centroid (by Euclidean distance). 3) Recompute each centroid as the mean of all points currently assigned to it. 4) Repeat steps 2-3 until assignments stop changing (convergence).

Walkthrough: this is a chicken-and-egg problem solved by alternating — you can't know the best centroids without knowing the assignments, and you can't know the best assignments without knowing the centroids, so you guess one, solve for the other, and iterate until things stabilize. This is a specific case of a more general pattern called **Expectation-Maximization (EM)** (covered fully in [probabilistic-models.md](probabilistic-models.md)), though basic k-means is usually taught and implemented without invoking that framing explicitly.

**Why it mattered.** k-means is the simplest, fastest, most widely taught clustering algorithm, and remains a standard first tool for exploratory grouping of data (customer segmentation, document clustering, image color quantization) precisely because of that simplicity.

**Current status.** Still very widely used for quick, exploratory clustering; displaced by more sophisticated methods when cluster shapes are non-spherical or of very different sizes/densities.

**Strengths & limitations.**
- Strengths: fast, simple, scales well to large datasets.
- Limitations: requires choosing k in advance; assumes roughly spherical, similarly-sized clusters (struggles badly with elongated or unevenly-sized/dense clusters); sensitive to initialization (can converge to a poor local optimum — common practice is to run it multiple times with different initializations and keep the best result); sensitive to feature scaling.

## Hierarchical clustering

**Name & definition.** Hierarchical clustering builds a tree of nested clusters, either by starting with every point as its own cluster and repeatedly merging the closest pair (**agglomerative**, the more common direction) or by starting with one cluster and repeatedly splitting it (**divisive**).

**Core mechanism.** Agglomerative clustering computes pairwise distances between all clusters, merges the two closest, and repeats — recording the sequence of merges as a tree (a **dendrogram**). "Distance between clusters" itself requires a choice (a **linkage criterion**): single linkage (distance between the closest pair of points across the two clusters), complete linkage (the farthest pair), or average linkage, among others.

**Why it mattered.** Unlike k-means, hierarchical clustering doesn't require choosing the number of clusters up front — you build the whole tree and then decide, by inspecting the dendrogram, where to "cut" it to get however many clusters seem appropriate.

**Current status.** Useful for smaller datasets and for exploratory analysis where the natural number of clusters is itself unknown; the full pairwise-distance computation makes it expensive (roughly quadratic or worse in the number of points) for very large datasets compared to k-means.

## DBSCAN

**Name & definition.** DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups together points that are closely packed (high density), and marks points in low-density regions as noise/outliers rather than forcing them into a cluster.

**Origin.** Ester, Kriegel, Sander, Xu (1996).

**Core mechanism.** DBSCAN defines a neighborhood radius ε (epsilon) and a minimum point count. A point is a "core point" if at least that many other points fall within ε of it. Clusters are formed by chaining together core points that are within ε of each other (and their neighbors); points that don't belong to any dense chain are labeled noise.

**Why it mattered.** DBSCAN doesn't require specifying the number of clusters (unlike k-means), can find arbitrarily-shaped clusters (unlike k-means' spherical assumption), and explicitly identifies outliers rather than forcing every point into some cluster.

**Current status.** A standard choice when cluster shapes are irregular or when explicit outlier/noise detection matters; requires tuning ε and the minimum-point threshold, and can struggle when clusters have very different densities from one another.

## Principal Component Analysis (PCA)

**Name & definition.** PCA is a dimensionality-reduction technique that finds a small number of new axes (linear combinations of the original features) that capture as much of the data's variance as possible.

**Origin.** Pearson (1901), later formalized statistically by Hotelling (1933).

**Core mechanism.** PCA computes the **eigenvectors** of the data's covariance matrix (a matrix describing how every pair of features co-varies with each other); these eigenvectors are the "principal components" — new axes, each orthogonal (at a right angle) to the others, ordered by how much of the data's total variance they capture. Projecting the data onto the top few principal components gives a lower-dimensional representation that preserves as much of the original variance as possible for that number of dimensions.

Walkthrough: imagine a cloud of data points shaped like a flattened ellipse in 3D space — most of the "spread" is along one long axis, less along a second, and very little along a third. PCA finds exactly those axes mathematically (via the eigenvectors, ordered by their corresponding eigenvalues, which measure how much variance lies along each axis) and lets you discard the low-variance axes with minimal information loss, e.g., representing the data with 2 numbers instead of 3.

**Why it mattered.** PCA is the foundational, still-widely-taught linear approach to dimensionality reduction: fast, deterministic (no random initialization sensitivity, unlike k-means), and useful both for visualization (project to 2-3 dimensions) and as a preprocessing step to reduce feature counts before other algorithms.

**Current status.** Still standard for linear dimensionality reduction, denoising, and as an interpretable, fast baseline. For nonlinear structure, t-SNE and UMAP (below) are typically preferred, especially for visualization.

## t-SNE

**Name & definition.** t-SNE (t-Distributed Stochastic Neighbor Embedding) is a nonlinear dimensionality-reduction technique specialized for visualization, which tries to preserve local neighborhood structure (which points are close to which) when projecting high-dimensional data down to 2 or 3 dimensions.

**Origin.** van der Maaten and Hinton, "Visualizing Data using t-SNE" (2008).

**Core mechanism (conceptual).** t-SNE converts pairwise distances in the high-dimensional space into probabilities of "being neighbors," then searches for a low-dimensional layout whose corresponding neighbor-probabilities match as closely as possible (measured via KL divergence — see [loss-functions.md](../01-foundations/loss-functions.md)). The "t" refers to using a heavy-tailed t-distribution for the low-dimensional similarities specifically to avoid the "crowding problem" (in low dimensions, there isn't enough room for many points to all be moderately far from a given point simultaneously, the way there is in high dimensions).

**Why it mattered.** t-SNE produces visually compelling, well-separated cluster visualizations from high-dimensional data (like neural network embeddings) that PCA's linear projections often can't reveal, and became a near-standard tool for visually inspecting learned representations throughout the 2010s.

**Current status.** Still widely used for visualization; important caveats the field has learned to respect — t-SNE distorts global distances and cluster sizes (only *local* neighborhoods are reliably preserved, so the relative sizes and distances between visually separated clusters shouldn't be over-interpreted), is sensitive to its perplexity hyperparameter, and is not designed for general-purpose dimensionality reduction upstream of other ML algorithms (unlike PCA or UMAP).

## UMAP

**Name & definition.** UMAP (Uniform Manifold Approximation and Projection) is a newer nonlinear dimensionality-reduction technique with similar visualization goals to t-SNE, built on a different mathematical foundation (topological data analysis and manifold learning theory).

**Origin.** McInnes, Healy, Melville, "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction" (2018).

**Why it mattered / current status.** UMAP is generally faster than t-SNE (scales better to larger datasets), does a better job preserving some aspects of global structure (relative distances between clusters are somewhat more meaningful than in t-SNE, though still not fully reliable), and can be used more generally as a preprocessing/dimensionality-reduction step before downstream algorithms, not just for visualization. As of 2026, UMAP has substantially displaced t-SNE as the default choice for visualizing high-dimensional embeddings in most practical workflows, though t-SNE remains common in older codebases and some scientific fields with established conventions around it.

## Autoencoders (basic, non-variational)

**Name & definition.** An autoencoder is a neural network trained to reconstruct its own input, forced through a narrow "bottleneck" layer (a latent representation with fewer dimensions than the input) that compels it to learn a compressed, useful representation rather than just copying data through.

**Core mechanism.** An **encoder** network compresses the input x into a lower-dimensional latent vector z = encoder(x); a **decoder** network reconstructs x̂ = decoder(z) from that compressed vector. Training minimizes reconstruction loss (e.g., MSE between x and x̂ — see [loss-functions.md](../01-foundations/loss-functions.md)). Because z has fewer dimensions than x, the network cannot simply memorize an identity function — it's forced to discover which features of the input are actually important to preserve to reconstruct it well.

**Why it mattered.** Basic autoencoders demonstrated that neural networks could learn useful compressed representations without any labels at all, using reconstruction as a free, automatically-available training signal — a foundational idea for self-supervised learning broadly (see [pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md)).

**Current status.** Rarely used in their most basic form for production systems today, but they are the direct conceptual and architectural ancestor of Variational Autoencoders (a probabilistic extension — see [vaes.md](../04-generative-models/vaes.md)), and the encoder/decoder bottleneck framing recurs throughout modern architectures.

**Relationship to other algorithms.** Basic autoencoders are deterministic (a given input always maps to the same latent vector); VAEs (see [vaes.md](../04-generative-models/vaes.md)) make this probabilistic, which is what turns an autoencoder from a compression tool into a proper generative model capable of producing new, realistic samples.

## Comparison table

| Method | Finds | Requires cluster count upfront? | Handles non-spherical clusters? |
|---|---|---|---|
| k-means | Flat clusters | Yes | No |
| Hierarchical clustering | Nested cluster tree | No (choose cut point after) | Depends on linkage |
| DBSCAN | Density-based clusters + outliers | No | Yes |
| PCA | Linear low-dim projection | N/A (choose # components) | N/A (linear only) |
| t-SNE | Nonlinear 2D/3D visualization | N/A | N/A (local structure only) |
| UMAP | Nonlinear low-dim projection/visualization | N/A | N/A |

## Relationship to other algorithms

- k-means is a special case of the Expectation-Maximization framework covered fully in [probabilistic-models.md](probabilistic-models.md).
- t-SNE's use of KL divergence to match neighbor-probability distributions connects to [loss-functions.md](../01-foundations/loss-functions.md).
- Autoencoders are the direct predecessor of VAEs — see [vaes.md](../04-generative-models/vaes.md).
- PCA is conceptually related to the linear-algebra intuition behind embeddings (see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)), though embeddings are learned rather than derived from a closed-form eigen-decomposition.

## Sources

- Lloyd, "Least Squares Quantization in PCM" (Bell Labs technical report, 1957; published 1982); MacQueen, "Some Methods for Classification and Analysis of Multivariate Observations" (1967)
- Ester, Kriegel, Sander, Xu, "A Density-Based Algorithm for Discovering Clusters" (1996) [DBSCAN]
- Pearson, "On Lines and Planes of Closest Fit to Systems of Points in Space" (1901); Hotelling, "Analysis of a Complex of Statistical Variables into Principal Components" (1933)
- van der Maaten, Hinton, "Visualizing Data using t-SNE" (2008)
- McInnes, Healy, Melville, "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction" (2018)
