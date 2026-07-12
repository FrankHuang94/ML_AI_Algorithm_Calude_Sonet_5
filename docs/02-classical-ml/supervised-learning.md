# Supervised Learning

Supervised learning is the task of learning a function from labeled examples — pairs of inputs and correct outputs — that generalizes to new, unseen inputs. This file covers the foundational classical (pre-deep-learning) algorithms for this task. They remain in wide, non-nostalgic use today, especially on tabular data (spreadsheet-like data with rows of examples and columns of named features), where deep learning frequently does *not* beat them (see [ensemble-methods.md](ensemble-methods.md) for why).

The fastest way to build intuition for *how these classifiers differ in character* is to look at the shape of the **decision boundary** each one draws — the boundary is the dividing surface in feature space where the model switches from predicting one class to another. Picture a 2D dataset with two classes (`o` and `x`); here is roughly how each classifier separates them:

```
 Logistic regression      Decision tree           k-NN                  SVM (RBF kernel)
 (one straight line)      (axis-aligned boxes)    (irregular, local)    (smooth curved margin)
   o o o │ x x x            o o │ x x x             o o o⌐x x x            o o o ╱ x x x
   o o o │ x x x            o o │ x x x             o o⌐x⌐x x x            o o ╱  x x x
   o o o │ x x x          ──────┼─────             o⌐o x x⌐x x            o o│  x x x x
   o o o │ x x x            o o o│x x             o o⌐o⌐o x x x            o o ╲  x x x
         │ (a single       ──────┴─── (staircase   (boundary hugs        o o o ╲ x x x
          linear cut)       of right angles)        individual points)   (one smooth curve)
```

Read off each algorithm's personality: logistic regression can only draw a single straight cut (a linear boundary); a decision tree draws a staircase of axis-aligned rectangles; k-NN's boundary is jagged and wraps tightly around individual training points; an SVM with an RBF kernel draws one smooth curve with as wide a margin as possible. When you read "this model can't capture nonlinearity without feature engineering," that just means "its boundary is stuck being a straight line." Keep this picture in mind as you read each entry below.

A second framing worth internalizing up front, because it cuts across several of these algorithms: the distinction between **discriminative** and **generative** classifiers. A discriminative model (logistic regression, SVMs, trees) learns the boundary between classes directly — it answers "given these features, which class?" A generative model (Naive Bayes, and the probabilistic models in [probabilistic-models.md](probabilistic-models.md)) instead learns what each class's data *looks like* — it models "what features does a spam email tend to have?" — and then uses Bayes' rule to flip that around into a classification. Discriminative models usually win on raw predictive accuracy when data is plentiful; generative models can need less data and naturally produce calibrated probabilities and handle missing features. This distinction recurs throughout the repository, all the way up to the difference between how a classifier head and a generative language model are trained.

## Linear regression

**Name & definition.** Linear regression predicts a continuous numeric output as a weighted sum of input features.

**Origin.** The method of least squares dates to Legendre (1805) and Gauss (early 1800s), long predating computing — one of the oldest algorithms in this entire repository.

**Core mechanism.** ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b, trained by minimizing MSE loss (see [loss-functions.md](../01-foundations/loss-functions.md)) between predictions and true values. For this simple case, the optimal weights can be computed in closed form (the "normal equation," via linear algebra) rather than needing iterative gradient descent, though gradient-based fitting is used in practice at scale. Walkthrough: each feature gets a learned coefficient describing how much it pushes the prediction up or down; b is a baseline offset.

**Why it mattered.** It's the simplest possible model that captures the idea of "learn weights from data to predict a number," and its assumptions (linearity, independent errors) make it mathematically tractable enough to have a rich body of statistical theory (confidence intervals, significance tests) built around it that purely predictive models like neural networks generally lack.

**Current status.** Still genuinely useful whenever interpretability matters or the relationship really is close to linear (or has been feature-engineered to be); a common baseline against which more complex models are judged.

**Strengths & limitations.**
- Strengths: interpretable (each weight has a direct, understandable meaning), fast to train, no hyperparameter tuning required, well-understood statistical guarantees.
- Limitations: cannot capture nonlinear relationships without manual feature engineering; sensitive to outliers (because of the squared-error loss); assumes features are not too correlated with each other (multicollinearity distorts coefficient estimates).

**Relationship to other algorithms.** Logistic regression is linear regression's classification counterpart (below); linear regression with L1/L2 penalties becomes LASSO/ridge regression (see [regularization-techniques.md](../01-foundations/regularization-techniques.md)).

## Logistic regression

**Name & definition.** Logistic regression is a linear model for binary (or multi-class) classification: it predicts the *probability* of a class by passing a linear combination of features through the sigmoid function.

**Origin.** The logistic function itself dates to 19th-century population-growth modeling (Verhulst, 1830s-40s); its use as a statistical classification model was formalized in the mid-20th century and became a standard statistics and ML tool from the 1970s-80s onward.

**Core mechanism.**

```
p = σ(w·x + b) = 1 / (1 + e^−(w·x + b))
```

trained by minimizing binary cross-entropy loss. Walkthrough: w·x + b produces an unbounded real number (a **logit**); the sigmoid function σ squashes it into the range (0, 1), giving a valid probability. Despite the name, logistic regression is a classification algorithm, not a regression algorithm — the "regression" in the name is a historical artifact from its statistical origins.

**Why it mattered.** It's the simplest model that connects a linear decision boundary to a properly calibrated probability output, and it directly generalizes to the final layer of nearly every deep classification network (a deep net's last layer is typically just logistic/softmax regression applied to a learned, nonlinear feature representation — the earlier layers do the feature engineering the classical version required by hand).

**Current status.** Widely used directly for simple classification tasks, as a baseline, and in latency-critical applications where a full neural network is unnecessary overhead; also the conceptual final layer of virtually every neural classifier.

**Relationship to other algorithms.** Generalizes to softmax regression for multi-class problems (see [loss-functions.md](../01-foundations/loss-functions.md)); is, in effect, "the last layer of a neural network" once you view deep learning as learned feature extraction followed by a simple linear classifier.

## Decision trees

**Name & definition.** A decision tree predicts an output by asking a sequence of if/then questions about the input features, structured as a binary (or multi-way) tree, arriving at a prediction at each leaf.

**Origin.** ID3 (Quinlan, 1986), C4.5 (Quinlan, 1993), and CART (Classification and Regression Trees — Breiman et al., 1984) are the three classic tree-induction algorithms.

**Core mechanism.** Trees are built greedily, top-down: at each node, the algorithm considers every feature and every possible split point, and picks the split that best separates the data into purer subgroups (subgroups more dominated by a single class, for classification, or with lower variance, for regression). "Purity" is measured differently across variants: **ID3** and **C4.5** use **information gain** (the reduction in entropy — see [loss-functions.md](../01-foundations/loss-functions.md) — achieved by a split); **CART** typically uses the **Gini impurity** (the probability of misclassifying a randomly chosen element if you labeled it randomly according to the subgroup's class distribution) for classification, or variance reduction for regression. The process recurses on each resulting subgroup until a stopping criterion is met (max depth, minimum leaf size, or no further improvement available).

Walkthrough: imagine deciding whether to approve a loan by first asking "is income above $50k?", then within each branch asking a further question like "is credit score above 700?", and so on, until you reach a leaf that says "approve" or "deny." Each question is chosen, at training time, specifically because it does the best job of separating approved from denied cases at that point in the data.

**Why it mattered.** Trees produce a model that's directly human-readable (you can literally read off the decision logic) and require no feature scaling or distributional assumptions — a sharp contrast to linear/logistic regression's linearity assumption.

**Current status.** Rarely used standalone in production today (a single tree tends to overfit and has high variance — small changes in training data can produce a very different tree) but is the fundamental building block for the ensemble methods (Random Forest, gradient boosting) that dominate tabular ML — see [ensemble-methods.md](ensemble-methods.md).

**Strengths & limitations.**
- Strengths: interpretable, handles mixed numeric/categorical features natively, no scaling needed, captures nonlinear interactions automatically.
- Limitations: prone to overfitting (a deep enough tree can memorize training data exactly); unstable (small data changes can produce very different trees); poor extrapolation outside the range of training data.

## Support Vector Machines (SVMs)

**Name & definition.** An SVM is a classifier that finds the decision boundary between classes that maximizes the margin — the distance between the boundary and the nearest training examples of each class.

**Origin.** Cortes and Vapnik, "Support-Vector Networks" (1995), building on earlier margin-maximization theory from Vapnik and Chervonenkis dating to the 1960s-70s.

**Core mechanism.** For linearly separable data, an SVM finds the hyperplane w·x + b = 0 that maximizes the margin between the two classes, trained by minimizing hinge loss (see [loss-functions.md](../01-foundations/loss-functions.md)) with an L2 penalty on w. The examples closest to the boundary — the ones that actually determine where it sits — are called **support vectors**; every other training example could be deleted without changing the learned boundary at all.

**The kernel trick (conceptual).** Real data is often not linearly separable in its original feature space. The **kernel trick** is a mathematical shortcut that lets an SVM effectively operate as if the data had been transformed into a much higher-dimensional space (where a linear separator might exist) — without ever actually computing that transformation explicitly. This works because the SVM's math only ever needs the *dot products* between pairs of transformed points, not the transformed points themselves, and a kernel function can compute that dot product directly and cheaply from the original (untransformed) inputs. Common kernels include the polynomial kernel and the RBF (radial basis function, or Gaussian) kernel, which effectively measures similarity as a smooth function of distance and can represent extremely flexible, locally-adaptive decision boundaries.

**Why it mattered.** Before deep learning's rise, SVMs (especially with the RBF kernel) were often the best-performing classifier available for a wide range of small-to-medium-sized, moderately high-dimensional datasets, and the margin-maximization framing gave the field strong generalization guarantees that were rarer in earlier ML methods.

**Current status.** Niche today — displaced by gradient-boosted trees for most tabular problems (see [ensemble-methods.md](ensemble-methods.md)) and by deep learning for anything involving raw images, audio, or text. Still used in some smaller-data or high-dimensional-but-limited-sample scientific and bioinformatics settings where its generalization properties remain attractive.

**Strengths & limitations.**
- Strengths: strong theoretical generalization guarantees; effective in high-dimensional spaces; kernel trick allows flexible nonlinear boundaries without explicit feature engineering.
- Limitations: doesn't scale well to very large datasets (naive training is roughly quadratic-to-cubic in the number of examples); kernel choice and hyperparameters (like the regularization strength and kernel-specific parameters) require tuning; less interpretable than trees or linear models; no natural probability output (probability estimates require an extra calibration step).

## k-Nearest Neighbors (k-NN)

**Name & definition.** k-NN classifies (or predicts a value for) a new point by looking at the k closest points in the training set and taking a majority vote (classification) or an average (regression).

**Origin.** Fix and Hodges (1951, technical report), later formalized by Cover and Hart, "Nearest Neighbor Pattern Classification" (1967).

**Core mechanism.** No training phase in the usual sense — k-NN simply stores the entire training set ("lazy learning" or "instance-based learning"). At prediction time, it computes the distance (commonly Euclidean) from the query point to every training point, finds the k nearest, and aggregates their labels.

**Why it mattered.** It's the simplest possible nonparametric method (making no assumption about the functional form of the relationship between inputs and outputs) and is still a useful conceptual baseline: "what would the simplest possible similarity-based prediction look like?"

**Current status.** Rarely the best choice for high-dimensional or large-scale problems today (distance metrics become less meaningful in high dimensions — the "curse of dimensionality" — and storing/searching the full training set at inference time is expensive), but the underlying "look up similar examples" idea is very much alive in modern retrieval-augmented systems and approximate nearest-neighbor search over embeddings.

**Strengths & limitations.**
- Strengths: no training required; simple and intuitive; naturally handles multi-class problems.
- Limitations: slow at inference time on large datasets (naive search is linear in dataset size, though approximate methods mitigate this); performance degrades in high-dimensional spaces; sensitive to irrelevant/unscaled features and to the choice of k and distance metric.

## Naive Bayes

**Name & definition.** Naive Bayes is a probabilistic classifier that applies Bayes' rule under the ("naive") simplifying assumption that all features are conditionally independent given the class.

**Origin.** Rooted in Bayesian probability theory from the 18th-19th centuries; applied to text classification (e.g., spam filtering) starting in the 1990s-2000s.

**Core mechanism.** By Bayes' rule, P(class | features) ∝ P(class) · P(features | class). The "naive" assumption lets you factor P(features | class) as the product of each individual feature's probability given the class: P(f₁|class)·P(f₂|class)·...·P(fₙ|class), which is vastly cheaper to estimate from data than the full joint distribution over all features. Walkthrough: to classify an email as spam, you multiply together how likely each individual word in it is to appear in spam emails versus non-spam emails (each estimated independently from training data), weighted by the overall spam vs. non-spam base rates, and pick whichever class has the higher resulting score.

**Why it mattered.** Despite an assumption (feature independence given the class) that's almost never literally true, Naive Bayes works surprisingly well in practice for many text classification tasks, is extremely fast to train (closed-form probability estimates, no iterative optimization needed) and requires very little data relative to more expressive models.

**Current status.** Displaced by more powerful deep learning and gradient-boosted approaches for most serious classification work, but still used as a fast, simple, well-calibrated baseline, especially for lightweight text classification (e.g., basic spam filters) where its speed and low data requirements are valuable.

**Relationship to other algorithms.** Its probabilistic (generative, in the sense of modeling P(features | class) rather than directly modeling the decision boundary) framing connects it to the probabilistic models covered in [probabilistic-models.md](probabilistic-models.md); it's a useful conceptual contrast to logistic regression, which models P(class | features) directly (a "discriminative" model) rather than going through P(features | class).

## Comparison table

| Algorithm | Year | Interpretable? | Handles nonlinearity? | Typical use today |
|---|---|---|---|---|
| Linear regression | 1805 | Yes | No (without manual features) | Baselines, interpretable numeric prediction |
| Logistic regression | ~1830s (function); ~1940s-70s (as classifier) | Yes | No (without manual features) | Baselines, final layer of neural classifiers |
| Decision trees | 1984-93 | Yes | Yes | Rare standalone; core of ensembles |
| SVM | 1995 | Low | Yes (via kernel trick) | Niche — small/medium high-dim data |
| k-NN | 1951/1967 | Moderate | Yes (locally) | Baselines; conceptual basis of retrieval systems |
| Naive Bayes | Bayes: 18th c.; ML use: 1990s-2000s | Yes | No (assumes independence) | Lightweight text classification, baselines |

## Relationship to other algorithms

- Decision trees are the base learner for [ensemble-methods.md](ensemble-methods.md) (Random Forest, gradient boosting).
- Logistic regression's sigmoid/softmax output connects directly to [loss-functions.md](../01-foundations/loss-functions.md) and is architecturally embedded as the final layer of most neural classifiers (see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)).
- Naive Bayes is a bridge to the fully probabilistic models in [probabilistic-models.md](probabilistic-models.md).
- k-NN's "find similar examples" idea underlies modern retrieval-augmented generation approaches referenced in [pretraining-strategies.md](../05-training-methodology/pretraining-strategies.md).

## Sources

- Legendre, "Nouvelles méthodes pour la détermination des orbites des comètes" (1805) [least squares]
- Verhulst, "Notice sur la loi que la population suit dans son accroissement" (1838) [logistic function]
- Quinlan, "Induction of Decision Trees" (1986) [ID3]; "C4.5: Programs for Machine Learning" (1993)
- Breiman, Friedman, Olshen, Stone, "Classification and Regression Trees" (1984) [CART]
- Cortes, Vapnik, "Support-Vector Networks" (1995)
- Fix, Hodges, "Discriminatory Analysis: Nonparametric Discrimination" (1951); Cover, Hart, "Nearest Neighbor Pattern Classification" (1967)
