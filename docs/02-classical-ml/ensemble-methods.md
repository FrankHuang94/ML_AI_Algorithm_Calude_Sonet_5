# Ensemble Methods

An ensemble combines many individually weak or imperfect models into one stronger predictor. This family — particularly gradient-boosted decision trees — is the single most important piece of context for understanding why deep learning has *not* taken over tabular data (data organized in rows/columns with a mix of numeric and categorical features, like a spreadsheet or a database table), even in 2026.

## Bagging (Bootstrap Aggregating)

**Name & definition.** Bagging trains many copies of the same model type on different random resamples of the training data (sampled with replacement — a "bootstrap sample") and averages (regression) or majority-votes (classification) their predictions.

**Origin.** Breiman, "Bagging Predictors" (1996).

**Core mechanism.** Draw B bootstrap samples from the training set (each the same size as the original, sampled with replacement, so each sample omits some original examples and duplicates others), train one model per sample, and average the B models' predictions at inference time.

**Why it mattered.** Averaging many models trained on slightly different data reduces **variance** (how much a model's predictions would change if trained on a different random sample of data) without increasing **bias** (systematic error from the model's assumptions being wrong) — a mathematically clean way to make an unstable model (like a single decision tree) much more stable, since the models' individual errors partially cancel out when averaged.

**Current status.** Rarely used in isolation, but is the core mechanism inside Random Forest (below).

## Random Forest

**Name & definition.** Random Forest is bagging applied specifically to decision trees, with an added twist: at each split in each tree, only a random subset of features is considered, rather than all of them.

**Origin.** Breiman, "Random Forests" (2001), building on his own bagging work and on earlier "random subspace" ideas (Ho, 1998).

**Core mechanism.** Train many decision trees (see [supervised-learning.md](supervised-learning.md)), each on a bootstrap sample of the data and each considering only a random subset of features at every split. Average (or vote) their predictions.

**Why it mattered.** The random feature-subset restriction (on top of bagging's data resampling) further de-correlates the individual trees — without it, if one feature is very predictive, nearly every tree in the bag would use it at the top split and the trees would end up highly similar to each other, limiting how much variance reduction averaging can provide. With decorrelated trees, the variance-reduction benefit of averaging is much larger.

**Current status.** Still a strong, very commonly-used baseline for tabular data — simple to train, hard to badly misconfigure, and resistant to overfitting even with relatively little hyperparameter tuning. Generally outperformed by gradient boosting (below) when the extra tuning effort is worth it, but Random Forest remains the "reach for this first, tune later if needed" choice in many practical settings.

**Strengths & limitations.**
- Strengths: resistant to overfitting; requires minimal tuning; handles mixed feature types natively; provides useful feature-importance estimates.
- Limitations: usually beaten by well-tuned gradient boosting on accuracy; large forests can be slow and memory-heavy at inference time; less interpretable than a single tree.

## Boosting (concept) and AdaBoost

**Name & definition.** Boosting builds an ensemble *sequentially* rather than in parallel: each new model is trained specifically to correct the errors of the ensemble built so far, rather than being an independent resample like bagging.

**AdaBoost** ("Adaptive Boosting" — Freund and Schapire, 1997) is the original practical boosting algorithm: it trains a sequence of "weak learners" (often very shallow trees, sometimes just a single split — "decision stumps"), and after each one, it increases the weight of training examples the current ensemble got wrong, so the next weak learner is forced to focus on the hard cases. The final prediction is a weighted vote of all the weak learners, with more accurate learners given more voting weight.

**Why it mattered.** AdaBoost was the first algorithm to demonstrate, with real theoretical backing, that combining many learners only slightly better than random guessing could produce an arbitrarily accurate ensemble — a striking and influential result at the time.

**Current status.** Largely superseded in practice by gradient boosting (below), which generalizes the same "sequentially correct past mistakes" idea into a more flexible and effective optimization framework, but AdaBoost remains the historically important entry point to the whole boosting family.

## Gradient Boosting

**Name & definition.** Gradient boosting generalizes AdaBoost's "sequentially fix past mistakes" idea by having each new model in the sequence fit the *gradient* of the loss function with respect to the current ensemble's predictions — effectively, each new tree is trained to predict the residual errors of the ensemble so far.

**Origin.** Friedman, "Greedy Function Approximation: A Gradient Boosting Machine" (2001).

**Core mechanism.** Start with a simple initial prediction (e.g., the mean of the target). At each boosting round, compute the gradient of the loss with respect to the current ensemble's predictions for every training example (for MSE loss, this gradient is just proportional to the residual — true value minus current prediction); train a new small tree to predict that gradient; add the new tree's predictions to the ensemble, scaled by a small learning rate (an important regularizer — see [regularization-techniques.md](../01-foundations/regularization-techniques.md) — that prevents any single round from overcorrecting). Repeat for many rounds.

Walkthrough: instead of hand-crafting how to correct mistakes (as AdaBoost does via example reweighting), gradient boosting frames "correcting mistakes" as literally minimizing a loss function via gradient descent — except the "step" at each iteration is an entire new tree, not a small numeric nudge to existing parameters. This reframing is what lets gradient boosting work with any differentiable loss function (not just the exponential loss implicit in classic AdaBoost), making it far more general.

**Why it mattered.** This generalization made boosting applicable to regression, ranking, and any other task with a sensible loss function, and produced the single best-performing family of algorithms for tabular data through the 2010s and into the 2020s.

**Current status.** Dominant for tabular data as of 2026, primarily via its modern, heavily-optimized implementations below.

### XGBoost

**Origin.** Chen and Guestrin, "XGBoost: A Scalable Tree Boosting System" (2016).

**Core contribution.** XGBoost is an engineering-optimized, regularized implementation of gradient boosting: it adds explicit L1/L2 regularization on the tree structure itself (penalizing the number of leaves and the magnitude of leaf output values), uses a more principled approximation to find good splits efficiently (including for large or sparse datasets), and was engineered from the ground up for speed and parallelism (parallel split-finding within a tree, out-of-core computation for data too large for memory, and hardware-aware optimizations).

**Why it mattered.** XGBoost's combination of accuracy and engineering polish made it the dominant tool in machine learning competitions (e.g., on the Kaggle platform) for tabular data from around 2015 onward, and it remains a default first choice in production tabular ML.

### LightGBM

**Origin.** Ke et al. (Microsoft), "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (2017).

**Core contribution.** LightGBM speeds up training significantly, especially on large datasets, primarily through two innovations: **Gradient-based One-Side Sampling (GOSS)** (keep all training examples with large gradients — the ones the model is currently getting most wrong — but only randomly subsample the examples with small gradients, since they contribute less new information per training round) and **Exclusive Feature Bundling (EFB)** (combine sparse, mutually-exclusive features into single bundled features to reduce the effective feature count). It also grows trees leaf-wise (always splitting whichever leaf gives the greatest loss reduction next) rather than level-wise (splitting every leaf at the current depth before going deeper, as XGBoost traditionally did), which tends to reach lower loss faster for the same number of leaves, at some added overfitting risk on smaller datasets.

**Why it mattered.** Meaningfully faster training on large datasets than XGBoost at the time of release, without giving up much (often any) accuracy — this made gradient boosting practical on datasets that had previously been too large to iterate on quickly.

### CatBoost

**Origin.** Prokhorenkova et al. (Yandex), "CatBoost: Unbiased Boosting with Categorical Features" (2018).

**Core contribution.** CatBoost's headline feature is native, statistically principled handling of categorical features (features that take on discrete labels rather than numbers, like "country" or "product category") without requiring manual preprocessing like one-hot encoding. It uses an ordered, target-statistic-based encoding scheme specifically designed to avoid a subtle form of data leakage (where information about the target variable inadvertently "leaks" into features in a way that inflates training performance but doesn't generalize) that naive categorical-encoding schemes are prone to.

**Why it mattered.** For datasets rich in categorical features (a very common situation in real-world business data — e.g., product IDs, geographic regions, user segments), CatBoost often requires the least manual feature engineering of the three major gradient boosting libraries while matching or exceeding their accuracy.

## Stacking and blending

**Name & definition.** Stacking trains a "meta-model" to combine the predictions of several different base models (which can be entirely different algorithm types — e.g., a Random Forest, a gradient-boosted tree, and a neural network), learning how much to trust each one, rather than using a simple fixed average or vote.

**Core mechanism.** Train several diverse base models. Generate their predictions on held-out data (via cross-validation, to avoid the meta-model just learning to trust whichever base model overfit training data hardest). Train a final meta-model (often something simple, like logistic/linear regression) whose inputs are the base models' predictions and whose target is the original label.

**Blending** is a simpler, less rigorous variant that uses a single held-out validation split rather than full cross-validation to generate the meta-model's training data — faster and simpler to implement, at some cost in robustness.

**Why it mattered.** Stacking can extract a bit of extra accuracy by combining genuinely different model types whose errors are not perfectly correlated — a boosted tree and a neural network tend to make different kinds of mistakes, and a meta-model can learn when to lean on which.

**Current status.** Common in competitive ML (Kaggle-style competitions, where squeezing out marginal accuracy gains is the entire point) but less common in production, where the added complexity, latency, and maintenance burden of multiple models often isn't worth a small accuracy gain.

## Comparison table

| Method | Training speed | Interpretability | Typical use case |
|---|---|---|---|
| Bagging | Fast (parallelizable) | Low (many models) | Rarely used alone |
| Random Forest | Fast (parallelizable) | Moderate (feature importances) | Strong, low-effort tabular baseline |
| AdaBoost | Moderate (sequential) | Low | Largely historical/educational today |
| Gradient Boosting (general) | Slower (sequential) | Low | Superseded by XGBoost/LightGBM/CatBoost in practice |
| XGBoost | Fast (optimized, parallel split-finding) | Low | Default choice for tabular ML; competition-proven |
| LightGBM | Fastest on large data | Low | Large tabular datasets, fast iteration |
| CatBoost | Fast; efficient with categoricals | Low | Categorical-feature-heavy tabular data |
| Stacking/blending | Slow (trains multiple models + meta-model) | Very low | Competitions; occasionally production when justified |

## Why tabular data still belongs to trees, not deep learning

Gradient-boosted trees remain competitive with or superior to deep learning on most tabular datasets for a few structural reasons worth naming explicitly: tabular features are often heterogeneous and non-smooth (unlike pixels or audio samples, adjacent "feature values" like customer ID or zip code have no meaningful continuity for a neural network to exploit), tabular datasets are frequently small relative to the parameter counts needed for a competitive neural network, and trees handle missing values and mixed feature types with far less preprocessing effort. This is a genuinely important, non-obvious fact about the field: the "deep learning wins everywhere" narrative common outside ML circles is specifically false for tabular data, and this remains true as of 2026.

## Relationship to other algorithms

- Decision trees, the base learner here, are covered in [supervised-learning.md](supervised-learning.md).
- The learning-rate-as-regularizer idea in gradient boosting parallels learning rate schedules in [optimization-algorithms.md](../01-foundations/optimization-algorithms.md), though the mechanics of applying it (scaling a whole new tree's contribution vs. scaling a gradient step) differ.
- See [algorithm-comparison-master.md](../10-comparison-tables/algorithm-comparison-master.md) for a cross-cutting comparison including these methods alongside deep learning architectures.

## Sources

- Breiman, "Bagging Predictors" (1996)
- Breiman, "Random Forests" (2001)
- Ho, "The Random Subspace Method for Constructing Decision Forests" (1998)
- Freund, Schapire, "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting" (1997) [AdaBoost]
- Friedman, "Greedy Function Approximation: A Gradient Boosting Machine" (2001)
- Chen, Guestrin, "XGBoost: A Scalable Tree Boosting System" (2016)
- Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (2017)
- Prokhorenkova et al., "CatBoost: Unbiased Boosting with Categorical Features" (2018)
