# Curriculum and Data Strategies

As the Chinchilla-era scaling-law discussion in [pretraining-strategies.md](pretraining-strategies.md) makes clear, data quantity and quality are at least as important to a pretrained model's final performance as parameter count and architecture — this file covers the specific techniques used to curate, clean, order, and supplement the data that goes into training.

## Data curation and filtering

**Name & definition.** Data curation is the process of selecting, cleaning, and filtering raw data (commonly scraped from the web, alongside licensed and curated sources) before it's used for training, rather than training on everything available unfiltered.

**Why it matters.** Raw web-scraped text contains an enormous amount of low-quality, repetitive, spam-like, or otherwise unhelpful content, and filtering it out (using heuristic rules — like minimum length or specific formatting checks — and increasingly, learned quality classifiers trained to distinguish higher-quality from lower-quality text) has been widely reported to improve downstream model quality more than simply adding more unfiltered data would, especially since scaling laws (see [pretraining-strategies.md](pretraining-strategies.md)) generally assume the additional data is reasonably informative, not redundant or degenerate.

## Deduplication

**Name & definition.** Deduplication removes near-identical or exactly duplicated documents/passages from a training dataset.

**Why it matters at scale.** Large web-scraped datasets contain substantial duplication (the same news article syndicated across many sites, boilerplate legal text, repeated forum templates, and so on). Widely reported findings across multiple pretraining efforts indicate that heavily duplicated content can cause models to over-memorize that specific content (increasing the risk of verbatim regurgitation of training data) while providing comparatively little new information per additional token seen — duplicated tokens are, in a real sense, wasted training compute relative to equally-plentiful non-duplicated tokens. Deduplication is typically performed at multiple levels of granularity (exact-match deduplication, and "fuzzy"/near-duplicate detection using techniques like locality-sensitive hashing to catch documents that are nearly, but not exactly, identical) and is considered a standard, near-universal step in modern large-scale pretraining data pipelines.

## Curriculum learning (concept)

**Name & definition.** Curriculum learning trains a model on examples in a deliberately chosen order — typically easier or simpler examples first, progressing to harder or more complex ones — rather than presenting training data in a random or arbitrary order, based on the intuition (borrowed from how humans are often taught) that learning easier concepts first can make learning harder ones afterward more effective or efficient.

**Origin.** Bengio et al., "Curriculum Learning" (2009), which formalized the concept for machine learning broadly, though the general pedagogical intuition long predates this specific paper.

**Current status in large-scale pretraining.** Explicit, fine-grained per-example curriculum ordering (of the sort formalized in the original 2009 paper) is less commonly used for large-scale LLM pretraining specifically than a coarser-grained relative — deliberately structuring the overall *mix* and *order* of broad data sources and domains over the course of training (for example, some training recipes have reportedly emphasized certain data mixtures earlier or later in training, or introduced specific high-value data sources at particular training stages) — which is really a data-mixture-scheduling strategy more than a strict per-example difficulty curriculum. The narrower, textbook curriculum-learning idea remains more directly and clearly applied in some other training contexts (e.g., specific reinforcement learning setups, robotics, and some smaller-scale supervised settings) than in frontier LLM pretraining specifically.

## Synthetic data generation and self-distillation

**Name & definition.** Synthetic data generation uses one model (often a capable existing model) to generate training data — text, code, question-answer pairs, or other structured content — that's then used to train another model (or a later version of the same model).

**Self-distillation**, more specifically, refers to using a model's own outputs (or outputs from a stronger version of itself) as training signal, closely related to the distillation concept covered fully in [pruning-and-distillation.md](../06-inference-optimization/pruning-and-distillation.md), but applied here as a *data generation* strategy for pretraining/fine-tuning rather than purely as a model-compression technique.

**Why this has grown in importance.** As the highest-quality portions of naturally-occurring human-written text become more fully exploited (there's only so much genuinely high-quality, non-duplicated text on the internet), synthetic data generated by capable models has become an increasingly widely-reported tool for supplementing training data — particularly for structured domains like mathematics, code, and step-by-step reasoning, where a capable model can generate large quantities of correct, well-formatted worked examples that may be scarcer in naturally-occurring text. This comes with a real, actively-discussed risk worth naming plainly: training extensively on a model's own (or a similar model's) synthetic outputs, without careful quality control, risks amplifying that model's existing errors, quirks, or blind spots rather than correcting them — a failure mode sometimes discussed under the informal term "model collapse" in the research literature, where iteratively training on model-generated data without sufficient grounding in real, diverse data can degrade quality or diversity over successive generations.

## Data mixture weighting across domains

**Name & definition.** Real training datasets are blends of many different sources and domains (web text, code, books, scientific papers, dialogue data, and more), and the relative proportion — the **mixture weights** — allocated to each source is itself a significant design decision, not an afterthought.

**Why it matters.** Different domains contribute differently to different downstream capabilities (code data has been widely reported to help with certain kinds of structured/logical reasoning, for instance, beyond just code-generation ability itself), and the right mixture is generally determined empirically — via smaller-scale experiments measuring how changes to the mixture affect performance on a range of downstream evaluations — rather than derived from first principles. Mixture weights are also commonly adjusted over the course of training (a coarse-grained relative of the curriculum-learning idea above), for instance by up-weighting particular high-value domains during a later phase of pretraining.

**Current status.** Data mixture design remains a substantially empirical, experimentally-driven craft rather than a fully theorized science as of 2026 — this repository is explicit that there isn't yet a settled, general theory predicting optimal mixture weights from first principles, and practice in this area continues to be guided heavily by direct experimentation at each organization.

## Comparison table

| Strategy | What it targets | Primary benefit |
|---|---|---|
| Curation/filtering | Overall data quality | Removes low-value/spam content; improves quality-per-token |
| Deduplication | Redundancy | Reduces memorization risk; avoids wasting compute on repeated content |
| Curriculum/mixture scheduling | Order and timing of data exposure | Potentially more efficient learning progression |
| Synthetic data generation | Data scarcity in specific domains | Supplements naturally-occurring data, especially for structured domains (math, code) |
| Mixture weighting | Relative proportion of domains | Balances downstream capabilities across domains |

## Relationship to other algorithms

- This file's content is the practical complement to the scaling-law discussion in [pretraining-strategies.md](pretraining-strategies.md) and [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md) — scaling laws describe how much data helps, this file covers what that data should actually be.
- Self-distillation for data generation is conceptually related to (but distinct in purpose from) the model-compression distillation covered in [pruning-and-distillation.md](../06-inference-optimization/pruning-and-distillation.md).
- Data pipeline design interacts directly with the distributed training infrastructure covered in [distributed-training.md](distributed-training.md) (how sharded data is streamed to many parallel workers).

## Sources

- Bengio, Louradour, Collobert, Weston, "Curriculum Learning" (2009)
- Lee et al., "Deduplicating Training Data Makes Language Models Better" (2021)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022) [Chinchilla — data-scale motivation, cross-referenced from pretraining-strategies.md]
