# Pretraining Strategies

Pretraining is the (usually enormous) initial training phase that gives a model its general capabilities, before any task-specific fine-tuning (see [finetuning-and-peft.md](finetuning-and-peft.md)) or alignment work (see [rlhf-and-alignment.md](rlhf-and-alignment.md)) happens. This file covers the self-supervised objectives used for pretraining and the scaling-law-driven practice of deciding how much compute to spend on model size vs. data.

## First, a prerequisite: tokenization

Before any of the objectives below can run, raw text has to be turned into the **tokens** the model actually consumes (the term is used throughout this repository — see the glossary — but how text *becomes* tokens deserves a brief explanation, since it's a genuine part of the pretraining pipeline). A model does not read characters or whole words; it reads a fixed vocabulary of a few tens of thousands of **subword** units produced by a **tokenizer**.

The dominant approach is **Byte-Pair Encoding (BPE)** (and close relatives like WordPiece and Unigram/SentencePiece). BPE is learned from the training corpus by a simple greedy procedure: start with individual characters (or bytes), then repeatedly find the most frequent adjacent pair and merge it into a new single token, until you've built up a vocabulary of the target size. Common words end up as single tokens ("the", "ing"), rarer words get split into a few pieces ("tokenization" → "token" + "ization"), and truly novel strings fall back to characters/bytes — which guarantees the tokenizer can represent *any* input without an "unknown word" problem.

Why this matters beyond plumbing: the tokenizer determines what "one step" of next-token prediction even means, and it has real downstream consequences — token count drives both context-window usage and API cost; languages that the tokenizer splits into many pieces (often non-English or code-heavy text) effectively get less efficient use of the context window; and quirks of tokenization are behind a surprising number of model failures (e.g., difficulty with character-level tasks like counting letters, because the model never sees individual letters, only subword chunks). Modern tokenizers are typically **byte-level** BPE, meaning they operate on raw bytes and so can encode any Unicode text, emoji, or binary-ish content without ever failing.

## Self-supervised learning, framed

**Name & definition.** Self-supervised learning trains a model using labels that are automatically derived from the input data itself, rather than requiring humans to manually annotate examples.

**Why this framing matters.** The entire modern pretraining paradigm exists because self-supervision lets models train on effectively unlimited quantities of raw, unlabeled data (text scraped from the web, images without captions, etc.) — sidestepping the fundamental bottleneck of earlier, purely supervised deep learning, where dataset size was capped by how much labeled data humans could feasibly produce. The two dominant self-supervised objectives for language are covered next; both were introduced already, in their architectural context, in [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md) — this file focuses on them specifically as training strategies.

## Masked language modeling (BERT-style)

**Core mechanism.** Randomly hide (mask) some fraction of tokens in an input sequence (commonly around 15%) and train the model to predict the original identity of each masked token, using the surrounding context from *both* directions (since the encoder's self-attention is unrestricted/bidirectional — see [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md)).

**Why it's self-supervised.** No human needs to label anything — the "label" for each masked position is simply the token that was originally there, which is automatically known since the masking was applied by the training pipeline itself, not by a human annotator.

**What it's good for.** Because the model learns to use context from both before and after a given position, masked language modeling produces representations well-suited to *understanding* tasks — classification, extracting a span of text that answers a question, producing embeddings for semantic search — but doesn't naturally support open-ended generation (see the decoder-only discussion in [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md) for why).

## Causal / next-token pretraining (GPT-style)

**Core mechanism.** Train the model to predict each token using only the tokens before it (causally-masked self-attention), as covered in depth in [autoregressive-generation.md](../04-generative-models/autoregressive-generation.md).

**Why it's dominant for general-purpose LLMs.** As discussed in [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md), this objective unifies understanding and generation in a single model and single training signal, at the cost of giving up masked language modeling's bidirectional context during pretraining itself.

## Contrastive pretraining: CLIP

**Name & definition.** Contrastive pretraining trains a model to produce matching representations for genuinely-related pairs of inputs (e.g., an image and its caption) and dissimilar representations for unrelated pairs — using the InfoNCE-style contrastive loss covered in [loss-functions.md](../01-foundations/loss-functions.md).

**Origin.** Radford et al. (OpenAI), "Learning Transferable Visual Models From Natural Language Supervision" (2021) — CLIP (Contrastive Language-Image Pre-training).

**Core mechanism.** CLIP trains two separate encoders — one for images, one for text — such that for a batch of (image, caption) pairs scraped from the web, the image embedding and text embedding for a genuinely-matched pair have high cosine similarity, while embeddings for mismatched pairs (an image paired with a random other caption from the batch) have low similarity. This is trained via the InfoNCE loss, treating "which caption in the batch actually matches this image" as the classification target.

**Why it mattered.** CLIP produced image and text representations that live in a shared embedding space, enabling zero-shot image classification (classify an image by comparing its embedding to the embeddings of candidate text labels, with no task-specific training at all — see the glossary for "zero-shot") and became a foundational building block for later multimodal systems, including as the text-conditioning mechanism inside several text-to-image diffusion systems (see [diffusion-models.md](../04-generative-models/diffusion-models.md)).

**Current status.** CLIP-style contrastive pretraining remains a standard technique for building multimodal embedding models and for connecting vision and language representations, though the largest frontier multimodal systems increasingly train vision and language jointly within a single large model rather than only via a separate CLIP-style contrastive stage.

## Scaling laws in practice: compute-optimal training

The Kaplan et al. (2020) and Chinchilla (Hoffmann et al., 2022) scaling laws are introduced in architectural context in [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md); this section focuses on what they mean for pretraining practice specifically.

**The practical question these laws answer.** Given a fixed compute budget (a real, hard constraint — training compute costs real money and takes real wall-clock time), how should you split that budget between making the model bigger (more parameters) versus training on more data (more tokens)? Before these scaling laws existed, this was largely a matter of trial and error, guesswork, and inherited convention from prior projects.

**The Chinchilla finding, restated for practice.** Hoffmann et al.'s central empirical result was that prior large models (following Kaplan et al.'s original guidance) were, in a specific and measurable sense, **undertrained relative to their size** — too many parameters for the amount of data they'd seen. Their compute-optimal recipe calls for scaling training data roughly in proportion to model size as compute grows (a commonly cited rule of thumb from the paper is approximately 20 training tokens per parameter, though — worth repeating from [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md) — this specific ratio is an empirical finding from a particular experimental setup, not a universal law, and shouldn't be treated as a rigid constant that applies unchanged to every architecture, data mixture, or training objective).

**Data scale vs. parameter scale tradeoffs, in practice.** Following Chinchilla-style guidance strictly implies training even larger models than before on even more data than before, for a truly compute-optimal outcome — but real pretraining decisions since 2022 have not followed pure compute-optimality alone. A model that will be served to enormous numbers of users for a long deployment lifetime has a strong incentive to be trained on *more* data than the naive compute-optimal ratio would suggest, even past the point of pretraining compute-optimality — because inference cost (paid repeatedly, across every future request the model serves) can dwarf the one-time cost of additional pretraining compute, and a smaller, more heavily-trained model that reaches similar quality is cheaper to run at scale. This is a widely-reported consideration behind why several capable, publicly-known models have been trained on data quantities well beyond the raw Chinchilla-optimal point for their size — trading a bit of extra pretraining compute for a model that's cheaper and faster to actually serve. Data quality, deduplication, and mixture composition (see [curriculum-and-data-strategies.md](curriculum-and-data-strategies.md)) also increasingly matter as much as raw data quantity, since scaling laws generally assume reasonably high-quality, non-redundant data — pushing more low-quality or highly-duplicated tokens through training does not reliably reproduce the same gains the scaling curves predict for genuinely diverse, high-quality data.

## 2025-26 developments in pretraining practice

The objectives above are still the core of pretraining as of late 2026. What changed most in 2025-26 is *how cheaply* that compute is spent, *what extra signal* is squeezed from each sequence, and *how the run is staged*. This section is a brief map; the linked files hold the mechanics.

### Low-precision pretraining: FP8, then 4-bit

**The idea.** Neural network training has long used **mixed precision** — doing the bulk of the matrix multiplications in a lower-precision number format (16-bit) while keeping a higher-precision "master" copy of the weights and a few numerically sensitive operations in 32-bit. Fewer bits per number means less memory, less data movement, and — on hardware with dedicated low-precision units — more arithmetic per second. The 2025-26 news is that this has been pushed much further down for *pretraining itself*, not just for inference (where shrinking numbers is called quantization — see [quantization.md](../06-inference-optimization/quantization.md)).

- **FP8 at scale.** DeepSeek-V3 (December 2024; a 671B-total / 37B-active Mixture-of-Experts model) was trained with FP8 (8-bit floating point) mixed precision at frontier scale, demonstrating that 8-bit training was viable for a very large production run rather than only in small experiments.
- **4-bit (NVFP4) pretraining.** NVIDIA's "Pretraining LLMs with NVFP4" (arXiv 2509.25149, September 2025; revised March 2026) trained a 12B hybrid Mamba-Transformer on 10 trillion tokens in the 4-bit **NVFP4** format and reported MMLU-Pro accuracy of 62.58%, versus 62.62% for an FP8 baseline — i.e., essentially matched. A 4-bit float has only 16 possible values, so the trick is **microscaling**: in NVFP4 every block of 16 consecutive numbers shares its own FP8 scale factor (plus one FP32 scale for the whole tensor), so each small block gets a range fitted to its own values. (The related **MXFP4** format uses 32-element blocks with a power-of-two scale.) The paper also relied on several stabilizers: random Hadamard transforms (a cheap "mixing" rotation that spreads out outlier values before rounding), 2-D block scaling, **stochastic rounding** on gradients (rounding up or down at random in proportion to closeness, so rounding errors average out instead of accumulating in one direction), keeping a few sensitive layers in higher precision, and switching to higher precision late in training.

Honest caveat: the 4-bit result is one reported run at 12B scale from a hardware vendor; FP8 is now established at frontier scale, whereas 4-bit pretraining at the very largest scales is still emerging.

### Multi-Token Prediction (MTP)

**The idea.** Standard causal pretraining asks the model, at each position, to predict only the *next* token. **Multi-Token Prediction (MTP)** adds extra prediction heads (small additional output modules) that also predict tokens further ahead — the token after next, and so on — from the same position. Meta researchers (Gloeckle et al., 2024) studied this as a pretraining objective, and DeepSeek-V3 (2024) used an MTP objective in a frontier-scale model. Two payoffs are claimed: a denser training signal per sequence (each position supervises several future tokens), and a free accelerator at inference time — the extra head can be reused as the "draft" in speculative decoding, proposing upcoming tokens that the main model then verifies (see [speculative-decoding.md](../06-inference-optimization/speculative-decoding.md)).

### "Mid-training": a stage between pretraining and post-training

Recent technical reports increasingly describe a distinct phase after the bulk of pretraining and before post-training (fine-tuning and RL), often called **mid-training** (the name is informal and the boundaries vary by lab). Generically, it is a continuation of the same next-token objective on a deliberately shifted data diet and setup:

```
 pretraining (bulk)            →  mid-training                         →  post-training
 trillions of tokens, mostly      smaller, curated, higher-quality         SFT, RLHF/RL on
 general web/code/books,          mix: more math, code, reasoning          verifiable rewards
 shorter context windows          traces; long-context extension           (finetuning-and-peft.md,
                                  (training on much longer sequences)       rlhf-and-alignment.md)
```

- **Long-context extension.** Attention cost grows with sequence length, so most pretraining tokens are processed with comparatively short context windows; the ability to handle very long inputs is added late, by training on longer sequences and adjusting the positional encoding (a practice already visible in 2024, e.g., Llama 3 extended its context in stages near the end of pretraining).
- **Reasoning-oriented data.** Up-weighting math, code, and step-by-step reasoning text (including model-generated reasoning traces — see [curriculum-and-data-strategies.md](curriculum-and-data-strategies.md)) gives the subsequent RL stage a stronger starting point.

This is really the data-mixture-scheduling idea from [curriculum-and-data-strategies.md](curriculum-and-data-strategies.md) given its own name and its own budget.

### Million-token context in open models

By 2026, open-weight models advertised context windows of around one million tokens. DeepSeek-V4 (technical report, April 2026, reported) is an example: an MoE model family with a 1M-token context, made affordable by a hybrid of compressed and sparse attention layers that keep the KV cache and per-token attention cost far below that of ordinary full attention at that length (mechanics in [kv-cache-and-attention-optimization.md](../06-inference-optimization/kv-cache-and-attention-optimization.md)). The usual caveat applies: an advertised context length says how much text the model *accepts*, not how reliably it *uses* information spread across all of it.

## Comparison table

| Objective | Attention pattern | Learns from | Best suited for |
|---|---|---|---|
| Masked language modeling | Bidirectional | Predicting hidden tokens from surrounding context | Understanding, embeddings, classification |
| Causal / next-token prediction | Causal (masked, forward-only) | Predicting each token from prior context | Open-ended generation, general-purpose LLMs |
| Contrastive (CLIP-style) | N/A (dual encoders) | Matching vs. mismatched pairs across modalities | Multimodal embeddings, zero-shot classification |
| Multi-Token Prediction (auxiliary) | Causal | Predicting several future tokens from each position | Denser training signal; built-in speculative-decoding draft head |

## Relationship to other algorithms

- Masked and causal pretraining objectives are the training-strategy counterpart of the encoder-only vs. decoder-only architecture discussion in [transformer-architecture.md](../03-deep-learning-architectures/transformer-architecture.md).
- CLIP's training objective is a direct application of the InfoNCE contrastive loss in [loss-functions.md](../01-foundations/loss-functions.md).
- Data quality/curation, referenced here as a factor moderating raw scaling-law predictions, is covered fully in [curriculum-and-data-strategies.md](curriculum-and-data-strategies.md).
- Post-pretraining stages (fine-tuning, RLHF) that build on top of a pretrained base model are covered in [finetuning-and-peft.md](finetuning-and-peft.md) and [rlhf-and-alignment.md](rlhf-and-alignment.md).
- Low-precision number formats (FP8, NVFP4, MXFP4) are explained from the inference side in [quantization.md](../06-inference-optimization/quantization.md); their effect on cluster-scale training is noted in [distributed-training.md](distributed-training.md).
- The MTP head's reuse as a draft model connects to [speculative-decoding.md](../06-inference-optimization/speculative-decoding.md).
- BERT-style masked language modeling is the ancestor of the masked-diffusion objective used by diffusion language models — see [diffusion-models.md](../04-generative-models/diffusion-models.md).

## Sources

- Sennrich, Haddow, Birch, "Neural Machine Translation of Rare Words with Subword Units" (2016) [BPE for NLP]
- Devlin, Chang, Lee, Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2018)
- Radford et al., "Improving Language Understanding by Generative Pre-Training" (2018) [GPT-1]
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021) [CLIP]
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022) [Chinchilla]
- Micikevicius et al., "Mixed Precision Training" (2018); Micikevicius et al., "FP8 Formats for Deep Learning" (2022)
- Gloeckle et al. (Meta), "Better & Faster Large Language Models via Multi-token Prediction" (2024)
- Llama Team (Meta), "The Llama 3 Herd of Models" (2024) [staged long-context extension near the end of pretraining]
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024) [FP8 training at scale; MTP objective]
- NVIDIA, "Pretraining LLMs with NVFP4" (arXiv 2509.25149, 2025; v2 March 2026) [4-bit pretraining; reported MMLU-Pro 62.58% vs. 62.62% FP8]
- DeepSeek-AI, DeepSeek-V4 technical report (April 2026, reported) [1M-token context]
