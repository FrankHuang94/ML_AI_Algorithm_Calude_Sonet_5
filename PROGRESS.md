# Build Progress (scratch file — not part of the published reference)

This file tracks build status for the ML/AI Algorithm Knowledge Base so work can resume across sessions without re-reading the whole repo. Update it after finishing each file or group of files.

## Status: In progress — sections 00-03 complete, working through 04

## Running word count
~31,400 words (via `find docs -name "*.md" | xargs wc -w`, includes remaining placeholder files' boilerplate). Floor is 50,000. On track.

## Files completed
- [x] Scaffold (all placeholders)
- [x] 00-overview: scope-and-methodology.md, taxonomy-map.md, glossary.md (living, actively appended)
- [x] 01-foundations: optimization-algorithms.md, regularization-techniques.md, loss-functions.md
- [x] 02-classical-ml: supervised-learning.md, ensemble-methods.md, unsupervised-learning.md, probabilistic-models.md
- [x] 03-deep-learning-architectures: cnn-family.md, rnn-lstm-gru.md, transformer-architecture.md (cornerstone, extra depth), mixture-of-experts.md, state-space-models.md, graph-neural-networks.md

## Files remaining (in build order)
- [ ] 04-generative-models: gans.md, vaes.md, diffusion-models.md, autoregressive-generation.md, flow-based-models.md
- [ ] 05-training-methodology: pretraining-strategies.md, finetuning-and-peft.md, rlhf-and-alignment.md, distributed-training.md, curriculum-and-data-strategies.md
- [ ] 06-inference-optimization: quantization.md, pruning-and-distillation.md, kv-cache-and-attention-optimization.md, speculative-decoding.md, serving-and-batching.md
- [ ] 07-reinforcement-learning: value-based-methods.md, policy-gradient-methods.md, model-based-rl.md, rl-for-llms.md
- [ ] 08-history: timeline-1950s-2000s.md, timeline-2000s-2017.md, timeline-2017-2023.md, timeline-2023-present.md (each needs a Mermaid timeline diagram)
- [ ] 09-roadmaps: near-term-outlook.md, long-term-outlook.md, open-problems.md ([Projection] labeling required)
- [ ] 10-comparison-tables: algorithm-comparison-master.md, compute-cost-comparison.md, when-to-use-what.md (write LAST, synthesis only)
- [ ] README.md master index (write after all content done)
- [ ] Final consistency pass + CHANGELOG.md

## Open threads / cross-links to revisit later
- Every written file already cross-links forward to files not yet written (e.g., vaes.md, rlhf-and-alignment.md, kv-cache-and-attention-optimization.md) — these forward links use the filenames specified in the build spec, so they should resolve correctly once those files exist. Spot-check a sample during the final consistency pass.
- taxonomy-map.md mindmap was written early and references section names generically; re-check it still matches reality once everything is written.
- Comparison tables (section 10) intentionally deferred — they synthesize from everything else and must be written last.
- README.md intentionally deferred until all docs exist so its TOC/descriptions are accurate.

## Glossary status
`docs/00-overview/glossary.md` is live and has been appended to after every content file so far (~35 terms defined as of this checkpoint). Keep alphabetization correct — past mistakes (duplicate letter headers, misplaced entries) have already been caught and fixed once; double-check when adding new terms that the entry goes under the correct existing letter header rather than creating a duplicate.

## Word-count pacing note
Foundations/classical-ml/architecture files have averaged ~1,800-3,400 words each. At this rate, the remaining ~26 content files plus README should comfortably clear the 50,000-word floor without padding.
