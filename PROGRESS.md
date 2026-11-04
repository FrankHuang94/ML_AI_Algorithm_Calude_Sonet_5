# Build Progress (scratch file — not part of the published reference)

## Status: BUILD COMPLETE + v2 REVIEW PASS + v3 CURRENCY UPDATE (NOVEMBER 2026) COMPLETE

The v3 pass (see CHANGELOG "v3 — Currency update to November 2026") brought every section current through November 2026 — reasoning models and RLVR, the GRPO refinement family, on-policy distillation, hybrid linear-attention and sparse/compressed attention, MLA, FP8/MXFP4/NVFP4, MuonClip, diffusion language models, world models, TabPFN v2 — and mirrored all changes in the Chinese translation. Glossary at 147 terms. Internal links re-verified in both trees.

**Known follow-up:** parts of the Chinese translation that were *not* touched by v3 are still low-quality machine translation (notably untouched paragraphs of `transformer-architecture.md` and `optimization-algorithms.md`, and several files v3 did not need to change). A full Chinese retranslation pass would be worthwhile.

---

## (v2) Status: BUILD COMPLETE + v2 OPUS REVIEW PASS COMPLETE

The v2 pass (see CHANGELOG "v2 — Opus review-and-expand pass") reviewed every content file for correctness and expanded them for clarity: worked numeric examples, ~14 Mermaid diagrams plus many ASCII visuals, and genuinely current 2024-2025 material (Muon, WSD, flow matching, GRPO, VQ-VAE, tokenization/BPE, QK-norm, fine-grained MoE, DoRA, etc.). Content grew to ~78,000+ words. Glossary at ~105 terms, alphabetized. All internal links verified. All Mermaid blocks well-formed.

---

## (v1) Status: BUILD COMPLETE

All 45 planned content files are written, README.md master index is built, glossary is populated and alphabetized, cross-links verified, and CHANGELOG.md summarizes the build. See CHANGELOG.md for the full summary.

## Final stats
- ~69,600 words across `docs/` (via `find docs -name "*.md" | xargs wc -w`), well past the 50,000-word floor.
- 45/45 content files complete across sections 00-10.
- 681/681 internal relative links verified to resolve (script-checked during final consistency pass).
- Glossary: ~85 alphabetized terms, each linking to its covering file; alphabetization spot-checked and corrected where found out of order.
- Every file has at least one Mermaid diagram or substantial comparison table.

## If resuming work on this repository later
- This file (PROGRESS.md) and CONTRIBUTING.md describe the conventions to follow for any additions.
- Before adding a new algorithm/technique, check `docs/00-overview/glossary.md` first to keep terminology consistent with existing files.
- Re-run the link-check pattern used during the final consistency pass (walk all `.md` files, resolve relative links against their containing directory, resolve `/`-prefixed links against repo root) before committing structural changes.
