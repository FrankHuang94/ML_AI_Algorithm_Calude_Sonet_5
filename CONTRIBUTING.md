# Contributing

This repository is a reference knowledge base on ML/AI algorithms. If you're extending it:

1. **Follow the per-algorithm template** in `docs/00-overview/scope-and-methodology.md` (origin, mechanism, why it mattered, current status, strengths/limitations, usage, relationships, sources).
2. **Update the glossary.** Any new jargon term introduced should be added to `docs/00-overview/glossary.md`, alphabetized, with a one-line definition and a link back to the file that covers it in depth.
3. **Cite sources.** Attribute claims to papers (with year) or mark them as "widely reported." Avoid inventing precise benchmark numbers you can't source — use qualitative comparisons or ranges instead.
4. **Distinguish fact from projection.** In roadmap/outlook content, label speculative claims `[Projection]` and keep them visibly separate from sourced claims.
5. **Add at least one visual per file** — a Mermaid diagram or a substantial comparison table — using GitHub-flavored Markdown's native rendering.
6. **Cross-link.** Point to related/competing entries elsewhere in the repo rather than duplicating explanations.
7. **Keep terminology consistent** with the rest of the repo — check the glossary before introducing a synonym for an existing concept.
