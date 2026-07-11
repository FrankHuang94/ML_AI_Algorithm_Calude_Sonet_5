# Scope and Methodology

## What this repository is

This is a reference knowledge base covering the algorithms and techniques that matter in machine learning (ML) and artificial intelligence (AI) — both the classical statistical methods that predate the deep learning era and the neural architectures and training techniques that define it today. The repository has four jobs, one per major section grouping:

1. **What exists** — a catalog of every algorithm/technique of real importance in current ML/AI training and inference, not just deep learning.
2. **How we got here** — the development lineage of each major algorithm family: who introduced it, what problem it solved, and what it replaced.
3. **Where things stand now** — current state-of-the-art usage as of mid-2026, and which algorithms actually dominate in which regimes.
4. **Where things are headed** — near-term (6-18 month) and longer-term (2-5 year) trajectories, explicitly labeled as informed speculation rather than fact.

This is a single, coherent reference — not a swarm of disconnected notes. Every file follows the same structural conventions (below) so that the repository reads consistently whether you start at the README or drop into a single file from a search result.

## Audience calibration

This repository is written for a reader with **a bachelor's degree in computer science**: solid programming ability, standard CS fundamentals (data structures, algorithms, complexity/Big-O), and undergraduate-level linear algebra, calculus, and probability/statistics. This reader has **not** taken a graduate-level ML course.

Concretely, every file in this repository assumes you already know — without re-explanation:

- Vectors, matrices, matrix multiplication, dot products, and eigenvalues at a conceptual level.
- Derivatives, partial derivatives, and the chain rule.
- Basic probability: distributions, expectation, variance, conditional probability, Bayes' rule.
- Big-O notation, recursion, graphs, trees.
- General software engineering concepts.

And every file **defines on first use**, with a short inline gloss (not a lecture):

- ML-specific jargon (e.g., "loss landscape," "embedding," "latent space," "logits," "overfitting").
- Any notation beyond basic algebra/calculus (e.g., expectation notation 𝔼, summation over a dataset, softmax).
- Graduate-level math shortcuts — nothing is assumed from measure theory, deep information theory (beyond entropy/KL divergence, which is itself explained once and then reused), convex optimization theory, or advanced statistics. If a file needs to reference something like a "variational lower bound," it says what that means in a sentence, not just the name.
- Research-paper conventions like "ablation," "benchmark," "SOTA" (state of the art), "zero-shot," and "few-shot" — these are defined once, in the glossary below, and linked back to rather than re-defined in every file.

**Tone target:** the way a strong senior engineer would explain these topics to a smart colleague who's good at CS but skipped grad-school ML — technically precise, no hand-waving, but zero unexplained jargon. Equations are used where they clarify a mechanism, and every equation is followed by a plain-English walkthrough of what it's doing and why. An equation with no walkthrough is treated as a bug in the file.

If you hit a term that isn't explained where it's used, check **[the glossary](glossary.md)** first — it's the single running list of every jargon term used anywhere in the repository, alphabetized, each with a one-line definition and a link to the file that covers it in depth. The glossary is a living file, appended to throughout the build as new terms are introduced.

## The per-algorithm template

Every algorithm or technique entry in this repository — whether it's an entire file or a subsection within one — follows the same nine-part structure:

1. **Name & one-line definition**
2. **Origin** — year introduced, original paper/authors, and the problem it was designed to solve
3. **Core mechanism** — plain-language explanation first, then the key equation or pseudocode, then a plain-English walkthrough of that equation
4. **Why it mattered** — what it improved on or replaced, and what specifically was broken or limited before it
5. **Current status** — still dominant / niche / superseded / experimental, as of mid-2026
6. **Strengths & limitations** — an honest bullet list of tradeoffs, no marketing language
7. **Where it's used today** — concrete examples: model families, production systems, problem domains
8. **Relationship to other algorithms** — explicit links to related or competing entries elsewhere in the repository
9. **Sources** — papers with year; anything uncertain is flagged as "reported" or "widely reported" rather than stated as settled fact

Minor or closely related techniques get a lighter version of this template rather than the full nine-part treatment — enough to place them accurately without padding.

| # | Section | What it captures |
|---|---|---|
| 1 | Name & definition | What the thing is, in one line |
| 2 | Origin | Year, authors/paper, problem it solved |
| 3 | Core mechanism | Plain-language → equation/pseudocode → plain-English walkthrough |
| 4 | Why it mattered | What it replaced or improved on |
| 5 | Current status | Dominant / niche / superseded / experimental (mid-2026) |
| 6 | Strengths & limitations | Honest tradeoffs, no marketing language |
| 7 | Where it's used today | Concrete model families, systems, domains |
| 8 | Relationship to other algorithms | Explicit cross-links |
| 9 | Sources | Papers with year; "reported" flag where uncertain |

## How history and roadmap sections handle uncertainty

The [08-history](../08-history/) files are chronological and cite actual publication years and paper names. Where a date is disputed or the author isn't confident of exact precision, the text says so explicitly rather than inventing false precision.

The [09-roadmaps](../09-roadmaps/) files separate two categories of claim, and the separation is visible in the text itself:

- **Sourced claims** — documented lab/company public statements or published research directions, cited as such.
- **`[Projection]`** — the author's own reasoned extrapolation from public trends. This label is not decoration; it's a standing warning that the claim is informed speculation, not fact. If you see `[Projection]`, treat everything in that sentence or bullet as "our best guess," not "what will happen."

## Quality bar

- **Accuracy over comprehensiveness** where the two conflict. No padding to hit a word count.
- Every factual or historical claim is attributable — to a paper, a year, or explicitly marked "widely reported." Benchmark numbers the author isn't confident of are given as ranges or qualitative comparisons, not invented precision.
- Terminology is consistent across files. If two files seem to use different names for the same concept, that's a bug — check the glossary, and file it as an inconsistency to fix.

## Repository map

See **[taxonomy-map.md](taxonomy-map.md)** for a visual overview of how every algorithm family in this repository relates to every other one, and the [README](/README.md) for the full linked table of contents.
