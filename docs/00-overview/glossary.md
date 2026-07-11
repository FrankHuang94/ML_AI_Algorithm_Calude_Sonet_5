# Glossary

A living, alphabetized list of every jargon term used in this repository. Each entry gets a one-line definition and a link to the file where it's covered in depth (if any single file "owns" it). This file is appended to throughout the build — if you find a term used in a doc file that isn't listed here, that's a bug; please add it.

## How to use this

Files in this repository define jargon inline on first use with a short parenthetical, per the audience calibration in [scope-and-methodology.md](scope-and-methodology.md). This glossary exists so you don't have to hunt through files to re-find a definition, and so terminology stays consistent repo-wide (the same concept should never have two different names in two different files).

---

## A

- **Ablation** — an experiment where you remove or disable one component of a system (a layer, a loss term, a training trick) and re-measure performance, to find out how much that component actually contributed. If removing it doesn't hurt performance, it wasn't pulling its weight. Covered in: [scope-and-methodology.md](scope-and-methodology.md).

## B

- **Benchmark** — a standardized dataset/task pair (plus a scoring rule) used to compare models against each other under identical conditions. "SOTA on benchmark X" means "best published score on that specific standardized test," which is narrower than "best model overall." Covered in: [scope-and-methodology.md](scope-and-methodology.md).

## F

- **Few-shot (learning/prompting)** — giving a model a handful (typically 1-100) of example input/output pairs at inference time (in the prompt, not via weight updates) and expecting it to generalize the pattern to a new input. Contrast with zero-shot. Covered in: [scope-and-methodology.md](scope-and-methodology.md).

## S

- **SOTA (State of the Art)** — the best publicly reported result on a given benchmark at a given point in time. A moving target, not a fixed method — "SOTA" describes a leaderboard position, not any one algorithm. Covered in: [scope-and-methodology.md](scope-and-methodology.md).

## Z

- **Zero-shot (learning/prompting)** — asking a model to perform a task it was never given explicit examples of at inference time, relying entirely on what it learned during training/pretraining. Contrast with few-shot. Covered in: [scope-and-methodology.md](scope-and-methodology.md).

---

*(Additional terms are appended here as each subsequent file in the build is written.)*
