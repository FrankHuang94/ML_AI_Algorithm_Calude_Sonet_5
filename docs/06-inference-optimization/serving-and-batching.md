# Serving and Batching

This file covers how requests to a deployed LLM are actually scheduled and processed efficiently in production — the systems-level layer that sits on top of everything else in this section (quantization, KV cache management, speculative decoding) and determines how those techniques translate into real-world throughput and latency.

## The throughput vs. latency tradeoff, framed

**Throughput** is how many requests (or tokens) a serving system can process per unit of time, in aggregate, across all users. **Latency** is how long any single request takes to get its response, from the perspective of the user waiting for it. These two goals are frequently in tension: batching many requests together (processing them simultaneously) tends to improve throughput (better utilization of the accelerator's parallel compute capacity — running one request at a time on a GPU built for massive parallelism leaves most of its capacity idle) but can increase individual latency (a request might have to wait for a batch to fill up, or share compute with other concurrent requests, before finishing). Most of the design choices in this file are, at bottom, different points on this same tradeoff curve.

## Static batching

**Mechanism.** Collect a fixed group of requests, process them together as one batch from start to finish, and only start a new batch once the entire previous batch has completely finished (every request in it has generated its full response).

**Limitation.** Because generated response lengths vary a lot (some responses are one sentence, others are pages long), a static batch is only as fast as its *slowest* member — shorter requests in the batch finish early but their allocated compute slot sits idle, waiting for the longest request in the batch to complete, before the system can start serving new requests in that slot. This wastes a substantial amount of potential throughput, especially as response-length variance grows.

## Continuous (dynamic) batching

**Mechanism.** Rather than waiting for an entire batch to finish before starting new work, continuously monitor the batch and, as soon as any individual request in it finishes generating its response, immediately swap in a new waiting request to take that freed-up slot — the batch's composition changes dynamically, request by request, rather than being fixed for its entire duration.

**Why it mattered.** This directly fixes static batching's central inefficiency: no accelerator capacity sits idle waiting for the single longest request in a batch to finish, since finished requests are replaced immediately rather than at the end of a fixed batch cycle. Continuous batching (sometimes called dynamic batching) has been widely reported to substantially improve achievable serving throughput relative to static batching, and is a standard, expected feature of modern high-throughput LLM serving systems as of 2026.

**How this interacts with the KV cache.** Continuously swapping requests in and out of a batch means the serving system needs to efficiently allocate and free KV cache memory (see [kv-cache-and-attention-optimization.md](kv-cache-and-attention-optimization.md)) for individual requests on the fly, without needing to reshuffle or defragment the memory used by every other request still in progress — which is exactly the problem PagedAttention's block/page-table design (covered in that file) was built to solve efficiently. Continuous batching and PagedAttention-style memory management are, in practice, complementary techniques that are typically deployed together, not separately.

## Reference systems: vLLM and TensorRT-LLM

**vLLM.** An open-source LLM serving system (the system that introduced PagedAttention — see [kv-cache-and-attention-optimization.md](kv-cache-and-attention-optimization.md)) built specifically around continuous batching and efficient, paged KV cache memory management, widely adopted across the open-source and research community as a high-throughput serving solution.

**TensorRT-LLM (NVIDIA).** A serving/inference-optimization toolkit that applies hardware-specific compilation and kernel optimization (tailored specifically to NVIDIA GPU architectures) alongside techniques like continuous batching and quantization support, aimed at maximizing throughput and minimizing latency specifically on NVIDIA hardware.

**Why these are worth naming.** Both systems are widely-used, concrete, real-world examples that combine essentially every technique covered across this section — quantization (see [quantization.md](quantization.md)), efficient KV cache management, continuous batching, and (in various integrations) speculative decoding (see [speculative-decoding.md](speculative-decoding.md)) — into a single deployable serving stack, illustrating that these techniques are used in combination in practice, not as isolated academic curiosities.

## How KV cache management interacts with batching decisions

Tying together this section's core throughline: the KV cache's memory footprint (see [kv-cache-and-attention-optimization.md](kv-cache-and-attention-optimization.md)) directly limits how many concurrent requests can be batched together within a fixed memory budget — a larger per-request KV cache (from longer contexts, or from an architecture without MQA/GQA-style cache-size reduction) means fewer requests can be batched simultaneously, which directly reduces achievable throughput. This is why GQA-style architectural choices (decided at pretraining time), quantization (applied to the model and sometimes the KV cache itself), and PagedAttention-style memory management (applied at serving time) all ultimately serve the same downstream goal from different angles: fitting more useful concurrent work into a fixed amount of expensive accelerator memory, which is the central resource constraint that essentially every technique in this section of the repository is, in one way or another, trying to relax.

## Comparison table

| Approach | Throughput | Latency for short requests | Handles variable response length well? |
|---|---|---|---|
| Static batching | Lower (idle slots while waiting for longest request) | Can be poor (short requests wait for whole batch) | No |
| Continuous (dynamic) batching | Higher (slots refilled immediately) | Better (finished requests exit immediately) | Yes |

## Relationship to other algorithms

- Continuous batching's efficiency gains are directly enabled by the KV cache memory management covered in [kv-cache-and-attention-optimization.md](kv-cache-and-attention-optimization.md).
- Quantization (see [quantization.md](quantization.md)) reduces per-request memory footprint, directly increasing the number of requests that can be batched within a fixed memory budget.
- Speculative decoding (see [speculative-decoding.md](speculative-decoding.md)) and batching are complementary throughput levers, both commonly deployed together in production serving stacks like vLLM and TensorRT-LLM.

## Sources

- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (2022) — widely credited with introducing continuous/iteration-level batching for LLM serving
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) [vLLM]
