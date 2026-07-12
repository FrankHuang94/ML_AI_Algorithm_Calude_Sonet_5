# RNNs, LSTMs, and GRUs

Recurrent architectures were, for roughly two decades, the default way to process sequences (text, time series, audio) with neural networks. They are largely superseded by the Transformer (see [transformer-architecture.md](transformer-architecture.md)) for most large-scale applications today, but understanding them — and specifically the vanishing gradient problem they struggle with, and the pre-Transformer attention mechanism they gave rise to — is essential context for why the Transformer's design choices make sense.

## Vanilla RNN mechanics

**Name & definition.** A Recurrent Neural Network processes a sequence one element at a time, maintaining a **hidden state** (a vector summarizing everything relevant seen so far) that gets updated at each step and fed forward into the next.

**Origin.** Recurrent network architectures date to the 1980s (Rumelhart, Hinton, Williams and others); the specific "simple RNN" formulation used as a baseline today is often attributed to Elman (1990).

**Core mechanism.**

```
hₜ = tanh(W_h · h_{t-1} + W_x · xₜ + b)
yₜ = W_y · hₜ + b_y
```

Walkthrough: at each time step t, the network combines the previous hidden state h_{t-1} with the current input xₜ (each via its own learned weight matrix), passes the result through a nonlinearity (tanh, a squashing function similar in spirit to sigmoid but ranging from −1 to 1), and produces a new hidden state hₜ. This same set of weights (W_h, W_x, W_y) is reused at *every* time step — a form of weight sharing across time, analogous to how a CNN's filter is shared across spatial positions (see [cnn-family.md](cnn-family.md)) — which lets an RNN process sequences of any length with a fixed number of parameters. yₜ, an optional per-step output, is computed from the hidden state, and can be produced at every step (e.g., for tagging each word) or only at the final step (e.g., for classifying an entire sequence).

**Why it mattered.** The recurrent formulation gave neural networks a natural way to handle variable-length sequential data without needing a fixed input size, using the hidden state as a compressed "memory" of everything processed so far.

## The vanishing/exploding gradient problem, concretely

**The problem.** Training an RNN uses **backpropagation through time (BPTT)** — you unroll the recurrence across all T time steps (treating it as a very deep feedforward network with T layers, all sharing the same weights) and backpropagate the loss gradient through that unrolled chain. Because the same weight matrix W_h is applied repeatedly, the gradient flowing back to an early time step gets multiplied by (approximately) the same factor T times in a row — once per time step it passes through.

Walkthrough with a concrete illustration: if that per-step multiplicative factor is, say, 0.9, then after 100 time steps the gradient has been scaled by roughly 0.9^100 ≈ 0.00003 — it has vanished, and the network effectively cannot learn dependencies more than a few dozen steps back, because the training signal describing "the output 80 steps ago depended on this early input" never survives the trip backward with meaningful magnitude. Conversely, if the per-step factor is something like 1.1, the gradient explodes: 1.1^100 ≈ 13,781 — an enormous, destabilizing value (this is the same exploding-gradient phenomenon covered generally in [optimization-algorithms.md](../01-foundations/optimization-algorithms.md), but RNNs are its original and most acute setting, since the same weight matrix is reapplied at every single time step rather than passing through many distinct layers with independently-initialized weights). Real RNNs sit somewhere between pure vanishing and pure exploding depending on the eigenvalues of W_h, but in practice, vanilla RNNs reliably struggle to learn dependencies spanning more than roughly 10-20 time steps, which is a severe limitation for language (where a sentence's meaning can hinge on a word many tokens earlier) and other long sequences.

**Why this mattered so much.** This concrete failure mode — long-range dependencies simply don't survive training — is the direct motivation for both LSTMs and GRUs below, and, later, for abandoning recurrence altogether in favor of the Transformer's attention mechanism, which gives every position a direct, single-step path to every other position (see [transformer-architecture.md](transformer-architecture.md)) instead of a gradient path that grows with sequence length.

## LSTM (Long Short-Term Memory)

**Name & definition.** An LSTM is a recurrent unit augmented with an explicit **cell state** (a separate memory channel that can carry information across many time steps largely unchanged) and a set of learned **gates** that control what information is added to, removed from, and read out of that memory.

**Origin.** Hochreiter, Schmidhuber, "Long Short-Term Memory" (1997) — notably, this predates the deep learning boom by 15 years, and the vanishing gradient problem it addresses was identified by the same authors in earlier work (Hochreiter's 1991 diploma thesis).

**Core mechanism — the gates, one at a time.** An LSTM maintains both a hidden state hₜ (as in a vanilla RNN) and a separate cell state cₜ. At each time step, three gates (each a small neural network layer outputting values between 0 and 1 via a sigmoid, acting as a "how much to let through" dial) control the flow of information:

- **Forget gate** fₜ = σ(W_f·[h_{t-1}, xₜ] + b_f) — decides how much of the existing cell state to keep vs. discard. A value near 0 for a given memory dimension means "forget this"; near 1 means "keep it."
- **Input gate** iₜ = σ(W_i·[h_{t-1}, xₜ] + b_i) — decides how much of a candidate new value (c̃ₜ = tanh(W_c·[h_{t-1}, xₜ] + b_c)) to write into the cell state.
- **Output gate** oₜ = σ(W_o·[h_{t-1}, xₜ] + b_o) — decides how much of the (updated) cell state to expose as the new hidden state.

The cell state update is:

```
cₜ = fₜ ⊙ c_{t-1} + iₜ ⊙ c̃ₜ         (⊙ = element-wise multiplication)
hₜ = oₜ ⊙ tanh(cₜ)
```

The clearest way to see why this design fixes the vanishing gradient is to picture the **cell state as a conveyor belt** running straight across time, with gates that add to or erase from it but never repeatedly multiply it by a weight matrix:

```
             forget gate         input gate
             (erase some?)       (write some?)
                  │                   │
   c_{t-1} ──────►⊗────────────────►(+)──────────► cₜ   ← the "conveyor belt":
   (old memory)   ▲                   ▲              (mostly just carries memory
                  │                   │               forward, lightly edited)
                 fₜ         iₜ ⊙ c̃ₜ (candidate new info)
                                                     │
                                          output gate⊗──► hₜ (what to expose now)
                                                     ▲
                                                    oₜ
```

Walkthrough: the forget and input gates together decide, per memory dimension, "keep the old value, mostly replace it with new information, or some blend of both" — and critically, the cell-state update is dominated by an additive term (fₜ ⊙ c_{t-1} plus iₜ ⊙ c̃ₜ) rather than the repeated matrix multiplication that plagues the vanilla RNN's hidden state. When the forget gate is close to 1, information can flow across many time steps through the cₜ pathway with little decay, because there's no repeated multiplication by a weight matrix along that path — just an approximately-preserving addition at each step. This is the specific mechanical fix for the vanishing gradient problem: gradients can flow back through the cell-state path largely undiminished, as long as the forget gate stays open. (Notice the conceptual kinship with the residual/skip connection in [cnn-family.md](cnn-family.md) and the Transformer: in all three, the trick for training depth is an *additive* path that gradients can travel along without being repeatedly scaled down. The LSTM was doing this for depth-in-time two decades before residual connections did it for depth-in-layers.)

**Why it mattered.** LSTMs could reliably learn dependencies spanning hundreds of time steps, dramatically outperforming vanilla RNNs on language modeling, translation, and speech tasks, and became the default recurrent architecture for essentially all sequence modeling work from the late 2000s (once compute caught up to make deep LSTM training practical) through the mid-2010s.

## GRU (Gated Recurrent Unit)

**Name & definition.** The GRU is a simplified alternative to the LSTM that merges the cell state and hidden state into one, and reduces three gates to two.

**Origin.** Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (2014).

**Core mechanism (brief).** A GRU has an **update gate** (which blends the previous hidden state with a candidate new hidden state — effectively combining the roles of the LSTM's forget and input gates into one) and a **reset gate** (which controls how much of the previous hidden state is used when computing the candidate new state). With fewer gates and no separate cell state, a GRU has fewer parameters than an LSTM of the same hidden size.

**Why it mattered / current status.** GRUs typically perform comparably to LSTMs on many tasks while being cheaper to train (fewer parameters, simpler computation graph), and became a popular default when a lighter-weight recurrent unit was preferred. Like LSTMs, GRUs have been largely displaced by Transformer-based architectures for large-scale sequence modeling as of 2026, though both remain reasonable, lower-compute choices for smaller-scale sequence tasks (e.g., certain time-series forecasting or embedded/on-device applications where a full Transformer is overkill).

## Seq2seq architectures

**Name & definition.** Sequence-to-sequence (seq2seq) models map an input sequence to an output sequence of a potentially different length, using an **encoder** RNN/LSTM to process the entire input into a fixed-size summary vector, and a **decoder** RNN/LSTM that generates the output sequence one element at a time, conditioned on that summary.

**Origin.** Sutskever, Vinyals, Le, "Sequence to Sequence Learning with Neural Networks" (2014); Cho et al. (2014, same paper as GRU above) introduced a closely related encoder-decoder framing simultaneously.

**Why it mattered.** Seq2seq gave a clean, general recipe for tasks like machine translation, summarization, and question answering — any task that maps one sequence to another — without requiring a task-specific architecture, and was the dominant paradigm for neural machine translation before the Transformer.

**The bottleneck problem.** The original seq2seq design compresses the *entire* input sequence into a single fixed-size vector (the encoder's final hidden state) that the decoder must rely on for everything it generates. For long inputs, this is a severe information bottleneck — a single vector, no matter how large, struggles to retain all the relevant detail from a long sentence or document. This limitation is exactly what pre-Transformer attention (below) was invented to fix.

## Bahdanau/Luong attention — the direct bridge to Transformers

**Name & definition.** Attention, in this original context, lets the decoder look back at *all* of the encoder's hidden states (one per input position) at every decoding step, dynamically computing a weighted combination of them, rather than relying solely on a single fixed summary vector.

**Origin.** Bahdanau, Cho, Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate" (2014); Luong, Pham, Manning, "Effective Approaches to Attention-based Neural Machine Translation" (2015), which proposed simplified, computationally cheaper variants (this is why you'll see both "Bahdanau attention" and "Luong attention" referenced as related-but-distinct historical mechanisms).

**Core mechanism.** At each decoder step, compute a similarity/relevance score between the decoder's current hidden state and *every* encoder hidden state; convert those scores into weights via softmax (see [loss-functions.md](../01-foundations/loss-functions.md)); take the weighted sum of encoder hidden states using those weights (a **context vector**); use that context vector, alongside the decoder's own hidden state, to help produce the next output.

Walkthrough: instead of forcing the whole input sentence through one bottleneck vector, the decoder gets to "look back" at the specific parts of the input most relevant to whatever it's currently trying to generate — e.g., when translating a sentence, the decoder can attend most strongly to the source word or phrase corresponding to the word it's about to output, similar to how a human translator's eyes might dart back to a particular part of the source sentence.

**Why it mattered — the direct line to Transformers.** This mechanism eliminated the seq2seq bottleneck and produced a substantial jump in translation quality. It is also, historically, the single most direct conceptual predecessor of the self-attention mechanism at the heart of the Transformer (see [transformer-architecture.md](transformer-architecture.md)): the core idea — compute relevance scores between a query and a set of candidates, softmax them into weights, and take a weighted sum — is exactly the same operation. The Transformer's key innovation was to realize that this same attention operation, applied not just from decoder to encoder but from *every* position to *every other* position (including within a single sequence, "self"-attention), could fully replace recurrence altogether — removing the sequential, one-step-at-a-time processing bottleneck that limits how well RNNs parallelize on modern hardware, and sidestepping the vanishing-gradient-over-long-sequences problem entirely, since attention gives every pair of positions a direct, single-step computational path regardless of how far apart they are in the sequence.

## Comparison table

| Architecture | Year | Key Innovation | Still Relevant (2026)? |
|---|---|---|---|
| Vanilla RNN | 1980s / 1990 | Recurrent hidden state, shared weights across time | Rare — vanishing gradients limit usefulness |
| LSTM | 1997 | Gated cell state fixes vanishing gradients over long sequences | Niche — smaller-scale/on-device sequence tasks |
| GRU | 2014 | Simplified gating, fewer parameters than LSTM | Niche — similar role to LSTM, lighter-weight |
| Seq2seq (encoder-decoder) | 2014 | General sequence-to-sequence framing | Superseded by Transformer encoder-decoder/decoder-only designs |
| Bahdanau/Luong attention | 2014-2015 | Dynamic, position-wise relevance weighting | Conceptually foundational — direct ancestor of self-attention |

## Relationship to other algorithms

- The vanishing/exploding gradient problem introduced here in its most acute form is covered generally (with gradient clipping as the standard mitigation) in [optimization-algorithms.md](../01-foundations/optimization-algorithms.md).
- Bahdanau/Luong attention is the direct conceptual predecessor of self-attention in [transformer-architecture.md](transformer-architecture.md) — read that file's opening section immediately after this one for the fullest sense of the lineage.
- Seq2seq's encoder-decoder framing reappears, transformed, as the encoder-decoder Transformer variant discussed in [transformer-architecture.md](transformer-architecture.md).

## Sources

- Rumelhart, Hinton, Williams, "Learning representations by back-propagating errors" (1986)
- Elman, "Finding Structure in Time" (1990)
- Hochreiter, "Untersuchungen zu dynamischen neuronalen Netzen" (diploma thesis, 1991) [early vanishing gradient analysis]
- Hochreiter, Schmidhuber, "Long Short-Term Memory" (1997)
- Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (2014) [GRU]
- Sutskever, Vinyals, Le, "Sequence to Sequence Learning with Neural Networks" (2014)
- Bahdanau, Cho, Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate" (2014)
- Luong, Pham, Manning, "Effective Approaches to Attention-based Neural Machine Translation" (2015)
