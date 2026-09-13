# Transformer language model from scratch

A decoder-only transformer (GPT-style) language model implemented in **pure NumPy**. Every gradient
is derived by hand, written out, and verified against central finite differences. There is no
autodiff anywhere in the repository, and that is the point.

## Why write the backward pass by hand

Autodiff makes the chain rule invisible, and a wrong backward pass does not crash. It trains --
slightly worse, slightly slower, to a slightly worse optimum -- and presents itself as a
hyperparameter problem. Drop the two correction terms from the LayerNorm gradient and the model
still learns. Use fancy-index assignment instead of a scatter-add in the embedding backward and the
model still learns. Halve the tied embedding gradient by assigning instead of accumulating and the
model still learns.

None of those bugs is visible in a loss curve. All three are caught immediately by comparing the
analytic gradient against a central finite difference, which is why every layer here ships with a
gradient test rather than a plot.

```
python -m minigpt.cli gradcheck
```

## Layout

| Module | Contents |
| --- | --- |
| `minigpt/layers.py` | `Module` base, Linear, Embedding, LayerNorm, GELU, Dropout, fused cross-entropy |
| `minigpt/attention.py` | Causal multi-head self-attention; a KV-cached single-token step |
| `minigpt/model.py` | Pre-LN blocks, tied embeddings, the full model backward pass |
| `minigpt/optim.py` | AdamW with decoupled decay, global-norm clipping, cosine warmup |
| `minigpt/tokenizer.py` | Byte-level BPE (and a plain byte tokenizer as a control) |
| `minigpt/sampling.py` | Temperature, top-k, nucleus sampling, cached generation |
| `minigpt/train.py` | Batching, the training loop, and the diagnostics worth running first |
| `minigpt/gradcheck.py` | The numerical checks that make the derivations trustworthy |
| `minigpt/cli.py` | Eight demonstrations, each establishing one fact |

## The derivations

**LayerNorm.** With `xhat = (x - mu) * inv`, `inv = 1/sqrt(var + eps)`, and `dxhat = dout * gamma`,
the row gradient is

```
dx = inv/D * (D*dxhat - sum(dxhat) - xhat * sum(dxhat * xhat))
```

The two subtracted terms are the mean path and the variance path. Both `mu` and `var` depend on
every element of the row, so each output depends on every input; the popular shortcut `dx = dxhat *
inv` ignores that coupling. It is close enough to keep training and wrong enough to hurt.

**Softmax + cross-entropy, fused.** Computing `log(softmax(x))` in two steps loses precision and
overflows for large logits. Using the log-sum-exp identity directly,

```
loss = logsumexp(z) - z[target]
dz   = softmax(z) - onehot(target)
```

The gradient is that simple *only* because the two operations are fused; separately, the softmax
Jacobian is a full matrix per row. The implementation is stable at logits of +/-1e4, which is
asserted in the tests rather than hoped for.

**Attention.** For `out = P V` with `P = softmax(S)` and `S = Q K^T / sqrt(d)`, the backward pass
is `dP = dout V^T`, `dV = P^T dout`, then the softmax Jacobian per row
`dS = P * (dP - sum(dP * P))`, then `dQ = dS K / sqrt(d)` and `dK = dS^T Q / sqrt(d)`. The masked
entries of `dS` are forced back to zero, because a masked position must not receive gradient
through a path that does not exist in the forward pass.

**Weight tying.** The token embedding is used twice: as a lookup table on the way in, and
transposed as the output projection on the way out. It is one array, so it collects gradient from
both paths and the two must be **added**. This is exactly why `Module` accumulates into
`self.grads` instead of assigning. A dedicated test decomposes the total gradient, subtracts the
output-projection term `dlogits^T h`, and asserts that what remains is non-zero only on the rows
of tokens that actually appeared in the batch -- the signature of the scatter-add path.

## Causality is exact, not approximate

Masked scores are set to `-inf` before the softmax, so masked positions have probability exactly
zero rather than merely small. Changing the last token of a sequence therefore leaves the outputs at
every earlier position **bit-identical**, and the test asserts equality rather than a tolerance:

```
python -m minigpt.cli causal
```

This property is what makes training on all `T` positions at once legitimate. A leaky mask produces
a beautiful training loss and generation that falls apart, because at inference the future is
genuinely absent.

## The four diagnostics worth running before anything else

1. **Loss at initialisation is `ln(V)`.** A model that has learnt nothing is uniform over the
   vocabulary, and the cross-entropy of the uniform distribution is exactly `ln(V)` -- 5.545 for
   256 byte tokens. Far above means the initialisation is too large; far below means information is
   leaking, and the mask is the first suspect. One forward pass. (`cli init`)
2. **Overfit a single batch.** A correct model with a correct optimiser can drive the loss on two
   random eight-token sequences to near zero, because it has the capacity to memorise them. If it
   cannot, something is broken and no amount of data will fix it. This exercises attention, both
   norms, GELU, the tied embedding and AdamW at once. (`cli overfit`)
3. **Gradient check every parameter.** Central differences, `1e-5` step, relative error under
   `1e-6`. A relative error of exactly 1.0 means one of the two numbers is zero, which is almost
   always a gradient that never got accumulated. (`cli gradcheck`)
4. **Cached and uncached generation must agree.** Two independent code paths for one function.
   (`cli sample`)

## Optimiser

AdamW with **decoupled** weight decay: the decay is applied to the parameter directly, not folded
into the gradient. With zero gradients the update is exactly `p * (1 - lr*wd)^t`, which a test
pins. Adam with L2 in the gradient cannot satisfy that, because the decay term would pass through
the `1/sqrt(v)` normaliser and a weight's effective regularisation would depend on its own gradient
history. Biases and LayerNorm gains are excluded from decay by default -- shrinking a gain towards
zero attenuates the activations it exists to rescale.

Bias correction is what makes the first step `lr`-sized at any gradient scale: for a constant
gradient `mhat/sqrt(vhat)` is exactly +/-1. Without it, `v` starts at zero and is about a thousand
times too small on step one at `beta2 = 0.999`, making the first step roughly 32x too large.
Gradient clipping is global rather than per-tensor, so it changes the update's length but never its
direction, and the *pre-clip* norm is what gets logged -- logging the clipped norm just reports the
threshold back to you.

## Tokenizer

Byte-level BPE. The base vocabulary is the 256 byte values, so **every** string is representable and
there is no `<UNK>` token: emoji, Cyrillic, a Base64 blob and a corrupted download all round-trip
exactly. Two rules make it correct:

- **Merges are replayed in training rank order.** Each merge was learned on text already rewritten
  by all previous merges, so applying a later merge first produces a tokenisation the model was
  never trained on. The encoder repeatedly picks the lowest-ranked applicable pair.
- **Ties are broken lexicographically.** Otherwise two runs on the same corpus produce different
  vocabularies and a checkpoint silently stops matching its tokenizer.

Text is pre-tokenised into `optional whitespace + non-whitespace` chunks, GPT-2 style, and merges
never cross a chunk boundary -- which is why `"cat"` and `" cat"` are different tokens everywhere
in the real world.

## KV cache

During generation the keys and values of every previous token are reused, so a new token costs one
attention over the prefix instead of a full re-encode: `O(T)` per token rather than `O(T^2)` for
the sequence. The uncached path is kept as the reference implementation, and a test asserts the two
produce identical token sequences from the same seed.

## Install and run

```bash
pip install -r requirements.txt          # numpy, plus pytest for the suite

python -m minigpt.cli gradcheck          # analytic gradients vs finite differences
python -m minigpt.cli causal             # the mask, verified exactly
python -m minigpt.cli init               # loss at initialisation vs ln(V)
python -m minigpt.cli overfit            # memorise one batch
python -m minigpt.cli optim              # AdamW's exact properties
python -m minigpt.cli tokenizer          # byte-level BPE, compression and round-trips
python -m minigpt.cli sample             # decoding knobs and cache equivalence
python -m minigpt.cli train --steps 300  # a short training run, then generation

pytest                                   # the full suite
```

Everything is CPU-only, deterministic given a seed, and runs in seconds. The gradient-check model is
two layers, two heads, 16 channels -- 10,816 parameters, of which 4,096 are the embedding that also
serves as the output projection. Correctness of the chain rule does not depend on width, so checking
a large model would cost thousands of forward passes and prove nothing further.

## What the test suite pins

| File | Asserts |
| --- | --- |
| `tests/test_layers.py` | Every layer's parameter and input gradients; duplicate-index accumulation; LayerNorm's mean and variance paths; softmax and cross-entropy stability at extreme logits; `ln(V)` for uniform logits |
| `tests/test_attention.py` | Mask exactness (bit-identical earlier outputs); causal probability rows; the `1/sqrt(d_head)` scale keeping the softmax out of saturation; head split/merge round-trip; KV-cache equivalence |
| `tests/test_model.py` | Every model gradient end to end; the tied gradient decomposed into both paths; the parameter count arithmetic; gradient reaching block 0; the residual identity path |
| `tests/test_optim.py` | Exact geometric shrinkage under decoupled decay; scale invariance of the first step; direction-preserving global clipping; the warmup and cosine schedule |
| `tests/test_tokenizer.py` | Round-trips for unseen words, CJK, emoji and whitespace; deterministic training; rank-ordered merges; no token spanning a word boundary |
| `tests/test_sampling.py` | Filter semantics including the non-empty nucleus; sampled frequencies matching the distribution; cached vs uncached generation; eval mode during generation |
| `tests/test_train.py` | Contiguous split with no overlap; targets shifted by exactly one; `ln(V)` at step 0; the overfit-a-batch bound; schedule and clipping bookkeeping |
| `tests/test_cli.py` | Each demonstration runs and prints the evidence it claims |

## Honest limitations

- **NumPy on CPU.** Fine for the models here (tens of thousands of parameters, kilobytes of text);
  nowhere near a GPU framework for anything real. The demos are demonstrations that the gradients
  and the optimiser work, not a language model.
- **Attention is `O(T^2)` in memory and time**, with no flash-attention-style tiling.
- **Absolute learned positions only**, capped at `block_size`. No RoPE, ALiBi, or sliding window,
  so generation refuses to exceed the block rather than silently degrading.
- **The BPE trainer is `O(corpus)` per merge.** Correct, and appropriate for megabytes; a production
  implementation keeps incremental pair counts in a priority queue.
- **No model checkpointing.** The tokenizer serialises; the weights do not.
- **Dropout is the only regulariser**, and the demo configurations disable it so that finite
  differences are meaningful on a deterministic loss.

## References

- Vaswani et al., *Attention Is All You Need* (2017) -- the architecture, post-LN.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (2019) -- GPT-2: pre-LN,
  byte-level BPE, the `1/sqrt(2L)` residual initialisation.
- Xiong et al., *On Layer Normalization in the Transformer Architecture* (2020) -- why pre-LN trains
  without warmup as a crutch.
- Loshchilov and Hutter, *Decoupled Weight Decay Regularization* (2019) -- AdamW.
- Hendrycks and Gimpel, *Gaussian Error Linear Units* (2016) -- GELU, and the tanh approximation
  whose derivative must match the forward it is paired with.
- Holtzman et al., *The Curious Case of Neural Text Degeneration* (2020) -- nucleus sampling.

## License

MIT. See [LICENSE](LICENSE).
