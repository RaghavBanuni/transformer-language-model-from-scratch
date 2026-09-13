"""Causal multi-head self-attention, forward and backward, plus a KV-cached step.

Why the 1/sqrt(d_head) scale exists
-----------------------------------
If the components of ``q`` and ``k`` are roughly independent with unit variance, their dot product
over ``d_head`` dimensions has variance ``d_head`` -- so with a head size of 64 the logits arrive
with a standard deviation of 8. Feed that to a softmax and it saturates: one probability near 1,
the rest near 0, and a gradient of almost exactly zero everywhere. Dividing by ``sqrt(d_head)``
restores unit variance and keeps the softmax in the region where it has a usable derivative. The
scale is not a tuning constant; it is what makes deep attention trainable at all.

Why the mask is the whole game
------------------------------
A language model is trained on every position at once, which is only valid if position ``t``
cannot see position ``t+1``. Get the mask wrong and the model reads the answer: training loss
collapses, the numbers look wonderful, and generation is garbage because at inference the future
is not there. It is the most consequential single line in the file, and it is tested directly --
perturbing a *later* token must leave the logits at earlier positions bit-identical
(``tests/test_attention.py``), which is a sharp property no loss curve can express.

The mask is applied as ``-inf`` rather than a large negative number. With ``-inf`` the masked
softmax entries are exactly zero, so no gradient can leak backwards through them; with ``-1e9``
they are merely very small, and "very small" accumulates over layers and steps. Since the
row-wise max is always a finite unmasked entry (the diagonal), subtracting it keeps the
stable-softmax arithmetic free of ``inf - inf``.

The two forward paths
---------------------
:meth:`CausalSelfAttention.forward` processes a whole block and keeps what backward needs.
:meth:`CausalSelfAttention.forward_step` handles one new token against a cache of past keys and
values -- ``O(T)`` work per token instead of ``O(T^2)``, which is the difference between usable
and unusable generation. Two paths mean they can disagree, so a test asserts that cached
generation reproduces full recomputation to floating-point tolerance.
"""

from __future__ import annotations

import numpy as np

from .layers import Dropout, Linear, Module, softmax, softmax_backward


def causal_mask(size: int) -> np.ndarray:
    """``mask[i, j]`` is True where position ``i`` is allowed to attend to position ``j``."""
    return np.tril(np.ones((size, size), dtype=bool))


class CausalSelfAttention(Module):
    """Multi-head self-attention with a causal mask.

    Q, K and V come from one fused ``Linear(C, 3C)``. Fusing is not only faster: it also means
    there is a single weight-initialisation scale and a single backward path to get right, rather
    than three copies of the same code drifting apart.
    """

    def __init__(self, n_embd: int, n_head: int, rng: np.random.Generator,
                 dropout: float = 0.0, resid_scale: float | None = None) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd={n_embd} must be divisible by n_head={n_head}")
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.qkv = self.add_child("qkv", Linear(n_embd, 3 * n_embd, rng))
        self.proj = self.add_child("proj", Linear(n_embd, n_embd, rng, scale=resid_scale))
        self.attn_dropout = self.add_child("attn_dropout", Dropout(dropout, rng))
        self.resid_dropout = self.add_child("resid_dropout", Dropout(dropout, rng))

    # -- shape helpers -------------------------------------------------------------------

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """``(B, T, C) -> (B, H, T, head_dim)``."""
        batch, time, _ = x.shape
        return x.reshape(batch, time, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """``(B, H, T, head_dim) -> (B, T, C)``, the exact inverse of :meth:`_split_heads`."""
        batch, _, time, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, time, self.n_embd)

    # -- training path -------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch, time, channels = x.shape
        if channels != self.n_embd:
            raise ValueError(f"expected last dimension {self.n_embd}, got {channels}")

        qkv = self.qkv.forward(x)
        query, key, value = np.split(qkv, 3, axis=-1)
        q = self._split_heads(query)
        k = self._split_heads(key)
        v = self._split_heads(value)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        mask = causal_mask(time)
        scores = np.where(mask, scores, -np.inf)

        probabilities = softmax(scores, axis=-1)
        dropped = self.attn_dropout.forward(probabilities)
        context = dropped @ v

        merged = self._merge_heads(context)
        out = self.resid_dropout.forward(self.proj.forward(merged))

        self._cache = (q, k, v, probabilities, dropped, mask)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Reverse the chain: projection, context product, softmax, scores, then Q/K/V."""
        q, k, v, probabilities, dropped, mask = self._cache

        d_merged = self.proj.backward(self.resid_dropout.backward(dout))
        d_context = self._split_heads(d_merged)

        # context = dropped @ v
        d_dropped = d_context @ v.transpose(0, 1, 3, 2)
        d_v = dropped.transpose(0, 1, 3, 2) @ d_context

        d_probabilities = self.attn_dropout.backward(d_dropped)
        d_scores = softmax_backward(probabilities, d_probabilities)
        # Masked entries have probability exactly zero, so their gradient is already zero; being
        # explicit costs nothing and documents the invariant.
        d_scores = np.where(mask, d_scores, 0.0)

        d_q = (d_scores @ k) * self.scale
        d_k = (d_scores.transpose(0, 1, 3, 2) @ q) * self.scale

        d_qkv = np.concatenate(
            (self._merge_heads(d_q), self._merge_heads(d_k), self._merge_heads(d_v)), axis=-1
        )
        return self.qkv.backward(d_qkv)

    # -- generation path -----------------------------------------------------------------

    def forward_step(
        self,
        x: np.ndarray,
        cache: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Attend one new token over a cache of previous keys and values.

        ``x`` is ``(B, 1, C)``; the returned cache is ``(B, H, T_seen, head_dim)`` per tensor.
        No mask is needed here -- everything in the cache is by construction in the past, which is
        the same causality expressed by construction rather than by masking. Nothing is stored for
        backward: this path is inference only.
        """
        if x.shape[1] != 1:
            raise ValueError("forward_step expects exactly one time step")

        qkv = self.qkv.forward(x)
        query, key, value = np.split(qkv, 3, axis=-1)
        q = self._split_heads(query)
        k = self._split_heads(key)
        v = self._split_heads(value)

        if cache is not None:
            past_k, past_v = cache
            k = np.concatenate((past_k, k), axis=2)
            v = np.concatenate((past_v, v), axis=2)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        context = softmax(scores, axis=-1) @ v
        out = self.proj.forward(self._merge_heads(context))
        return out, (k, v)
