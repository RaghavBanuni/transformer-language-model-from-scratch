"""The decoder-only transformer: blocks, weight tying, and the full backward pass.

Pre-normalisation, and why it is not a style choice
--------------------------------------------------
The original transformer put layer norm *after* the residual add::

    x = LayerNorm(x + Attention(x))          # post-LN, Vaswani et al. 2017

Everything since puts it before the sub-layer, inside the branch::

    x = x + Attention(LayerNorm(x))          # pre-LN, used here

The difference is what the residual path looks like to the gradient. With pre-LN the skip
connection is a clean identity from the loss all the way to the embedding, so the gradient reaches
layer 0 undistorted. With post-LN every skip passes through a normalisation whose Jacobian
rescales it, and the product of those rescalings across depth is what makes post-LN models need
learning-rate warmup to train at all (Xiong et al., 2020). Pre-LN trains without that crutch.

Weight tying, and the gradient bug it invites
---------------------------------------------
The token embedding matrix is used twice: once as a lookup table on the way in, and once --
transposed -- as the output projection on the way out. It is the same array, so it collects
gradient from both paths, and the two must be *added*.

This is exactly why :class:`minigpt.layers.Module` accumulates into ``self.grads`` instead of
assigning. An implementation that assigns keeps whichever path ran last, halving the embedding
gradient, and nothing in the loss curve says so -- the model still trains, just worse. The
gradient check on ``tok_emb.W`` is what catches it, and there is a dedicated test that the tied
gradient equals the sum of the two paths.

Tying also removes ``vocab_size * n_embd`` parameters, which for a small model is most of them,
and it encodes a real prior: a token's input representation and the direction that predicts it
ought to be related.

Residual projection scaling
---------------------------
Every block adds to the residual stream, so after ``L`` blocks the stream has accumulated ``2L``
branch outputs and its variance grows with depth. Scaling the initialisation of the projections
that *write* to the stream by ``1/sqrt(2L)`` keeps activation variance roughly constant with
depth, which is the GPT-2 initialisation trick and the reason a deep stack does not need its
first few hundred steps to undo its own initialisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attention import CausalSelfAttention
from .layers import GELU, Dropout, LayerNorm, Linear, Module, cross_entropy


@dataclass(frozen=True)
class ModelConfig:
    """Model shape. Deliberately small defaults: this runs on a laptop CPU in NumPy."""

    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class MLP(Module):
    """Position-wise feed-forward: expand by 4, GELU, project back.

    The 4x expansion is convention rather than derivation, but the shape matters: this is where
    most of the parameters and most of the per-token computation live, while attention supplies
    the mixing across positions. Widening here is the cheapest way to add capacity.
    """

    def __init__(self, n_embd: int, rng: np.random.Generator, dropout: float = 0.0,
                 resid_scale: float | None = None) -> None:
        super().__init__()
        hidden = 4 * n_embd
        self.fc = self.add_child("fc", Linear(n_embd, hidden, rng))
        self.act = self.add_child("act", GELU())
        self.proj = self.add_child("proj", Linear(hidden, n_embd, rng, scale=resid_scale))
        self.dropout = self.add_child("dropout", Dropout(dropout, rng))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.dropout.forward(self.proj.forward(self.act.forward(self.fc.forward(x))))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return self.fc.backward(self.act.backward(self.proj.backward(self.dropout.backward(dout))))


class Block(Module):
    """One transformer block: pre-LN attention, then pre-LN MLP, both residual.

    The backward pass of ``x = x + f(norm(x))`` is ``dx = dout + norm'(f'(dout))``. That first
    term is the identity path, and forgetting to add it is a classic error: the model still runs,
    gradients still flow through the branch, and the deepest layers simply stop learning.
    """

    def __init__(self, config: ModelConfig, rng: np.random.Generator) -> None:
        super().__init__()
        resid_scale = 0.02 / np.sqrt(2 * config.n_layer)
        self.ln1 = self.add_child("ln1", LayerNorm(config.n_embd))
        self.attn = self.add_child(
            "attn",
            CausalSelfAttention(
                config.n_embd, config.n_head, rng, config.dropout, resid_scale=resid_scale
            ),
        )
        self.ln2 = self.add_child("ln2", LayerNorm(config.n_embd))
        self.mlp = self.add_child(
            "mlp", MLP(config.n_embd, rng, config.dropout, resid_scale=resid_scale)
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x + self.attn.forward(self.ln1.forward(x))
        x = x + self.mlp.forward(self.ln2.forward(x))
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        d = dout + self.ln2.backward(self.mlp.backward(dout))
        d = d + self.ln1.backward(self.attn.backward(d))
        return d

    def forward_step(self, x: np.ndarray, cache=None):
        attn_out, new_cache = self.attn.forward_step(self.ln1.forward(x), cache)
        x = x + attn_out
        x = x + self.mlp.forward(self.ln2.forward(x))
        return x, new_cache


class TransformerLM(Module):
    """A decoder-only language model with tied input and output embeddings."""

    def __init__(self, config: ModelConfig, seed: int = 0) -> None:
        super().__init__()
        self.config = config
        rng = np.random.default_rng(seed)

        from .layers import Embedding  # local import keeps the public surface of layers tidy

        self.tok_emb = self.add_child("tok_emb", Embedding(config.vocab_size, config.n_embd, rng))
        # Learned absolute positions. Registered directly on the model because there is no
        # lookup-table backward to do: every position in the block is used exactly once per row.
        self.register("pos_emb", rng.normal(0.0, 0.02, size=(config.block_size, config.n_embd)))
        self.drop = self.add_child("drop", Dropout(config.dropout, rng))
        self.blocks = [
            self.add_child(f"block{index}", Block(config, rng)) for index in range(config.n_layer)
        ]
        self.ln_f = self.add_child("ln_f", LayerNorm(config.n_embd))

    # -- forward -------------------------------------------------------------------------

    def forward(self, ids: np.ndarray) -> np.ndarray:
        """``(B, T)`` token ids -> ``(B, T, vocab_size)`` logits."""
        ids = np.asarray(ids)
        if ids.ndim != 2:
            raise ValueError(f"expected ids of shape (batch, time), got {ids.shape}")
        batch, time = ids.shape
        if time > self.config.block_size:
            raise ValueError(f"sequence length {time} exceeds block size {self.config.block_size}")

        x = self.tok_emb.forward(ids) + self.params["pos_emb"][:time]
        x = self.drop.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        x = self.ln_f.forward(x)

        self._final_hidden = x
        self._time = time
        return x @ self.tok_emb.params["W"].T

    def backward(self, dlogits: np.ndarray) -> None:
        """Accumulate parameter gradients from ``dlogits`` of shape ``(B, T, vocab_size)``.

        The first two statements are the tied-weight path: the output projection contributes
        ``dlogits^T x`` to the embedding gradient, and the lookup contributes a scatter-add later.
        Both land in the same accumulator, which is the point.
        """
        x = self._final_hidden
        embedding = self.tok_emb.params["W"]

        d_flat = dlogits.reshape(-1, self.config.vocab_size)
        x_flat = x.reshape(-1, self.config.n_embd)
        self.tok_emb.grads["W"] += d_flat.T @ x_flat
        dx = (d_flat @ embedding).reshape(x.shape)

        dx = self.ln_f.backward(dx)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        dx = self.drop.backward(dx)

        self.grads["pos_emb"][: self._time] += dx.sum(axis=0)
        self.tok_emb.backward(dx)

    def loss(self, ids: np.ndarray, targets: np.ndarray) -> float:
        """Forward only: mean cross-entropy over all positions."""
        logits = self.forward(ids)
        loss_value, _ = cross_entropy(
            logits.reshape(-1, self.config.vocab_size), np.asarray(targets).reshape(-1)
        )
        return loss_value

    def loss_and_backward(self, ids: np.ndarray, targets: np.ndarray) -> float:
        """Forward, then accumulate gradients. Does not zero them -- the caller decides when."""
        logits = self.forward(ids)
        loss_value, dlogits = cross_entropy(
            logits.reshape(-1, self.config.vocab_size), np.asarray(targets).reshape(-1)
        )
        self.backward(dlogits.reshape(logits.shape))
        return loss_value

    # -- generation ----------------------------------------------------------------------

    def forward_step(self, ids: np.ndarray, position: int, caches=None):
        """One token forward, using and returning per-block KV caches.

        ``position`` is the absolute index of this token, needed because the positional embedding
        is absolute: reusing position 0 for every step is a bug that produces fluent output with
        no sense of order, which is easy to miss by eye.
        """
        ids = np.asarray(ids)
        if ids.shape[1] != 1:
            raise ValueError("forward_step expects a single token per sequence")
        if position >= self.config.block_size:
            raise ValueError(f"position {position} exceeds block size {self.config.block_size}")

        x = self.tok_emb.forward(ids) + self.params["pos_emb"][position : position + 1]
        caches = list(caches) if caches is not None else [None] * len(self.blocks)
        new_caches = []
        for block, cache in zip(self.blocks, caches):
            x, updated = block.forward_step(x, cache)
            new_caches.append(updated)
        x = self.ln_f.forward(x)
        logits = x @ self.tok_emb.params["W"].T
        return logits, new_caches
