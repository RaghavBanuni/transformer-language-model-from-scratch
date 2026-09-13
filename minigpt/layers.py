"""Primitive layers, with backward passes derived by hand.

Why write the backward pass at all
----------------------------------
Autodiff makes the chain rule invisible, and what stays invisible stays unlearned. Every
gradient here is derived, written out, and then checked numerically against a central finite
difference (:mod:`minigpt.gradcheck`). The check is what makes the derivation trustworthy: a
wrong backward pass still trains -- badly, slowly, or to a worse optimum -- and looks like a
hyperparameter problem rather than a bug. That failure mode is why every layer in this
repository has a gradient test rather than a loss curve as its evidence.

The module protocol
-------------------
Each layer keeps its parameters in ``self.params`` and *accumulates* into ``self.grads``.
Accumulation rather than assignment is deliberate: a tied parameter receives gradient from more
than one path (the token embedding here is used twice, once as a lookup table and once as the
output projection), and assignment would silently keep only the last contribution. That bug
costs a factor of two on the embedding gradient and is invisible in the loss curve.

``forward`` stores whatever the backward pass needs on ``self``, and ``backward`` returns the
gradient with respect to the layer's input. Parameters are read from ``self.params`` on every
forward call, never cached in a local attribute, so a gradient check can perturb an array in
place and see the effect.

Numerical stability is not optional
-----------------------------------
``softmax`` subtracts the row maximum and :func:`cross_entropy` uses log-sum-exp. Without them,
logits of a few hundred -- entirely reachable during training -- overflow ``exp`` to ``inf`` and
the loss becomes ``nan``, at which point every parameter is ruined and the run is unrecoverable.
A test asserts finite loss at logits of 1e4.
"""

from __future__ import annotations

import numpy as np


class Module:
    """Minimal base class: named parameters, gradient accumulation, train/eval mode."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}
        self.children: dict[str, Module] = {}
        self.training: bool = True

    def register(self, name: str, array: np.ndarray) -> np.ndarray:
        self.params[name] = array
        self.grads[name] = np.zeros_like(array)
        return array

    def add_child(self, name: str, module: "Module") -> "Module":
        self.children[name] = module
        return module

    def named_parameters(self, prefix: str = ""):
        for name, array in self.params.items():
            yield f"{prefix}{name}", array
        for child_name, child in self.children.items():
            yield from child.named_parameters(f"{prefix}{child_name}.")

    def named_gradients(self, prefix: str = ""):
        for name, array in self.grads.items():
            yield f"{prefix}{name}", array
        for child_name, child in self.children.items():
            yield from child.named_gradients(f"{prefix}{child_name}.")

    def n_parameters(self) -> int:
        return sum(array.size for _, array in self.named_parameters())

    def zero_grad(self) -> None:
        """Reset accumulators. Forgetting this is the classic silent training bug."""
        for array in self.grads.values():
            array.fill(0.0)
        for child in self.children.values():
            child.zero_grad()

    def train(self) -> "Module":
        self.training = True
        for child in self.children.values():
            child.train()
        return self

    def eval(self) -> "Module":
        self.training = False
        for child in self.children.values():
            child.eval()
        return self


class Linear(Module):
    """``y = x @ W + b`` over the last axis, for inputs of any leading shape.

    Initialisation is ``N(0, std)`` with ``std = 1/sqrt(in_features)`` by default. The scale is
    not cosmetic: it keeps the variance of the output roughly equal to the variance of the input,
    so activations neither vanish nor explode as depth grows. ``scale`` is exposed because
    residual projections want a smaller value -- see :class:`minigpt.model.Block`.
    """

    def __init__(self, in_features: int, out_features: int, rng: np.random.Generator,
                 bias: bool = True, scale: float | None = None) -> None:
        super().__init__()
        std = 1.0 / np.sqrt(in_features) if scale is None else scale
        self.register("W", rng.normal(0.0, std, size=(in_features, out_features)))
        self.use_bias = bias
        if bias:
            self.register("b", np.zeros(out_features))
        self.in_features = in_features
        self.out_features = out_features
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        y = x @ self.params["W"]
        if self.use_bias:
            y = y + self.params["b"]
        return y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """``dW = x^T dout``, ``db = sum(dout)``, ``dx = dout W^T``.

        Leading axes are flattened so a ``(B, T, C)`` activation and a ``(N, C)`` one take the
        same path; keeping two code paths here is how a shape bug hides.
        """
        x = self._x
        if x is None:
            raise RuntimeError("backward called before forward")
        x_flat = x.reshape(-1, self.in_features)
        d_flat = dout.reshape(-1, self.out_features)
        self.grads["W"] += x_flat.T @ d_flat
        if self.use_bias:
            self.grads["b"] += d_flat.sum(axis=0)
        return (d_flat @ self.params["W"].T).reshape(x.shape)


class Embedding(Module):
    """Integer index -> row of a table.

    The backward pass is a scatter-add, and it must be ``np.add.at`` rather than fancy-index
    assignment: a token that appears twice in a batch has to accumulate both gradients, and
    ``dW[ids] += dout`` silently keeps only one of them because of how NumPy buffers duplicate
    indices.
    """

    def __init__(self, num_embeddings: int, dim: int, rng: np.random.Generator,
                 std: float = 0.02) -> None:
        super().__init__()
        self.register("W", rng.normal(0.0, std, size=(num_embeddings, dim)))
        self.num_embeddings = num_embeddings
        self.dim = dim
        self._ids: np.ndarray | None = None

    def forward(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids)
        if ids.size and (ids.min() < 0 or ids.max() >= self.num_embeddings):
            raise IndexError("token id outside the embedding table")
        self._ids = ids
        return self.params["W"][ids]

    def backward(self, dout: np.ndarray) -> None:
        np.add.at(self.grads["W"], self._ids, dout)
        return None  # no gradient flows to integer indices


class LayerNorm(Module):
    """Normalise the last axis to zero mean and unit variance, then scale and shift.

    The backward pass is the one people get wrong, because ``mu`` and ``sigma`` both depend on
    every element of the row, so each output depends on every input. Writing
    ``xhat = (x - mu) * inv`` and ``dxhat = dout * gamma``, the row gradient collapses to

        dx = inv/D * (D*dxhat - sum(dxhat) - xhat * sum(dxhat * xhat))

    The two subtracted terms are exactly the mean and variance paths; dropping them (the usual
    shortcut) gives a gradient that is close enough to keep training and wrong enough to hurt,
    which is why this has a numerical test rather than a comment claiming it is right.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.register("gamma", np.ones(dim))
        self.register("beta", np.zeros(dim))
        self.dim = dim
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        centred = x - mean
        var = (centred**2).mean(axis=-1, keepdims=True)
        self._inv = 1.0 / np.sqrt(var + self.eps)
        self._xhat = centred * self._inv
        return self._xhat * self.params["gamma"] + self.params["beta"]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        xhat, inv = self._xhat, self._inv
        axes = tuple(range(dout.ndim - 1))
        self.grads["gamma"] += (dout * xhat).sum(axis=axes)
        self.grads["beta"] += dout.sum(axis=axes)

        dxhat = dout * self.params["gamma"]
        dim = self.dim
        return (
            inv
            / dim
            * (
                dim * dxhat
                - dxhat.sum(axis=-1, keepdims=True)
                - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
            )
        )


SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
GELU_COEFF = 0.044715


class GELU(Module):
    """The tanh approximation of GELU, and the derivative *of that approximation*.

        gelu(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))

    Using the exact erf-based derivative with the tanh forward pass is a real and popular bug:
    the two disagree by a few parts in a thousand, which a finite-difference check catches
    immediately and a training run never does. So the derivative here is differentiated from the
    expression actually computed:

        d/dx = 0.5 (1 + t) + 0.5 x (1 - t^2) sqrt(2/pi) (1 + 3*0.044715 x^2)

    where ``t`` is the tanh term.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x**3)
        self._tanh = np.tanh(self._inner)
        return 0.5 * x * (1.0 + self._tanh)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, t = self._x, self._tanh
        d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_COEFF * x**2)
        return dout * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * d_inner)


class Dropout(Module):
    """Inverted dropout: scale by ``1/(1-p)`` at training time, identity at eval time.

    Scaling during training rather than at inference is what makes the eval path a plain
    identity. The alternative -- multiply by ``(1-p)`` at inference -- means the deployed model
    computes something different from the trained one, and forgetting it is a silent accuracy
    loss with no error message.

    Note that ``p=0`` short-circuits entirely, which is what the gradient checks rely on: a
    stochastic function has no well-defined finite difference.
    """

    def __init__(self, p: float = 0.0, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.p == 0.0 or not self.training:
            self._mask = None
            return x
        keep = 1.0 - self.p
        self._mask = (self.rng.random(x.shape) < keep) / keep
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout if self._mask is None else dout * self._mask


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Subtracting the maximum is mathematically a no-op and practically the difference between a
    probability and a ``nan``: ``exp(800)`` overflows in float64, and attention logits reach
    that range once the model is confident.
    """
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=axis, keepdims=True)


def softmax_backward(probabilities: np.ndarray, dout: np.ndarray, axis: int = -1) -> np.ndarray:
    """Jacobian-vector product for softmax, without ever forming the Jacobian.

    For a single row, ``J = diag(p) - p p^T``, so ``J^T g = p * (g - <g, p>)``. Forming the
    ``T x T`` Jacobian per row would cost ``T^2`` memory per attention head for nothing.
    """
    return probabilities * (dout - np.sum(dout * probabilities, axis=axis, keepdims=True))


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cross-entropy over a flat batch, plus its gradient.

    Returns ``(loss, dlogits)``. Softmax and the log are fused rather than composed: computing
    ``log(softmax(x))`` in two steps loses precision for confident predictions and can produce
    ``log(0) = -inf`` where the fused form gives a large finite number.

    The gradient is famously simple -- ``(p - onehot)/N`` -- and that simplicity is a consequence
    of the fusion, not a coincidence. The mean, not the sum, keeps the gradient scale independent
    of batch size, so a learning rate tuned at one batch size survives a change to it.
    """
    logits = np.atleast_2d(logits)
    targets = np.asarray(targets).reshape(-1)
    n_rows, n_classes = logits.shape
    if targets.shape[0] != n_rows:
        raise ValueError(f"got {n_rows} logit rows and {targets.shape[0]} targets")
    if targets.size and (targets.min() < 0 or targets.max() >= n_classes):
        raise ValueError("target index outside the vocabulary")

    log_probabilities = log_softmax(logits, axis=-1)
    rows = np.arange(n_rows)
    loss = float(-log_probabilities[rows, targets].mean())

    dlogits = np.exp(log_probabilities)
    dlogits[rows, targets] -= 1.0
    dlogits /= n_rows
    return loss, dlogits
