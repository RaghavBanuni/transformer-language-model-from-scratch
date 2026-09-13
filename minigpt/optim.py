"""AdamW, gradient clipping, and a cosine schedule with warmup.

What AdamW fixes, precisely
---------------------------
The obvious way to add weight decay to Adam is to add ``wd * p`` to the gradient. That is L2
regularisation, and inside Adam it does something nobody intends: the added term goes through the
same ``1/sqrt(v)`` normalisation as the real gradient, so a parameter with a long history of large
gradients gets *less* decay than a quiet one. The strength of your regulariser ends up coupled to
the gradient statistics of each individual weight.

AdamW decouples them (Loshchilov and Hutter, 2019): the adaptive step is computed from the
gradient alone, and the decay is applied directly to the parameter::

    p <- p - lr * mhat/(sqrt(vhat) + eps)  -  lr * wd * p

This has a clean, testable consequence. With zero gradients the update reduces to exact geometric
shrinkage, ``p_t = p_0 (1 - lr*wd)^t``, which ``tests/test_optim.py`` asserts to floating-point
tolerance. The coupled version cannot satisfy that, so the test distinguishes the two
implementations rather than merely exercising one.

Why bias correction is not optional
-----------------------------------
``m`` and ``v`` start at zero, so early estimates are biased towards zero by exactly ``1 - beta^t``.
At ``t=1`` with ``beta2=0.999``, ``v`` is a thousand times too small, so ``1/sqrt(v)`` is about 32
times too large -- a first step 32x the intended size, straight into a region the model may never
recover from. Dividing by ``1 - beta^t`` removes it exactly.

Corrected, the first step has magnitude almost exactly ``lr`` no matter how large or small the
gradient is, because ``mhat/sqrt(vhat)`` is ``+/-1`` for a constant gradient. That scale invariance
is Adam's real selling point and is asserted with gradients of 1e-6 and 1e3.

Which parameters get decayed
----------------------------
Biases and LayerNorm gains are excluded. They are one per channel rather than one per connection,
they are not a capacity knob, and shrinking a LayerNorm gain towards zero attenuates the signal
path itself. The rule here is "decay 2-D tensors, leave 1-D ones alone", which is what most real
implementations do and is worth stating rather than leaving as a mysterious parameter group.

Clipping
--------
Gradient clipping rescales the *global* norm across all parameters, not each tensor separately.
Per-tensor clipping changes the direction of the update; global clipping preserves it and only
shortens the step, which is the entire intent -- survive a rare bad batch without distorting the
descent direction.
"""

from __future__ import annotations

import numpy as np


def global_grad_norm(gradients: dict[str, np.ndarray]) -> float:
    """L2 norm of all gradients concatenated."""
    total = sum(float(np.sum(array**2)) for array in gradients.values())
    return float(np.sqrt(total))


def clip_grad_norm(gradients: dict[str, np.ndarray], max_norm: float) -> float:
    """Scale every gradient by the same factor if the global norm exceeds ``max_norm``.

    Returns the norm *before* clipping, which is the number worth logging: a training run whose
    gradient norm spikes by three orders of magnitude has a data problem, and clipping hides the
    symptom while the log preserves the evidence.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    norm = global_grad_norm(gradients)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for array in gradients.values():
            array *= scale
    return norm


class AdamW:
    """AdamW with decoupled weight decay and bias correction."""

    def __init__(
        self,
        parameters: dict[str, np.ndarray],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decay_matrices_only: bool = True,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("betas must be in [0, 1)")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.decay_matrices_only = decay_matrices_only
        self.step_count = 0
        self.m = {name: np.zeros_like(array) for name, array in parameters.items()}
        self.v = {name: np.zeros_like(array) for name, array in parameters.items()}

    def decays(self, name: str, array: np.ndarray) -> bool:
        return not (self.decay_matrices_only and array.ndim < 2)

    def step(self, gradients: dict[str, np.ndarray], lr: float | None = None) -> None:
        """One update. ``lr`` overrides the constructor value, which is how schedules attach."""
        learning_rate = self.lr if lr is None else lr
        self.step_count += 1
        bias_correction1 = 1.0 - self.beta1**self.step_count
        bias_correction2 = 1.0 - self.beta2**self.step_count

        for name, array in self.parameters.items():
            if name not in gradients:
                raise KeyError(f"no gradient supplied for parameter {name!r}")
            gradient = gradients[name]
            if gradient.shape != array.shape:
                raise ValueError(f"{name}: gradient shape {gradient.shape} != {array.shape}")

            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * gradient
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * gradient**2

            m_hat = self.m[name] / bias_correction1
            v_hat = self.v[name] / bias_correction2

            # In place, because the parameter arrays are shared with the model.
            array -= learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay and self.decays(name, array):
                array -= learning_rate * self.weight_decay * array


def cosine_schedule_with_warmup(
    step: int,
    total_steps: int,
    max_lr: float,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup, then cosine decay to ``min_lr_ratio * max_lr``.

    Warmup exists because Adam's second-moment estimate is garbage for the first few steps -- it
    has seen one or two gradients -- so full-size steps at that point are taken in a direction the
    optimiser has no information about. Ramping the learning rate lets ``v`` fill in first.

    The cosine tail matters for a different reason: late training needs small steps to settle into
    a minimum rather than bouncing across it, and decaying to a small floor instead of exactly
    zero keeps the last few thousand steps from being pure no-ops.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps]")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(max_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine))
