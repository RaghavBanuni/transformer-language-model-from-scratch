"""Numerical gradient checking -- the only reason to trust a hand-derived backward pass.

The method
----------
For a scalar loss ``L(theta)``, the central difference

    dL/dtheta_i  ~=  (L(theta_i + h) - L(theta_i - h)) / (2h)

has error ``O(h^2)``, against ``O(h)`` for the one-sided version. That is not a detail: at
``h = 1e-5`` the central form is accurate to about 1e-10 while the forward form is accurate to
about 1e-5, which is the same order as a plausible bug.

Choosing ``h`` is a genuine trade-off. Too large and the truncation error dominates; too small
and catastrophic cancellation in ``L(+h) - L(-h)`` dominates, since two nearly equal float64
numbers lose most of their significant digits on subtraction. Around 1e-5 is the usual sweet
spot for float64, and this module defaults there.

The comparison is a *relative* error,

    |analytic - numeric| / max(|analytic| + |numeric|, eps)

because an absolute threshold is meaningless across parameters whose gradients differ by many
orders of magnitude. Under 1e-6 is right; 1e-2 or worse usually means a missing term, and
exactly 1.0 means one of the two is zero -- typically a gradient that never got accumulated.

Cost
----
Two forward passes per parameter *element*. A model with 100k parameters would need 200k forward
passes, so gradient checking is for tiny configurations only -- which is exactly enough, because
correctness of the chain rule does not depend on width. Sampling a subset of coordinates keeps
it affordable, and the sampling must be seeded so a failure can be reproduced.

Randomness must be off. Dropout makes ``L`` stochastic, and a finite difference of a stochastic
function measures noise. Every check here runs with dropout disabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def relative_error(analytic: np.ndarray | float, numeric: np.ndarray | float,
                   eps: float = 1e-12) -> np.ndarray:
    """Symmetric relative difference, safe when both values are zero."""
    analytic = np.asarray(analytic, dtype=float)
    numeric = np.asarray(numeric, dtype=float)
    denominator = np.maximum(np.abs(analytic) + np.abs(numeric), eps)
    return np.abs(analytic - numeric) / denominator


@dataclass(frozen=True)
class GradCheckResult:
    """Outcome of checking one parameter array."""

    name: str
    n_checked: int
    max_relative_error: float
    worst_analytic: float
    worst_numeric: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.max_relative_error <= self.tolerance

    def __str__(self) -> str:
        status = "ok  " if self.passed else "FAIL"
        return (
            f"{status} {self.name:<28} n={self.n_checked:<4} "
            f"max rel err {self.max_relative_error:.3e} "
            f"(analytic {self.worst_analytic:+.6f} vs numeric {self.worst_numeric:+.6f})"
        )


def check_parameter(
    loss_fn: Callable[[], float],
    parameter: np.ndarray,
    analytic_grad: np.ndarray,
    name: str = "param",
    n_samples: int = 12,
    h: float = 1e-5,
    tolerance: float = 1e-6,
    seed: int = 0,
) -> GradCheckResult:
    """Compare an analytic gradient against central differences at sampled coordinates.

    ``loss_fn`` must recompute the loss from scratch using the *current* contents of
    ``parameter``, which is perturbed in place and restored exactly afterwards. Restoring the
    saved float rather than adding ``h`` back matters: ``(x + h) - h != x`` in floating point,
    and a drifting parameter turns a later check into a mystery.
    """
    if parameter.shape != analytic_grad.shape:
        raise ValueError(f"{name}: parameter shape {parameter.shape} != grad shape {analytic_grad.shape}")

    rng = np.random.default_rng(seed)
    flat_parameter = parameter.reshape(-1)
    flat_grad = analytic_grad.reshape(-1)
    n_samples = min(n_samples, flat_parameter.size)
    indices = rng.choice(flat_parameter.size, size=n_samples, replace=False)

    worst = (-1.0, 0.0, 0.0)
    for index in indices:
        original = flat_parameter[index]

        flat_parameter[index] = original + h
        loss_plus = loss_fn()
        flat_parameter[index] = original - h
        loss_minus = loss_fn()
        flat_parameter[index] = original

        numeric = (loss_plus - loss_minus) / (2.0 * h)
        analytic = float(flat_grad[index])
        error = float(relative_error(analytic, numeric))
        if error > worst[0]:
            worst = (error, analytic, numeric)

    return GradCheckResult(
        name=name,
        n_checked=n_samples,
        max_relative_error=worst[0],
        worst_analytic=worst[1],
        worst_numeric=worst[2],
        tolerance=tolerance,
    )


def check_module(
    module,
    loss_fn: Callable[[], float],
    n_samples: int = 8,
    h: float = 1e-5,
    tolerance: float = 1e-6,
    seed: int = 0,
) -> list[GradCheckResult]:
    """Check every named parameter of a module.

    The caller is responsible for having run a forward and backward pass first, so that
    ``module.named_gradients()`` holds the analytic values being tested. Gradients are read
    before any perturbation, since the perturbed forward passes would otherwise overwrite the
    cached activations the analytic values came from.
    """
    gradients = {name: array.copy() for name, array in module.named_gradients()}
    results = []
    for name, array in module.named_parameters():
        results.append(
            check_parameter(
                loss_fn,
                array,
                gradients[name],
                name=name,
                n_samples=n_samples,
                h=h,
                tolerance=tolerance,
                seed=seed,
            )
        )
    return results


def check_input_gradient(
    forward_fn: Callable[[np.ndarray], float],
    x: np.ndarray,
    analytic_grad: np.ndarray,
    n_samples: int = 12,
    h: float = 1e-5,
    tolerance: float = 1e-6,
    seed: int = 0,
) -> GradCheckResult:
    """Same check for the gradient with respect to a layer's input.

    Worth doing separately from the parameter check: a layer can have correct parameter
    gradients and a wrong input gradient, in which case it trains fine in isolation and
    poisons every layer beneath it.
    """
    rng = np.random.default_rng(seed)
    flat_x = x.reshape(-1).copy()
    flat_grad = analytic_grad.reshape(-1)
    n_samples = min(n_samples, flat_x.size)
    indices = rng.choice(flat_x.size, size=n_samples, replace=False)

    worst = (-1.0, 0.0, 0.0)
    for index in indices:
        original = flat_x[index]

        flat_x[index] = original + h
        loss_plus = forward_fn(flat_x.reshape(x.shape))
        flat_x[index] = original - h
        loss_minus = forward_fn(flat_x.reshape(x.shape))
        flat_x[index] = original

        numeric = (loss_plus - loss_minus) / (2.0 * h)
        analytic = float(flat_grad[index])
        error = float(relative_error(analytic, numeric))
        if error > worst[0]:
            worst = (error, analytic, numeric)

    return GradCheckResult("input", n_samples, worst[0], worst[1], worst[2], tolerance)
