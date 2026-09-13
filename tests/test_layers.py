"""Every layer's backward pass, against central finite differences.

The pattern throughout: build a scalar loss ``sum(out * w)`` for a fixed random ``w``, so the
incoming gradient is exactly ``w`` and the analytic result can be compared coordinate by
coordinate. Any structured choice of ``w`` -- all ones, say -- risks a bug that cancels; random
weights do not.
"""

import numpy as np
import pytest

from minigpt.gradcheck import check_input_gradient, check_parameter, relative_error
from minigpt.layers import (
    GELU,
    Dropout,
    Embedding,
    LayerNorm,
    Linear,
    cross_entropy,
    log_softmax,
    softmax,
    softmax_backward,
)

TOLERANCE = 1e-6


def scalar_loss(layer, x, weights):
    """Forward, contract with fixed weights, and return the scalar plus the seed gradient."""
    out = layer.forward(x)
    return float(np.sum(out * weights)), weights


# -- Linear ------------------------------------------------------------------------------


def test_linear_parameter_and_input_gradients():
    rng = np.random.default_rng(0)
    layer = Linear(5, 3, rng)
    x = rng.normal(size=(4, 5))
    weights = rng.normal(size=(4, 3))

    loss, seed = scalar_loss(layer, x, weights)
    layer.backward(seed)

    def loss_fn():
        return float(np.sum(layer.forward(x) * weights))

    for name in ("W", "b"):
        result = check_parameter(loss_fn, layer.params[name], layer.grads[name], name=name)
        assert result.passed, str(result)

    layer.zero_grad()
    dx = layer.backward(scalar_loss(layer, x, weights)[1])
    input_result = check_input_gradient(
        lambda probe: float(np.sum(layer.forward(probe) * weights)), x, dx
    )
    assert input_result.passed, str(input_result)


def test_linear_handles_three_dimensional_activations():
    """A (B, T, C) activation must take the same code path as (N, C)."""
    rng = np.random.default_rng(1)
    layer = Linear(6, 4, rng)
    x = rng.normal(size=(2, 3, 6))
    weights = rng.normal(size=(2, 3, 4))

    out = layer.forward(x)
    assert out.shape == (2, 3, 4)
    dx = layer.backward(weights)
    assert dx.shape == x.shape

    result = check_parameter(
        lambda: float(np.sum(layer.forward(x) * weights)), layer.params["W"], layer.grads["W"], "W"
    )
    assert result.passed, str(result)


def test_linear_accumulates_across_two_backward_calls():
    """Accumulation, not assignment: the tied embedding depends on it."""
    rng = np.random.default_rng(2)
    layer = Linear(3, 2, rng)
    x = rng.normal(size=(2, 3))
    dout = np.ones((2, 2))

    layer.forward(x)
    layer.backward(dout)
    first = layer.grads["W"].copy()
    layer.forward(x)
    layer.backward(dout)
    assert np.allclose(layer.grads["W"], 2 * first)


def test_linear_without_bias_has_no_bias_parameter():
    layer = Linear(3, 2, np.random.default_rng(0), bias=False)
    assert "b" not in layer.params
    assert layer.forward(np.zeros((1, 3))).shape == (1, 2)


def test_backward_before_forward_is_an_error_not_a_wrong_answer():
    layer = Linear(3, 2, np.random.default_rng(0))
    with pytest.raises(RuntimeError):
        layer.backward(np.ones((1, 2)))


# -- Embedding ---------------------------------------------------------------------------


def test_embedding_accumulates_duplicate_indices():
    """The np.add.at requirement, as a test.

    ``dW[ids] += dout`` silently keeps one contribution when an index repeats, because NumPy
    buffers the duplicate writes. A token appearing twice in a batch is entirely ordinary, so this
    bug is always live.
    """
    layer = Embedding(5, 3, np.random.default_rng(0))
    ids = np.array([[2, 2, 4]])
    layer.forward(ids)
    layer.backward(np.ones((1, 3, 3)))
    assert np.allclose(layer.grads["W"][2], 2.0)
    assert np.allclose(layer.grads["W"][4], 1.0)
    assert np.allclose(layer.grads["W"][0], 0.0)


def test_embedding_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    layer = Embedding(6, 4, rng)
    ids = np.array([[1, 3, 3]])
    weights = rng.normal(size=(1, 3, 4))

    layer.forward(ids)
    layer.backward(weights)
    result = check_parameter(
        lambda: float(np.sum(layer.forward(ids) * weights)),
        layer.params["W"],
        layer.grads["W"],
        "W",
        n_samples=10,
    )
    assert result.passed, str(result)


def test_embedding_rejects_out_of_range_ids():
    layer = Embedding(4, 2, np.random.default_rng(0))
    with pytest.raises(IndexError):
        layer.forward(np.array([[4]]))


# -- LayerNorm ---------------------------------------------------------------------------


def test_layernorm_normalises_each_row():
    rng = np.random.default_rng(4)
    layer = LayerNorm(8)
    x = rng.normal(3.0, 5.0, size=(3, 8))  # deliberately not zero-mean or unit-variance
    out = layer.forward(x)
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-12)
    assert np.allclose(out.var(axis=-1), 1.0, atol=1e-4)


def test_layernorm_gradients_include_the_mean_and_variance_paths():
    """The full three-term gradient, not the popular shortcut.

    Because mu and sigma depend on every element of the row, each output depends on every input.
    Dropping the two correction terms leaves a gradient close enough to keep training and wrong
    enough to hurt; finite differences reject it immediately.
    """
    rng = np.random.default_rng(5)
    layer = LayerNorm(6)
    layer.params["gamma"] += rng.normal(0, 0.2, size=6)
    layer.params["beta"] += rng.normal(0, 0.2, size=6)
    x = rng.normal(size=(3, 6))
    weights = rng.normal(size=(3, 6))

    layer.forward(x)
    dx = layer.backward(weights)

    def loss_fn():
        return float(np.sum(layer.forward(x) * weights))

    for name in ("gamma", "beta"):
        result = check_parameter(loss_fn, layer.params[name], layer.grads[name], name)
        assert result.passed, str(result)

    input_result = check_input_gradient(
        lambda probe: float(np.sum(layer.forward(probe) * weights)), x, dx
    )
    assert input_result.passed, str(input_result)


def test_layernorm_survives_a_constant_row():
    """Zero variance would divide by zero without eps; the output must stay finite."""
    layer = LayerNorm(4)
    out = layer.forward(np.full((1, 4), 7.0))
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)


def test_layernorm_works_on_three_dimensional_input():
    rng = np.random.default_rng(6)
    layer = LayerNorm(5)
    x = rng.normal(size=(2, 3, 5))
    weights = rng.normal(size=(2, 3, 5))
    layer.forward(x)
    layer.backward(weights)
    assert layer.grads["gamma"].shape == (5,)
    result = check_parameter(
        lambda: float(np.sum(layer.forward(x) * weights)),
        layer.params["gamma"],
        layer.grads["gamma"],
        "gamma",
    )
    assert result.passed, str(result)


# -- GELU --------------------------------------------------------------------------------


def test_gelu_values_at_reference_points():
    layer = GELU()
    out = layer.forward(np.array([-10.0, 0.0, 10.0]))
    assert np.isclose(out[1], 0.0)
    assert np.isclose(out[2], 10.0, atol=1e-6)   # saturates to identity
    assert abs(out[0]) < 1e-6                     # and to zero


def test_gelu_derivative_matches_the_forward_it_is_paired_with():
    """Using the erf derivative with the tanh forward is a real and popular mismatch.

    The two agree to a few parts in a thousand -- invisible in training, immediately visible here.
    """
    rng = np.random.default_rng(7)
    layer = GELU()
    x = rng.normal(size=(4, 5))
    weights = rng.normal(size=(4, 5))
    layer.forward(x)
    dx = layer.backward(weights)
    result = check_input_gradient(
        lambda probe: float(np.sum(layer.forward(probe) * weights)), x, dx, n_samples=15
    )
    assert result.passed, str(result)


def test_gelu_is_not_relu():
    """A small negative input keeps a small negative output, which is the entire difference."""
    out = GELU().forward(np.array([-0.5]))
    assert -0.2 < out[0] < 0.0


# -- Dropout -----------------------------------------------------------------------------


def test_dropout_is_identity_at_eval():
    layer = Dropout(0.5, np.random.default_rng(0))
    x = np.ones((4, 100))
    layer.eval()
    assert np.array_equal(layer.forward(x), x)


def test_dropout_preserves_expectation_during_training():
    """Inverted dropout scales by 1/(1-p) so the mean is unchanged."""
    layer = Dropout(0.5, np.random.default_rng(1))
    x = np.ones((200, 200))
    out = layer.forward(x)
    assert abs(out.mean() - 1.0) < 0.02
    assert np.any(out == 0.0)  # something was actually dropped


def test_dropout_backward_uses_the_same_mask():
    layer = Dropout(0.5, np.random.default_rng(2))
    x = np.ones((10, 10))
    out = layer.forward(x)
    dx = layer.backward(np.ones_like(x))
    assert np.array_equal(out == 0.0, dx == 0.0)


def test_dropout_zero_probability_short_circuits():
    """Gradient checking depends on this: a stochastic loss has no finite difference."""
    layer = Dropout(0.0)
    x = np.arange(6.0).reshape(2, 3)
    assert np.array_equal(layer.forward(x), x)
    assert np.array_equal(layer.backward(x), x)


def test_dropout_rejects_impossible_probability():
    with pytest.raises(ValueError):
        Dropout(1.0)


# -- softmax and cross-entropy -----------------------------------------------------------


def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(8)
    probabilities = softmax(rng.normal(size=(5, 7)) * 10)
    assert np.allclose(probabilities.sum(axis=-1), 1.0)
    assert np.all(probabilities >= 0)


def test_softmax_survives_logits_that_would_overflow_exp():
    """exp(10000) is inf; the max-subtraction is what keeps this finite."""
    probabilities = softmax(np.array([[1e4, 1e4 - 1.0, -1e4]]))
    assert np.all(np.isfinite(probabilities))
    assert np.isclose(probabilities.sum(), 1.0)


def test_softmax_is_shift_invariant():
    logits = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(softmax(logits), softmax(logits + 100.0))


def test_softmax_backward_matches_finite_differences():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(3, 5))
    weights = rng.normal(size=(3, 5))

    probabilities = softmax(x)
    analytic = softmax_backward(probabilities, weights)

    result = check_input_gradient(
        lambda probe: float(np.sum(softmax(probe) * weights)), x, analytic, n_samples=15
    )
    assert result.passed, str(result)


def test_log_softmax_agrees_with_log_of_softmax_where_both_are_stable():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(4, 6))
    assert np.allclose(log_softmax(x), np.log(softmax(x)))


def test_uniform_logits_give_exactly_log_vocab_size():
    """The identity every training run should be checked against at step 0."""
    for vocab_size in (2, 10, 256, 5000):
        loss, _ = cross_entropy(np.zeros((3, vocab_size)), np.zeros(3, dtype=int))
        assert np.isclose(loss, np.log(vocab_size))


def test_cross_entropy_gradient_matches_finite_differences():
    rng = np.random.default_rng(11)
    logits = rng.normal(size=(4, 7))
    targets = rng.integers(0, 7, size=4)

    _, analytic = cross_entropy(logits, targets)
    result = check_input_gradient(
        lambda probe: cross_entropy(probe, targets)[0], logits, analytic, n_samples=20
    )
    assert result.passed, str(result)


def test_cross_entropy_is_stable_at_extreme_logits():
    """The fused log-sum-exp form, rather than log(softmax(x)) in two steps."""
    loss, gradient = cross_entropy(np.array([[1e4, -1e4]]), np.array([0]))
    assert np.isfinite(loss) and loss < 1e-6
    assert np.all(np.isfinite(gradient))

    loss, gradient = cross_entropy(np.array([[-1e4, 1e4]]), np.array([0]))
    assert np.isfinite(loss) and loss > 1000
    assert np.all(np.isfinite(gradient))


def test_cross_entropy_averages_rather_than_sums():
    """The mean keeps the gradient scale independent of batch size."""
    logits = np.array([[2.0, 0.5, -1.0]])
    targets = np.array([1])
    single_loss, single_grad = cross_entropy(logits, targets)

    repeated_loss, repeated_grad = cross_entropy(
        np.repeat(logits, 8, axis=0), np.repeat(targets, 8)
    )
    assert np.isclose(single_loss, repeated_loss)
    assert np.allclose(repeated_grad[0], single_grad[0] / 8)


def test_cross_entropy_validates_shapes_and_targets():
    with pytest.raises(ValueError, match="logit rows"):
        cross_entropy(np.zeros((3, 4)), np.zeros(2, dtype=int))
    with pytest.raises(ValueError, match="outside the vocabulary"):
        cross_entropy(np.zeros((1, 4)), np.array([4]))


def test_relative_error_handles_two_zeros():
    assert relative_error(0.0, 0.0) == 0.0
    assert relative_error(1.0, -1.0) == 1.0
