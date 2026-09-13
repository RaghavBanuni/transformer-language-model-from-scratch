"""Attention: the mask, the scale, the backward pass, and the cached step."""

import numpy as np
import pytest

from minigpt.attention import CausalSelfAttention, causal_mask
from minigpt.gradcheck import check_input_gradient, check_parameter
from minigpt.layers import softmax


def build(n_embd=8, n_head=2, seed=0):
    return CausalSelfAttention(n_embd, n_head, np.random.default_rng(seed), dropout=0.0)


def test_mask_is_lower_triangular_and_includes_the_diagonal():
    mask = causal_mask(4)
    assert mask.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_output_shape_matches_input():
    layer = build()
    x = np.random.default_rng(1).normal(size=(2, 5, 8))
    assert layer.forward(x).shape == x.shape


def test_a_future_token_cannot_change_an_earlier_position():
    """The single most consequential property in the file.

    Training on every position at once is only valid if position t cannot see t+1. A leaky mask
    produces a wonderful training loss and useless generation, because at inference the future is
    genuinely absent. Masked entries are exactly zero (``-inf`` before the softmax, not a large
    negative number), so the earlier outputs are bit-identical rather than merely close -- and this
    asserts equality, which is a far sharper claim than a tolerance.
    """
    layer = build().eval()
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1, 6, 8))

    baseline = layer.forward(x)
    perturbed_input = x.copy()
    perturbed_input[0, 5, :] += 100.0  # a violent change, to the last position only
    perturbed = layer.forward(perturbed_input)

    assert np.array_equal(baseline[:, :5, :], perturbed[:, :5, :])
    assert not np.allclose(baseline[:, 5, :], perturbed[:, 5, :])


def test_attention_probabilities_are_a_causal_distribution():
    layer = build()
    x = np.random.default_rng(3).normal(size=(1, 4, 8))
    layer.forward(x)
    probabilities = layer._cache[3]

    assert np.allclose(probabilities.sum(axis=-1), 1.0)
    upper = np.triu(np.ones((4, 4), dtype=bool), k=1)
    assert np.all(probabilities[..., upper] == 0.0)
    # Position 0 can attend only to itself, so its distribution is degenerate.
    assert np.allclose(probabilities[:, :, 0, 0], 1.0)


def test_scaling_keeps_the_softmax_out_of_saturation():
    """Why 1/sqrt(d_head) is structural rather than a tuning constant.

    Dot products over ``d_head`` dimensions of unit-variance vectors have standard deviation
    ``sqrt(d_head)`` -- a spread of 8 at a head size of 64. Feed that to a softmax and it collapses
    onto one entry, where its gradient is almost exactly zero and learning stops.
    """
    rng = np.random.default_rng(4)
    head_dim = 64
    q = rng.normal(size=(256, head_dim))
    k = rng.normal(size=(256, head_dim))
    unscaled = q @ k.T
    scaled = unscaled / np.sqrt(head_dim)

    assert unscaled.std() > 6.0
    assert 0.7 < scaled.std() < 1.4

    saturated = softmax(unscaled[:1], axis=-1)
    healthy = softmax(scaled[:1], axis=-1)
    # The scaled distribution is spread over many tokens; the unscaled one is not.
    assert saturated.max() > 10 * healthy.max()


def test_attention_parameter_and_input_gradients():
    layer = build(n_embd=8, n_head=2, seed=5)
    rng = np.random.default_rng(6)
    x = rng.normal(size=(2, 4, 8))
    weights = rng.normal(size=(2, 4, 8))

    def loss_fn():
        return float(np.sum(layer.forward(x) * weights))

    layer.zero_grad()
    layer.forward(x)
    dx = layer.backward(weights)

    gradients = {name: array.copy() for name, array in layer.named_gradients()}
    for name, array in layer.named_parameters():
        result = check_parameter(loss_fn, array, gradients[name], name=name, n_samples=6)
        assert result.passed, str(result)

    input_result = check_input_gradient(
        lambda probe: float(np.sum(layer.forward(probe) * weights)), x, dx, n_samples=12
    )
    assert input_result.passed, str(input_result)


@pytest.mark.parametrize("n_head", [1, 2, 4])
def test_gradients_are_correct_for_any_head_count(n_head):
    """Head splitting and merging must be exact inverses at every extreme."""
    layer = build(n_embd=8, n_head=n_head, seed=7)
    rng = np.random.default_rng(8)
    x = rng.normal(size=(1, 3, 8))
    weights = rng.normal(size=(1, 3, 8))

    layer.zero_grad()
    layer.forward(x)
    layer.backward(weights)

    result = check_parameter(
        lambda: float(np.sum(layer.forward(x) * weights)),
        layer.qkv.params["W"],
        layer.qkv.grads["W"],
        name=f"qkv.W (heads={n_head})",
        n_samples=6,
    )
    assert result.passed, str(result)


def test_head_split_and_merge_round_trip():
    layer = build(n_embd=12, n_head=3)
    x = np.arange(2 * 4 * 12, dtype=float).reshape(2, 4, 12)
    assert np.array_equal(layer._merge_heads(layer._split_heads(x)), x)


def test_cached_step_reproduces_the_full_forward():
    """Two code paths for one function, so they are checked against each other.

    The cache is what makes generation O(T) per token instead of O(T^2). If it disagreed with the
    training-time path, the model would generate from a slightly different function than the one
    that was trained -- a discrepancy that no loss curve can reveal.
    """
    layer = build(n_embd=8, n_head=2, seed=9).eval()
    x = np.random.default_rng(10).normal(size=(1, 5, 8))

    full = layer.forward(x)

    cache = None
    stepped = []
    for position in range(x.shape[1]):
        out, cache = layer.forward_step(x[:, position : position + 1, :], cache)
        stepped.append(out)

    assert np.allclose(full, np.concatenate(stepped, axis=1), atol=1e-10)


def test_cache_grows_by_one_position_per_step():
    layer = build().eval()
    x = np.random.default_rng(11).normal(size=(1, 3, 8))
    cache = None
    for position in range(3):
        _, cache = layer.forward_step(x[:, position : position + 1, :], cache)
        assert cache[0].shape[2] == position + 1
        assert cache[1].shape[2] == position + 1


def test_forward_step_rejects_more_than_one_token():
    layer = build().eval()
    with pytest.raises(ValueError):
        layer.forward_step(np.zeros((1, 2, 8)))


def test_head_count_must_divide_the_embedding_width():
    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(10, 3, np.random.default_rng(0))


def test_wrong_input_width_is_rejected():
    layer = build(n_embd=8)
    with pytest.raises(ValueError):
        layer.forward(np.zeros((1, 3, 7)))
