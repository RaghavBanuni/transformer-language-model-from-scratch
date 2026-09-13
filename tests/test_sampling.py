"""Decoding: the filters, the sampler, and the two generation paths that must agree."""

import numpy as np
import pytest

from minigpt import demo
from minigpt.layers import softmax
from minigpt.model import TransformerLM
from minigpt.sampling import apply_temperature, generate, sample_from_logits, top_k_filter, top_p_filter

LOGITS = np.log(np.array([[0.40, 0.25, 0.20, 0.10, 0.04, 0.01]]))


def entropy(probabilities: np.ndarray) -> float:
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log(probabilities)))


def test_temperature_sharpens_and_flattens():
    cold = softmax(apply_temperature(LOGITS, 0.5), axis=-1)[0]
    neutral = softmax(LOGITS, axis=-1)[0]
    warm = softmax(apply_temperature(LOGITS, 2.0), axis=-1)[0]
    assert entropy(cold) < entropy(neutral) < entropy(warm)
    assert cold.argmax() == neutral.argmax() == warm.argmax()  # ordering is preserved


def test_temperature_is_applied_to_logits_not_probabilities():
    """Dividing probabilities by T and renormalising is a different, wrong transform."""
    scaled_logits = softmax(apply_temperature(LOGITS, 0.5), axis=-1)
    probabilities = softmax(LOGITS, axis=-1)
    naive = probabilities / 0.5
    naive = naive / naive.sum(axis=-1, keepdims=True)
    assert not np.allclose(scaled_logits, naive)


def test_zero_and_negative_temperature_are_rejected_by_the_filter():
    with pytest.raises(ValueError):
        apply_temperature(LOGITS, 0.0)
    with pytest.raises(ValueError):
        apply_temperature(LOGITS, -1.0)


def test_top_k_keeps_exactly_k_tokens():
    for k in (1, 2, 3):
        filtered = top_k_filter(LOGITS, k)
        assert np.isfinite(filtered).sum() == k
        probabilities = softmax(filtered, axis=-1)
        assert probabilities.sum() == pytest.approx(1.0)
        assert (probabilities > 0).sum() == k


def test_top_k_masks_with_exactly_zero_probability():
    """``-inf`` rather than a large negative number: "almost zero" is still sampleable."""
    probabilities = softmax(top_k_filter(LOGITS, 1), axis=-1)
    assert probabilities[0, 0] == 1.0
    assert np.all(probabilities[0, 1:] == 0.0)


def test_top_k_larger_than_the_vocabulary_is_a_no_op():
    assert np.array_equal(top_k_filter(LOGITS, 99), LOGITS)


def test_top_k_rejects_zero():
    with pytest.raises(ValueError):
        top_k_filter(LOGITS, 0)


def test_top_p_keeps_the_smallest_sufficient_set():
    kept = np.isfinite(top_p_filter(LOGITS, 0.9)).sum()
    assert kept == 4  # 0.40 + 0.25 + 0.20 + 0.10 first reaches 0.9
    assert np.isfinite(top_p_filter(LOGITS, 0.5)).sum() == 2


def test_top_p_is_never_empty_even_when_one_token_dominates():
    """The token that crosses the threshold is kept; the off-by-one here empties the nucleus."""
    peaked = np.log(np.array([[0.99, 0.005, 0.005]]))
    assert np.isfinite(top_p_filter(peaked, 0.5)).sum() == 1


def test_top_p_of_one_keeps_everything():
    assert np.array_equal(top_p_filter(LOGITS, 1.0), LOGITS)


def test_top_p_adapts_to_the_shape_of_the_distribution():
    """The difference from top-k: the same p keeps different counts in different contexts."""
    flat = np.zeros((1, 10))
    peaked = np.log(np.array([[0.9] + [0.1 / 9] * 9]))
    assert np.isfinite(top_p_filter(flat, 0.9)).sum() > np.isfinite(top_p_filter(peaked, 0.9)).sum()


def test_top_p_rejects_impossible_thresholds():
    with pytest.raises(ValueError):
        top_p_filter(LOGITS, 0.0)
    with pytest.raises(ValueError):
        top_p_filter(LOGITS, 1.5)


def test_greedy_decoding_takes_the_argmax():
    ids = sample_from_logits(LOGITS, np.random.default_rng(0), temperature=0.0)
    assert ids.tolist() == [0]


def test_sampling_follows_the_distribution():
    """A frequency check, so the sampler is tested rather than merely exercised."""
    rng = np.random.default_rng(0)
    draws = np.array([sample_from_logits(LOGITS, rng)[0] for _ in range(4000)])
    frequencies = np.bincount(draws, minlength=6) / 4000
    assert np.allclose(frequencies, [0.40, 0.25, 0.20, 0.10, 0.04, 0.01], atol=0.03)


def test_filtered_tokens_are_never_drawn():
    rng = np.random.default_rng(1)
    draws = {sample_from_logits(LOGITS, rng, top_k=2)[0] for _ in range(500)}
    assert draws <= {0, 1}


def test_sampling_is_reproducible_from_a_seed():
    first = [sample_from_logits(LOGITS, np.random.default_rng(7))[0] for _ in range(5)]
    second = [sample_from_logits(LOGITS, np.random.default_rng(7))[0] for _ in range(5)]
    assert first == second


def test_sampling_handles_a_batch():
    logits = np.concatenate([LOGITS, LOGITS[:, ::-1]], axis=0)
    ids = sample_from_logits(logits, np.random.default_rng(0), temperature=0.0)
    assert ids.tolist() == [0, 5]


# -- generation --------------------------------------------------------------------------


def test_generation_extends_the_prompt_without_altering_it():
    model = demo.tiny_model(n_embd=32)
    prompt = np.array([[1, 2, 3]])
    out = generate(model, prompt, 4, np.random.default_rng(0))
    assert out.shape == (1, 7)
    assert out[0, :3].tolist() == [1, 2, 3]


def test_cached_and_uncached_generation_agree():
    """Two independent implementations of the same function, checked against each other.

    Greedy decoding makes the comparison exact: any divergence in the logits large enough to change
    an argmax shows up as a different token, and the sequences are compared element by element.
    """
    model = demo.tiny_model(n_embd=32)
    prompt = np.array([[5, 6]])
    cached = generate(model, prompt, 5, np.random.default_rng(0), temperature=0.0, use_cache=True)
    uncached = generate(model, prompt, 5, np.random.default_rng(0), temperature=0.0, use_cache=False)
    assert np.array_equal(cached, uncached)


def test_generation_runs_in_eval_mode_and_restores_the_previous_mode():
    """Generating with dropout live samples from a noisier model than the one that was trained.

    Nothing errors when that happens, which is what makes it worth a test: the only symptom is
    slightly worse output.
    """
    config = demo.tiny_config(dropout=0.5, n_embd=32)
    model = TransformerLM(config, seed=0)
    assert model.training

    prompt = np.array([[1, 2]])
    first = generate(model, prompt, 4, np.random.default_rng(0), temperature=0.0)
    second = generate(model, prompt, 4, np.random.default_rng(0), temperature=0.0)
    assert np.array_equal(first, second)  # deterministic, so dropout was off
    assert model.training  # and the mode was handed back


def test_generation_refuses_to_exceed_the_block_size():
    """This model has learned absolute positions only up to block_size; there is no window."""
    model = demo.tiny_model()
    with pytest.raises(ValueError, match="block"):
        generate(model, np.array([[1, 2, 3]]), model.config.block_size, np.random.default_rng(0))


def test_generation_needs_a_prompt():
    model = demo.tiny_model()
    with pytest.raises(ValueError, match="prompt"):
        generate(model, np.zeros((1, 0), dtype=int), 2, np.random.default_rng(0))


def test_generation_accepts_a_one_dimensional_prompt():
    model = demo.tiny_model(n_embd=32)
    out = generate(model, np.array([1, 2]), 3, np.random.default_rng(0))
    assert out.shape == (1, 5)
