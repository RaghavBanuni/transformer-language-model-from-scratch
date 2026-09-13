"""AdamW, clipping, and the schedule -- as exact properties rather than "the loss went down"."""

import numpy as np
import pytest

from minigpt.optim import AdamW, clip_grad_norm, cosine_schedule_with_warmup, global_grad_norm


def test_decoupled_decay_is_exact_geometric_shrinkage():
    """The property that names the optimiser.

    With zero gradients the update must be exactly ``p * (1 - lr*wd)^t``. Adam with L2 folded into
    the gradient cannot satisfy this: the decay term would pass through the ``1/sqrt(v)``
    normaliser, so how strongly a weight is regularised would depend on its own gradient history --
    parameters with large gradients would barely decay at all. That is the bug AdamW fixes, and it
    is checkable rather than a matter of taste.
    """
    start = np.array([[1.0, -2.0, 0.5]])
    weights = {"W": start.copy()}
    zero_gradients = {"W": np.zeros_like(start)}
    lr, decay = 0.1, 0.1
    optimizer = AdamW(weights, lr=lr, weight_decay=decay)

    for step in range(5):
        optimizer.step(zero_gradients)
        assert np.allclose(weights["W"], start * (1.0 - lr * decay) ** (step + 1))


def test_decay_skips_biases_and_gains():
    """Decaying a LayerNorm gain towards zero shrinks the activations it is there to rescale.

    Biases and gains carry no capacity worth regularising, so the convention is to decay matrices
    only. This asserts the 1-D parameters are genuinely left alone.
    """
    parameters = {"W": np.ones((2, 2)), "b": np.ones(2), "gamma": np.ones(3)}
    gradients = {name: np.zeros_like(array) for name, array in parameters.items()}
    AdamW(parameters, lr=0.1, weight_decay=0.5).step(gradients)

    assert np.all(parameters["W"] < 1.0)
    assert np.all(parameters["b"] == 1.0)
    assert np.all(parameters["gamma"] == 1.0)


def test_decay_can_be_applied_to_everything_when_asked():
    parameters = {"b": np.ones(2)}
    AdamW(parameters, lr=0.1, weight_decay=0.5, decay_matrices_only=False).step(
        {"b": np.zeros(2)}
    )
    assert np.all(parameters["b"] < 1.0)


@pytest.mark.parametrize("gradient_scale", [1e-6, 1.0, 1e3, 1e6])
def test_first_step_is_learning_rate_sized_at_any_gradient_scale(gradient_scale):
    """What bias correction buys: scale invariance from the very first step.

    For a constant gradient ``mhat/sqrt(vhat)`` is exactly +/-1, so the step is ``lr`` whatever the
    gradient's magnitude. Without correction, ``v`` starts at zero and is about 1000x too small on
    step one at beta2=0.999, making the first step roughly 32x too large -- which is how a run
    diverges before the schedule has warmed anything up.
    """
    parameter = {"w": np.zeros(1)}
    AdamW(parameter, lr=0.01).step({"w": np.array([gradient_scale])})
    assert abs(float(parameter["w"])) == pytest.approx(0.01, rel=0.02)


def test_sign_of_the_step_opposes_the_gradient():
    parameter = {"w": np.zeros(2)}
    AdamW(parameter, lr=0.1).step({"w": np.array([1.0, -1.0])})
    assert parameter["w"][0] < 0 and parameter["w"][1] > 0


def test_adam_minimises_a_quadratic():
    """An end-to-end sanity check on a problem with a known answer."""
    parameter = {"w": np.array([3.0, -4.0])}
    optimizer = AdamW(parameter, lr=0.1)
    for _ in range(300):
        optimizer.step({"w": 2.0 * parameter["w"]})  # d/dw of sum(w^2)
    assert np.allclose(parameter["w"], 0.0, atol=1e-3)


def test_step_state_is_per_parameter():
    """Sharing one moment buffer across parameters is a real and quiet bug."""
    parameters = {"a": np.zeros(1), "b": np.zeros(1)}
    optimizer = AdamW(parameters, lr=0.1)
    optimizer.step({"a": np.array([1.0]), "b": np.array([0.0])})
    assert parameters["a"][0] != 0.0
    assert parameters["b"][0] == 0.0


def test_updates_are_in_place_so_the_model_sees_them():
    """The optimiser holds the model's own arrays; rebinding them would silently detach it."""
    array = np.ones((2, 2))
    optimizer = AdamW({"W": array}, lr=0.1)
    optimizer.step({"W": np.ones((2, 2))})
    assert array is optimizer.parameters["W"]
    assert np.all(array < 1.0)


def test_optimiser_rejects_missing_or_mismatched_gradients():
    optimizer = AdamW({"W": np.ones((2, 2))}, lr=0.1)
    with pytest.raises(KeyError):
        optimizer.step({})
    with pytest.raises(ValueError, match="shape"):
        optimizer.step({"W": np.ones(2)})


def test_hyperparameter_validation():
    with pytest.raises(ValueError, match="lr"):
        AdamW({"w": np.zeros(1)}, lr=0.0)
    with pytest.raises(ValueError, match="betas"):
        AdamW({"w": np.zeros(1)}, betas=(0.9, 1.0))
    with pytest.raises(ValueError, match="weight_decay"):
        AdamW({"w": np.zeros(1)}, weight_decay=-0.1)


# -- clipping ----------------------------------------------------------------------------


def test_global_norm_is_computed_over_all_parameters_at_once():
    """Per-tensor clipping changes the update *direction*; global clipping only its length."""
    gradients = {"a": np.array([3.0]), "b": np.array([4.0])}
    assert global_grad_norm(gradients) == pytest.approx(5.0)


def test_clipping_preserves_direction_and_reports_the_original_norm():
    gradients = {"a": np.array([3.0, 0.0]), "b": np.array([4.0])}
    direction = np.concatenate([gradients["a"], gradients["b"]]) / 5.0

    reported = clip_grad_norm(gradients, 1.0)
    assert reported == pytest.approx(5.0)  # the pre-clip norm, which is what you want to log

    clipped = np.concatenate([gradients["a"], gradients["b"]])
    assert np.linalg.norm(clipped) == pytest.approx(1.0)
    assert np.allclose(clipped / np.linalg.norm(clipped), direction)


def test_clipping_leaves_small_gradients_untouched():
    gradients = {"a": np.array([0.3, 0.4])}
    before = gradients["a"].copy()
    assert clip_grad_norm(gradients, 1.0) == pytest.approx(0.5)
    assert np.array_equal(gradients["a"], before)


def test_clipping_handles_an_all_zero_gradient():
    gradients = {"a": np.zeros(3)}
    assert clip_grad_norm(gradients, 1.0) == 0.0
    assert np.all(np.isfinite(gradients["a"]))


def test_clipping_rejects_a_non_positive_threshold():
    with pytest.raises(ValueError):
        clip_grad_norm({"a": np.ones(2)}, 0.0)


# -- schedule ----------------------------------------------------------------------------


def test_warmup_ramps_linearly_and_reaches_the_peak_exactly():
    warmup, peak = 10, 1e-3
    values = [cosine_schedule_with_warmup(step, 100, peak, warmup) for step in range(warmup)]
    assert values[-1] == pytest.approx(peak)
    assert all(later > earlier for earlier, later in zip(values, values[1:]))
    assert values[0] == pytest.approx(peak / warmup)


def test_cosine_decays_monotonically_to_the_floor():
    total, peak, floor_ratio = 100, 1e-3, 0.1
    values = [
        cosine_schedule_with_warmup(step, total, peak, 0, floor_ratio) for step in range(total)
    ]
    assert values[0] == pytest.approx(peak)
    assert all(later <= earlier + 1e-15 for earlier, later in zip(values, values[1:]))
    assert values[-1] > peak * floor_ratio  # a floor, not zero: late steps still do something
    assert cosine_schedule_with_warmup(total, total, peak, 0, floor_ratio) == pytest.approx(
        peak * floor_ratio
    )


def test_schedule_never_exceeds_the_peak_or_drops_below_the_floor():
    for step in range(0, 200):
        value = cosine_schedule_with_warmup(step, 100, 1e-3, 10, 0.1)
        assert 1e-4 - 1e-12 <= value <= 1e-3 + 1e-12


def test_schedule_validation():
    with pytest.raises(ValueError, match="total_steps"):
        cosine_schedule_with_warmup(0, 0, 1e-3)
    with pytest.raises(ValueError, match="warmup_steps"):
        cosine_schedule_with_warmup(0, 10, 1e-3, 11)
    with pytest.raises(ValueError, match="min_lr_ratio"):
        cosine_schedule_with_warmup(0, 10, 1e-3, 0, 1.5)
