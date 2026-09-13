"""Batching, the loop, and the diagnostics -- including the one assertion that matters most."""

import numpy as np
import pytest

from minigpt import demo
from minigpt.model import ModelConfig, TransformerLM
from minigpt.train import (
    Dataset,
    evaluate,
    initial_loss_report,
    overfit_batch,
    train,
    uniform_loss,
)


def small_model(seed: int = 0) -> TransformerLM:
    config = ModelConfig(vocab_size=256, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    return TransformerLM(config, seed=seed)


# -- the dataset -------------------------------------------------------------------------


def test_split_is_contiguous_and_covers_the_stream():
    """Why the split is contiguous rather than random windows.

    With random windows drawn from one stream, a validation window overlaps training windows almost
    surely -- so the model has already seen the continuation it is being scored on. The resulting
    number looks like generalisation and is really memorisation, which is one of the easiest ways to
    fool yourself. Contiguous halves cannot overlap.
    """
    tokens = np.arange(100)
    dataset = Dataset(tokens, train_fraction=0.8)
    assert len(dataset.train) == 80
    assert len(dataset.val) == 20
    assert np.array_equal(np.concatenate([dataset.train, dataset.val]), tokens)
    assert dataset.train.max() < dataset.val.min()  # no shared positions at all


def test_targets_are_the_inputs_shifted_by_exactly_one():
    """The entire training signal. An off-by-one here teaches the model to copy its input.

    That bug reaches a suspiciously low loss and generates nothing but repetition, so it is worth
    asserting on the actual arrays rather than trusting the slicing.
    """
    dataset = Dataset(np.arange(200), train_fraction=1.0)
    x, y = dataset.batch(np.random.default_rng(0), batch_size=4, block_size=8)
    assert x.shape == y.shape == (4, 8)
    assert np.array_equal(y[:, :-1], x[:, 1:])
    assert np.array_equal(y, x + 1)  # true for this synthetic ramp


def test_batches_stay_inside_the_stream():
    """The window start is bounded so the shifted target never runs off the end."""
    dataset = Dataset(np.arange(30), train_fraction=1.0)
    for _ in range(50):
        x, y = dataset.batch(np.random.default_rng(None), 4, 8)
        assert x.max() < 30 and y.max() < 30


def test_batching_is_reproducible_from_a_generator():
    dataset = Dataset(np.arange(200), train_fraction=1.0)
    first = dataset.batch(np.random.default_rng(3), 4, 8)
    second = dataset.batch(np.random.default_rng(3), 4, 8)
    assert np.array_equal(first[0], second[0])


def test_dataset_validation():
    with pytest.raises(ValueError, match="1-D"):
        Dataset(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="train_fraction"):
        Dataset(np.arange(10), train_fraction=0.0)
    with pytest.raises(ValueError, match="need at least"):
        Dataset(np.arange(10), train_fraction=1.0).batch(np.random.default_rng(0), 2, 20)


# -- the diagnostics ---------------------------------------------------------------------


def test_uniform_loss_is_log_vocab_size():
    assert uniform_loss(256) == pytest.approx(np.log(256))


def test_initial_loss_report_accepts_a_healthy_model_and_names_the_failure_modes():
    model = demo.tiny_model().eval()
    x, y = demo.random_batch(model.config)
    report = initial_loss_report(model, x, y)
    assert "as expected" in report
    assert "SUSPICIOUS" not in report


def test_initial_loss_report_flags_an_oversized_initialisation():
    """The diagnostic has to be able to fail, or it is decoration."""
    model = demo.tiny_model().eval()
    model.tok_emb.params["W"] *= 200.0  # confident and wrong at step 0
    x, y = demo.random_batch(model.config)
    assert "SUSPICIOUS" in initial_loss_report(model, x, y)


def test_a_correct_model_can_memorise_one_batch():
    """The sharpest correctness test available for a model implementation.

    These are random token sequences with no structure at all, so the only way to fit them is to
    memorise -- which a model with tens of thousands of parameters certainly can. If this fails,
    something is broken and no amount of data or patience will fix it. Passing exercises the whole
    chain at once: attention, both LayerNorms, GELU, the tied embedding and AdamW.
    """
    model = demo.tiny_model(n_embd=32)
    x, y = demo.random_batch(model.config, batch_size=2, seed=5)

    report = overfit_batch(model, x, y, steps=400, lr=3e-3)
    assert report.initial_loss == pytest.approx(uniform_loss(256), rel=0.1)
    assert report.final_loss < 0.1
    assert report.final_loss < report.initial_loss / 20


# -- the loop ----------------------------------------------------------------------------


def test_training_reduces_the_loss_on_the_demo_corpus():
    _, dataset = demo.demo_dataset()
    model = small_model()
    report = train(model, dataset, steps=150, batch_size=8, lr=3e-3, warmup_steps=15, eval_every=50)

    smoothed = report.smoothed(20)
    assert report.initial_loss == pytest.approx(uniform_loss(256), rel=0.1)
    assert smoothed[-1] < smoothed[0] - 1.0
    assert smoothed[-1] < uniform_loss(256)
    assert len(report.losses) == len(report.grad_norms) == len(report.learning_rates) == 150


def test_held_out_loss_is_recorded_and_also_beats_the_uniform_baseline():
    """The honest claim for a model this small: better than knowing nothing, on unseen text."""
    _, dataset = demo.demo_dataset()
    model = small_model()
    report = train(model, dataset, steps=150, batch_size=8, lr=3e-3, warmup_steps=15, eval_every=50)

    assert report.val_losses, "no held-out evaluation was recorded"
    _, final_val = report.val_losses[-1]
    assert final_val < uniform_loss(256)


def test_learning_rate_follows_the_schedule():
    _, dataset = demo.demo_dataset()
    model = small_model()
    report = train(model, dataset, steps=40, batch_size=4, lr=1e-3, warmup_steps=10)

    assert report.learning_rates[0] == pytest.approx(1e-4)      # first warmup step
    assert report.learning_rates[9] == pytest.approx(1e-3)      # peak at the end of warmup
    assert report.learning_rates[-1] < report.learning_rates[9] # then decaying


def test_gradient_norms_are_recorded_before_clipping():
    """Logging the post-clip norm would report the threshold back to you, which tells you nothing."""
    _, dataset = demo.demo_dataset()
    model = small_model()
    report = train(model, dataset, steps=20, batch_size=4, lr=1e-3, max_grad_norm=1e-6)
    assert max(report.grad_norms) > 1e-6


def test_training_is_reproducible_from_a_seed():
    _, dataset = demo.demo_dataset()
    first = train(small_model(), dataset, steps=20, batch_size=4, seed=11)
    second = train(small_model(), dataset, steps=20, batch_size=4, seed=11)
    assert first.losses == second.losses


def test_dropout_is_disabled_during_evaluation_and_the_mode_restored():
    config = ModelConfig(vocab_size=256, block_size=16, n_layer=1, n_head=2, n_embd=32, dropout=0.3)
    model = TransformerLM(config, seed=0)
    _, dataset = demo.demo_dataset()
    model.train()

    rng = np.random.default_rng(0)
    first = evaluate(model, dataset, np.random.default_rng(0), batch_size=4, n_batches=2)
    second = evaluate(model, dataset, np.random.default_rng(0), batch_size=4, n_batches=2)
    assert first == pytest.approx(second)  # deterministic, so dropout was off
    assert model.training, "evaluate must hand the training mode back"
    assert rng is not None


def test_smoothing_is_a_trailing_mean():
    _, dataset = demo.demo_dataset()
    report = train(small_model(), dataset, steps=12, batch_size=4)
    smoothed = report.smoothed(4)
    assert len(smoothed) == 12
    assert smoothed[0] == pytest.approx(report.losses[0])
    assert smoothed[5] == pytest.approx(float(np.mean(report.losses[2:6])))
