"""The whole model: every gradient, the tied embedding, and the shape of the parameter count."""

import numpy as np
import pytest

from minigpt import demo
from minigpt.gradcheck import check_module
from minigpt.layers import cross_entropy
from minigpt.model import Block, ModelConfig, TransformerLM
from minigpt.train import uniform_loss


def test_every_model_gradient_matches_finite_differences():
    """The end-to-end check, and the one that catches the weight-tying bug.

    ``tok_emb.W`` receives gradient from two places: the embedding lookup on the way in and the
    output projection on the way out. If :class:`minigpt.layers.Module` assigned instead of
    accumulating, this parameter's gradient would be wrong while every other parameter stayed
    correct -- and the model would still train, just worse. Nothing but a numerical check finds that.
    """
    model = demo.tiny_model()
    x, y = demo.random_batch(model.config)

    model.zero_grad()
    model.loss_and_backward(x, y)

    results = check_module(model, lambda: model.loss(x, y), n_samples=4, tolerance=1e-6)
    assert results, "no parameters were checked"
    failures = [str(result) for result in results if not result.passed]
    assert not failures, "\n".join(failures)


def test_the_tied_embedding_is_actually_checked():
    """Guard against the check above passing vacuously."""
    names = [name for name, _ in demo.tiny_model().named_parameters()]
    assert "tok_emb.W" in names
    assert "pos_emb" in names


def test_the_tied_gradient_is_the_sum_of_both_paths():
    """Decompose the embedding gradient and show the lookup contribution is really there.

    The output-projection term is ``dlogits^T @ h``, which is dense across the whole vocabulary. The
    lookup term is a scatter-add, so it is non-zero only on the rows for tokens that actually
    appeared in the batch. Subtracting the first from the total must therefore leave a matrix with
    exactly that sparsity pattern -- which it cannot if one path overwrote the other.
    """
    model = demo.tiny_model()
    x, y = demo.random_batch(model.config, batch_size=2, seed=9)
    vocab, channels = model.config.vocab_size, model.config.n_embd

    logits = model.forward(x)
    _, dlogits = cross_entropy(logits.reshape(-1, vocab), np.asarray(y).reshape(-1))
    output_path = dlogits.T @ model._final_hidden.reshape(-1, channels)

    model.zero_grad()
    model.backward(dlogits.reshape(logits.shape))
    lookup_path = model.tok_emb.grads["W"] - output_path

    used = np.unique(np.asarray(x))
    unused = np.setdiff1d(np.arange(vocab), used)
    assert np.any(lookup_path[used] != 0.0), "the lookup path contributed nothing"
    assert np.allclose(lookup_path[unused], 0.0, atol=1e-12), "unused rows got lookup gradient"
    assert np.any(output_path != 0.0)


def test_loss_at_initialisation_is_log_vocab_size():
    """The cheapest diagnostic there is: one forward pass rules out several classes of bug.

    Far above ln(V) means the initialisation is too large -- the model starts confident and wrong.
    Far below means information is leaking, and the causal mask is the first suspect.
    """
    for vocab_size in (64, 256):
        config = demo.tiny_config(vocab_size=vocab_size)
        model = TransformerLM(config, seed=0).eval()
        x, y = demo.random_batch(config, batch_size=4)
        assert abs(model.loss(x, y) - uniform_loss(vocab_size)) < 0.1 * uniform_loss(vocab_size)


def test_parameter_count_matches_the_arithmetic():
    """Pinning the count proves the output projection is tied rather than merely similar.

    For V=256, C=16, T=8, L=2:
      embedding      256*16 = 4096
      positions        8*16 =  128
      per block:  two LayerNorms 2*(16+16) =   64
                  qkv           16*48 + 48 =  816
                  attn proj     16*16 + 16 =  272
                  mlp fc        16*64 + 64 = 1088
                  mlp proj      64*16 + 16 = 1040   -> 3280
      two blocks                                    = 6560
      final LayerNorm                                =   32
                                                       -----
                                                       10816
    An untied model would carry another 4096 parameters for a separate output head.
    """
    assert demo.tiny_model().n_parameters() == 10816


def test_forward_shapes_and_limits():
    model = demo.tiny_model()
    x, _ = demo.random_batch(model.config)
    logits = model.forward(x)
    assert logits.shape == (x.shape[0], x.shape[1], model.config.vocab_size)

    with pytest.raises(ValueError, match="exceeds block size"):
        model.forward(np.zeros((1, model.config.block_size + 1), dtype=int))
    with pytest.raises(ValueError, match="shape"):
        model.forward(np.zeros(4, dtype=int))


def test_a_short_sequence_only_touches_the_positions_it_used():
    """Position embeddings beyond the sequence length must receive no gradient."""
    model = demo.tiny_model()
    used = 3
    model.zero_grad()
    model.loss_and_backward(np.zeros((1, used), dtype=int), np.ones((1, used), dtype=int))
    assert np.any(model.grads["pos_emb"][:used] != 0.0)
    assert np.all(model.grads["pos_emb"][used:] == 0.0)


def test_no_parameter_is_left_without_gradient():
    """A dead path -- a forgotten residual add, a detached branch -- shows up as an all-zero grad."""
    model = demo.tiny_model()
    x, y = demo.random_batch(model.config)
    model.zero_grad()
    model.loss_and_backward(x, y)

    for name, gradient in model.named_gradients():
        assert np.any(gradient != 0.0), f"{name} received no gradient"


def test_gradient_reaches_the_first_block_undiminished():
    """What the pre-LN residual path is for.

    With a clean identity skip from the loss down to the embedding, the first block's gradients are
    the same order of magnitude as the last block's. A vanishing ratio here is the signature of a
    residual path that has been normalised or rescaled somewhere it should not have been.
    """
    config = demo.tiny_config(n_layer=4)
    model = TransformerLM(config, seed=0)
    x, y = demo.random_batch(config)
    model.zero_grad()
    model.loss_and_backward(x, y)

    def block_norm(index: int) -> float:
        block = model.children[f"block{index}"]
        return float(np.sqrt(sum(np.sum(g**2) for _, g in block.named_gradients())))

    first, last = block_norm(0), block_norm(config.n_layer - 1)
    assert first > 0.0 and last > 0.0
    assert 0.02 < first / last < 50.0


def test_zero_grad_clears_everything():
    model = demo.tiny_model()
    x, y = demo.random_batch(model.config)
    model.loss_and_backward(x, y)
    assert any(np.any(g != 0) for _, g in model.named_gradients())
    model.zero_grad()
    assert all(np.all(g == 0) for _, g in model.named_gradients())


def test_two_backward_passes_accumulate():
    """The runner is responsible for zeroing; the model must not do it implicitly."""
    model = demo.tiny_model()
    x, y = demo.random_batch(model.config)

    model.zero_grad()
    model.loss_and_backward(x, y)
    once = model.ln_f.grads["gamma"].copy()
    model.loss_and_backward(x, y)
    assert np.allclose(model.ln_f.grads["gamma"], 2 * once)


def test_eval_mode_propagates_and_makes_the_forward_deterministic():
    config = demo.tiny_config(dropout=0.2)
    model = TransformerLM(config, seed=0)
    x, _ = demo.random_batch(config)

    assert not np.allclose(model.forward(x), model.forward(x))  # dropout is live

    model.eval()
    assert model.drop.training is False
    assert model.children["block0"].mlp.dropout.training is False
    assert np.allclose(model.forward(x), model.forward(x))


def test_cached_generation_path_matches_the_full_forward():
    """Model level, not just attention level: this also exercises absolute positions.

    Reusing position 0 at every step is a plausible bug that produces fluent-looking output with no
    sense of order, and it is invisible by eye.
    """
    model = demo.tiny_model(n_embd=32).eval()
    ids = demo.random_batch(model.config, batch_size=1)[0]

    full = model.forward(ids)

    caches = None
    stepped = []
    for position in range(ids.shape[1]):
        logits, caches = model.forward_step(ids[:, position : position + 1], position, caches)
        stepped.append(logits)

    assert np.allclose(full, np.concatenate(stepped, axis=1), atol=1e-9)


def test_forward_step_rejects_a_position_past_the_block():
    model = demo.tiny_model().eval()
    with pytest.raises(ValueError):
        model.forward_step(np.zeros((1, 1), dtype=int), model.config.block_size)


def test_config_validation():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(vocab_size=10, n_embd=10, n_head=3)
    with pytest.raises(ValueError, match="positive"):
        ModelConfig(vocab_size=0)
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(vocab_size=10, dropout=1.0)


def test_block_backward_keeps_the_identity_path():
    """``dx = dout + branch`` -- forgetting the first term stops the deepest layers learning.

    With both branch projections zeroed the block is exactly the identity, so its input gradient
    must be exactly the incoming gradient.
    """
    config = demo.tiny_config()
    block = Block(config, np.random.default_rng(0))
    for projection in (block.attn.proj, block.mlp.proj):
        projection.params["W"][:] = 0.0
        projection.params["b"][:] = 0.0

    x = np.random.default_rng(1).normal(size=(1, 4, config.n_embd))
    assert np.allclose(block.forward(x), x)

    dout = np.random.default_rng(2).normal(size=x.shape)
    assert np.allclose(block.backward(dout), dout)
