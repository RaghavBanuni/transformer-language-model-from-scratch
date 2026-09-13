"""Demonstrations. Each subcommand establishes one fact about the implementation.

``gradcheck``  every parameter's analytic gradient against central finite differences
``causal``     that a future token cannot influence an earlier position's logits
``init``       that the loss at initialisation equals ln(V), as it must
``overfit``    that the model can memorise one batch -- the sharpest correctness test there is
``optim``      AdamW's decoupled decay and scale invariance, as exact properties
``tokenizer``  byte-level BPE: compression, round-trips, and no unknown tokens
``sample``     decoding knobs, and that the KV cache reproduces full recomputation
``train``      a short training run on the demo corpus, then generation from it

Run ``python -m minigpt.cli <command>``. Everything is CPU-only and deterministic.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import demo
from .gradcheck import check_module
from .model import ModelConfig, TransformerLM
from .optim import AdamW, cosine_schedule_with_warmup
from .sampling import generate, sample_from_logits, top_k_filter, top_p_filter
from .tokenizer import BPETokenizer
from .train import Dataset, initial_loss_report, overfit_batch, train, uniform_loss

RULE = "=" * 78


def _heading(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def cmd_gradcheck(args) -> None:
    """Check every analytic gradient against central finite differences."""
    _heading("Gradient check: analytic backward vs central finite differences")
    model = demo.tiny_model()
    x, y = demo.random_batch(model.config)

    model.zero_grad()
    loss = model.loss_and_backward(x, y)
    print(f"{model.n_parameters():,} parameters, loss {loss:.6f}\n")

    results = check_module(model, lambda: model.loss(x, y), n_samples=6, tolerance=1e-6)
    for result in results:
        print(result)

    worst = max(results, key=lambda r: r.max_relative_error)
    print(
        f"\nworst relative error {worst.max_relative_error:.3e} on {worst.name}\n"
        "Anything under 1e-6 means the derivation is right. Around 1e-2 means a missing term.\n"
        "Exactly 1.0 means one of the two numbers is zero -- almost always a gradient that never\n"
        "got accumulated, which is the failure mode weight tying invites."
    )


def cmd_causal(args) -> None:
    """Show that the causal mask actually holds."""
    _heading("Causality: a future token cannot change an earlier position")
    model = demo.tiny_model().eval()
    rng = np.random.default_rng(3)
    ids = rng.integers(0, model.config.vocab_size, size=(1, model.config.block_size))

    baseline = model.forward(ids)
    modified = ids.copy()
    last = model.config.block_size - 1
    modified[0, last] = (modified[0, last] + 137) % model.config.vocab_size
    perturbed = model.forward(modified)

    earlier = float(np.abs(baseline[:, :last, :] - perturbed[:, :last, :]).max())
    at_change = float(np.abs(baseline[:, last, :] - perturbed[:, last, :]).max())
    print(f"changed the token at position {last}")
    print(f"  max logit change at positions 0..{last - 1}: {earlier:.3e}")
    print(f"  max logit change at position {last}:        {at_change:.3e}")
    print(
        "\nThe first number is exactly zero, not merely small. That is what makes training on all\n"
        "positions at once legitimate: each position is a genuine prediction of the next token\n"
        "rather than a lookup of it. A leaky mask shows up as a beautiful training loss and\n"
        "generation that falls apart, because at inference the future really is absent."
    )

    mask_row_sums = np.tril(np.ones((5, 5))).sum(axis=1)
    print(f"\nvisible positions per row for T=5: {mask_row_sums.astype(int).tolist()}")
    print("Position 0 attends to itself alone, which is why its prediction is the prior.")


def cmd_init(args) -> None:
    """Compare the loss at initialisation against ln(V)."""
    _heading("The first number to check: loss at initialisation")
    for vocab_size in (64, 256, 1024):
        config = demo.tiny_config(vocab_size=vocab_size)
        model = TransformerLM(config, seed=0).eval()
        x, y = demo.random_batch(config)
        print(initial_loss_report(model, x, y))
    print(
        "\nA model that has learnt nothing should be uniform over the vocabulary, and the\n"
        "cross-entropy of the uniform distribution is exactly ln(V). One forward pass rules out\n"
        "a broken loss, a broken initialisation and a leaky mask -- before any training time is\n"
        "spent chasing a hyperparameter that was never the problem."
    )


def cmd_overfit(args) -> None:
    """Memorise a single batch, which a correct implementation must be able to do."""
    _heading("Overfitting one batch: the sharpest correctness test available")
    model = demo.tiny_model(n_embd=32)
    x, y = demo.random_batch(model.config, batch_size=2, seed=5)

    print(initial_loss_report(model, x, y))
    started = time.perf_counter()
    report = overfit_batch(model, x, y, steps=400, lr=3e-3)
    elapsed = time.perf_counter() - started

    print(f"\n{'step':>6}  {'loss':>10}")
    for step in (0, 25, 50, 100, 200, 399):
        print(f"{step:>6}  {report.losses[step]:>10.5f}")
    print(
        f"\nln(V) = {uniform_loss(model.config.vocab_size):.3f} at the start, "
        f"{report.final_loss:.5f} after 400 steps ({elapsed:.1f}s)."
    )
    print(
        "These are random token sequences with no structure whatsoever, so the only way to fit\n"
        "them is to memorise -- which is the point. A model that cannot memorise two sequences of\n"
        "eight tokens has a bug, and no amount of data will fix it. Reaching a loss near zero\n"
        "exercises the whole chain: attention, both norms, GELU, tied embeddings, AdamW."
    )


def cmd_optim(args) -> None:
    """AdamW's two properties that distinguish it from Adam plus L2."""
    _heading("Decoupled weight decay: exact geometric shrinkage")
    weights = {"W": np.array([[1.0, -2.0]])}
    zero_grads = {"W": np.zeros((1, 2))}
    optimizer = AdamW(weights, lr=0.1, weight_decay=0.1)
    for step in range(4):
        optimizer.step(zero_grads)
        expected = np.array([[1.0, -2.0]]) * (1.0 - 0.1 * 0.1) ** (step + 1)
        print(f"  step {step + 1}: {weights['W'].round(6).tolist()}  expected {expected.round(6).tolist()}")
    print(
        "\nWith zero gradients the update is exactly p*(1 - lr*wd)^t. Adam with L2 folded into the\n"
        "gradient cannot do this: the decay term would be divided by sqrt(v), so how strongly a\n"
        "weight is regularised would depend on its own gradient history. That is the bug AdamW\n"
        "fixes, and it is a property a test can pin rather than a preference."
    )

    _heading("Bias correction: the first step is lr-sized whatever the gradient scale")
    for gradient_scale in (1e-6, 1.0, 1e3):
        parameter = {"w": np.zeros(1)}
        optimizer = AdamW(parameter, lr=0.01)
        optimizer.step({"w": np.array([gradient_scale])})
        print(f"  gradient {gradient_scale:>8.0e} -> first step {abs(float(parameter['w'])):.6f}")
    print(
        "\nAll three are lr, because mhat/sqrt(vhat) is +/-1 for a constant gradient. Without bias\n"
        "correction the first step would be roughly 32x too large at beta2=0.999, since v starts at\n"
        "zero and is a thousand times too small on step one."
    )

    _heading("Schedule: linear warmup then cosine decay")
    total = 100
    for step in (0, 4, 9, 10, 25, 50, 75, 99):
        print(f"  step {step:>3}: lr {cosine_schedule_with_warmup(step, total, 1e-3, 10):.3e}")
    print(
        "\nWarmup exists because Adam's variance estimate is meaningless for the first few steps;\n"
        "full-size steps then move in a direction the optimiser has no information about."
    )


def cmd_tokenizer(args) -> None:
    """Byte-level BPE: compression, round-trips, no unknown tokens."""
    _heading("Byte-level BPE")
    text = demo.DEMO_TEXT
    for vocab_size in (256, 300, 400, 512):
        tokenizer = BPETokenizer.train(text, vocab_size)
        n_tokens = len(tokenizer.encode(text))
        print(
            f"  vocab {tokenizer.vocab_size:>4}  tokens {n_tokens:>6}  "
            f"{tokenizer.compression_ratio(text):.2f} bytes/token"
        )

    tokenizer = BPETokenizer.train(text, 512)
    longest = sorted(tokenizer.vocab.values(), key=len, reverse=True)[:8]
    print(f"\nlongest learned tokens: {[token.decode('utf-8', errors='replace') for token in longest]}")
    print("Leading spaces stay attached to words, and no token spans a word boundary.")

    _heading("No input is out of vocabulary")
    for sample in [
        "the gradient of the loss",
        "a word never seen in training: zygomorphic",
        "unicode: \u4f60\u597d\u4e16\u754c and \u00e9\u00e0\u00fc",
        "emoji: \U0001f9ee\U0001f4c9",
        "",
    ]:
        ids = tokenizer.encode(sample)
        restored = tokenizer.decode(ids)
        status = "round-trips" if restored == sample else "LOST DATA"
        print(f"  {status}  {len(ids):>3} tokens  {sample[:40]!r}")
    print(
        "\nEvery one of these round-trips exactly, including text no merge was ever learned for,\n"
        "because the base vocabulary is the 256 byte values. There is no <UNK> token to reach for,\n"
        "which is the entire argument for working at byte level."
    )


def cmd_sample(args) -> None:
    """Decoding knobs, and KV-cache equivalence."""
    _heading("Decoding: what each knob does to the same distribution")
    logits = np.log(np.array([[0.40, 0.25, 0.20, 0.10, 0.04, 0.01]]))
    print("  base distribution:      [0.40 0.25 0.20 0.10 0.04 0.01]")
    for temperature in (0.5, 1.0, 2.0):
        from .layers import softmax

        probabilities = softmax(logits / temperature, axis=-1)[0]
        print(f"  temperature {temperature:<4}        {np.round(probabilities, 3).tolist()}")
    print("  Below 1 sharpens, above 1 flattens. Applied to logits, not to probabilities.")

    from .layers import softmax

    print(f"\n  top-k=2 keeps:          {np.round(softmax(top_k_filter(logits, 2), -1)[0], 3).tolist()}")
    print(f"  top-p=0.85 keeps:       {np.round(softmax(top_p_filter(logits, 0.85), -1)[0], 3).tolist()}")
    print(
        "  Top-k always keeps two, whatever the shape. Top-p keeps as many as it takes to reach\n"
        "  the mass, which is why it adapts to a peaked context and a flat one differently."
    )

    _heading("KV cache: same tokens, less work")
    model = demo.tiny_model(n_embd=32).eval()
    prompt = np.array([[7, 11, 13]])

    cached = generate(model, prompt, 4, np.random.default_rng(0), temperature=0.0, use_cache=True)
    uncached = generate(model, prompt, 4, np.random.default_rng(0), temperature=0.0, use_cache=False)
    print(f"  cached:   {cached.tolist()}")
    print(f"  uncached: {uncached.tolist()}")
    print(f"  identical: {np.array_equal(cached, uncached)}")
    print(
        "\n  Two independent code paths that must agree. The cache reuses the keys and values of\n"
        "  every previous token, so a new token costs one attention over the prefix instead of a\n"
        "  full re-encode: O(T) per token rather than O(T^2) for the sequence."
    )


def cmd_train(args) -> None:
    """A short training run on the demo corpus, then generation."""
    _heading("Training on a small, highly structured corpus")
    tokenizer, dataset = demo.demo_dataset()
    config = ModelConfig(vocab_size=256, block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0)
    model = TransformerLM(config, seed=0)
    print(f"{model.n_parameters():,} parameters, {len(dataset.tokens):,} training bytes")
    print(f"uniform-baseline loss ln(256) = {uniform_loss(256):.3f}")

    started = time.perf_counter()
    report = train(
        model, dataset, steps=args.steps, batch_size=8, lr=3e-3, warmup_steps=20, eval_every=50
    )
    elapsed = time.perf_counter() - started

    smoothed = report.smoothed(10)
    print(f"\n{'step':>6}  {'train (smoothed)':>18}")
    for step in sorted({0, args.steps // 4, args.steps // 2, args.steps - 1}):
        print(f"{step:>6}  {smoothed[step]:>18.4f}")
    for step, value in report.val_losses:
        print(f"  held-out loss at step {step:>4}: {value:.4f}")
    print(f"\n{args.steps} steps in {elapsed:.1f}s on CPU, in NumPy.")

    prompt = "the gradient"
    ids = np.array([tokenizer.encode(prompt)])
    out = generate(model, ids, 40, np.random.default_rng(0), temperature=0.8, top_k=20)
    print(f"\nprompt:     {prompt!r}")
    print(f"generated:  {tokenizer.decode(out[0])!r}")
    print(
        "\nA few thousand parameters on four kilobytes of text is not a language model; it is a\n"
        "demonstration that the gradients and the optimiser work. The held-out loss falling below\n"
        "ln(256) is the honest claim here, and the sample is worth exactly as much as its size\n"
        "suggests."
    )


COMMANDS = {
    "gradcheck": cmd_gradcheck,
    "causal": cmd_causal,
    "init": cmd_init,
    "overfit": cmd_overfit,
    "optim": cmd_optim,
    "tokenizer": cmd_tokenizer,
    "sample": cmd_sample,
    "train": cmd_train,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="minigpt", description="Demonstrations for the pure-NumPy transformer."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in COMMANDS.items():
        help_text = (handler.__doc__ or name).strip().splitlines()[0]
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.set_defaults(handler=handler)
        if name == "train":
            subparser.add_argument("--steps", type=int, default=300, help="training steps")

    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
