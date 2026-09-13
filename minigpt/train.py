"""Training: batching, the loop, and the two diagnostics worth running before anything else.

Diagnostic one: the loss at initialisation
------------------------------------------
An untrained model over ``V`` tokens should predict uniformly, giving a cross-entropy of exactly
``ln V``. For 256 byte tokens that is 5.545. If step 0 reports something far from it, training has
not started badly -- the model or the loss is *wrong*. Well above ``ln V`` means the initialisation
is too large (saturated softmax, confident and incorrect); well below means information is leaking,
and the usual culprit is a broken causal mask. This costs one forward pass and rules out a whole
class of bugs, so :func:`initial_loss_report` is the first thing the demos print.

Diagnostic two: overfit a tiny batch
------------------------------------
A correct model with a correct optimiser can drive the loss on a handful of sequences to almost
zero, because it has enough parameters to memorise them. If it cannot, something is broken and no
amount of data or patience will fix it. This is a far sharper test than "the loss went down", and
:func:`overfit_batch` runs it as an assertion rather than a plot.

The split
---------
Train and validation are contiguous halves, not random windows. With random windows of length ``T``
drawn from one stream, a validation window overlaps training windows almost surely, so the model has
seen the continuation it is being evaluated on. The resulting validation loss is a memorisation
score that looks like generalisation, and it is one of the easiest ways to fool yourself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import TransformerLM
from .optim import AdamW, clip_grad_norm, cosine_schedule_with_warmup


@dataclass
class Dataset:
    """A flat stream of token ids, split into contiguous train and validation parts."""

    tokens: np.ndarray
    train_fraction: float = 0.9

    def __post_init__(self) -> None:
        self.tokens = np.asarray(self.tokens, dtype=np.int64)
        if self.tokens.ndim != 1:
            raise ValueError("tokens must be a 1-D stream")
        if not 0.0 < self.train_fraction <= 1.0:
            raise ValueError("train_fraction must be in (0, 1]")
        split = int(len(self.tokens) * self.train_fraction)
        self.train = self.tokens[:split]
        self.val = self.tokens[split:]

    def batch(
        self, rng: np.random.Generator, batch_size: int, block_size: int, split: str = "train"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Random windows, with targets shifted one position left.

        The shift is the entire training signal: position ``t`` predicts the token at ``t+1``, so a
        window of length ``T`` needs ``T+1`` tokens available. Getting the off-by-one wrong here
        trains the model to predict the token it was just given, which reaches a suspiciously low
        loss and generates nothing but repetition.
        """
        stream = self.train if split == "train" else self.val
        if len(stream) < block_size + 1:
            raise ValueError(
                f"{split} split has {len(stream)} tokens, need at least {block_size + 1}"
            )
        starts = rng.integers(0, len(stream) - block_size, size=batch_size)
        x = np.stack([stream[start : start + block_size] for start in starts])
        y = np.stack([stream[start + 1 : start + block_size + 1] for start in starts])
        return x, y


@dataclass
class TrainReport:
    """What a run produced, for assertions rather than eyeballing."""

    losses: list[float] = field(default_factory=list)
    val_losses: list[tuple[int, float]] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)

    @property
    def initial_loss(self) -> float:
        return self.losses[0]

    @property
    def final_loss(self) -> float:
        return self.losses[-1]

    def smoothed(self, window: int = 10) -> list[float]:
        """Trailing mean. A single step's loss is dominated by which batch it drew."""
        out = []
        for index in range(len(self.losses)):
            start = max(0, index - window + 1)
            out.append(float(np.mean(self.losses[start : index + 1])))
        return out


def uniform_loss(vocab_size: int) -> float:
    """``ln V``: the cross-entropy of a model that knows nothing at all."""
    return float(np.log(vocab_size))


def initial_loss_report(model: TransformerLM, ids: np.ndarray, targets: np.ndarray) -> str:
    """Compare the loss at initialisation against ``ln V``, and say what a mismatch means."""
    expected = uniform_loss(model.config.vocab_size)
    observed = model.loss(ids, targets)
    ratio = observed / expected
    verdict = "as expected" if 0.9 <= ratio <= 1.1 else "SUSPICIOUS"
    note = ""
    if ratio > 1.1:
        note = " (initialisation too large? the model starts confident and wrong)"
    elif ratio < 0.9:
        note = " (information leaking? check the causal mask before anything else)"
    return (
        f"loss at initialisation {observed:.4f} vs ln({model.config.vocab_size}) = "
        f"{expected:.4f} -> {verdict}{note}"
    )


def train(
    model: TransformerLM,
    dataset: Dataset,
    steps: int,
    batch_size: int = 8,
    lr: float = 3e-3,
    weight_decay: float = 0.01,
    warmup_steps: int = 20,
    max_grad_norm: float = 1.0,
    eval_every: int = 0,
    eval_batches: int = 4,
    seed: int = 0,
    log_every: int = 0,
) -> TrainReport:
    """Train with AdamW, gradient clipping and a cosine schedule.

    The order inside the loop is not arbitrary: zero the gradients, forward and backward, clip,
    then step. Zeroing at the *start* rather than the end means an exception mid-loop cannot leave
    stale gradients to be applied on the next iteration, and clipping before the step is the only
    place it can do anything.
    """
    rng = np.random.default_rng(seed)
    parameters = dict(model.named_parameters())
    gradients = dict(model.named_gradients())
    optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay)
    report = TrainReport()

    model.train()
    for step in range(steps):
        x, y = dataset.batch(rng, batch_size, model.config.block_size, "train")

        model.zero_grad()
        loss = model.loss_and_backward(x, y)
        norm = clip_grad_norm(gradients, max_grad_norm)
        current_lr = cosine_schedule_with_warmup(step, steps, lr, warmup_steps)
        optimizer.step(gradients, lr=current_lr)

        report.losses.append(loss)
        report.grad_norms.append(norm)
        report.learning_rates.append(current_lr)

        if eval_every and (step + 1) % eval_every == 0 and len(dataset.val) > model.config.block_size:
            report.val_losses.append((step, evaluate(model, dataset, rng, batch_size, eval_batches)))
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"step {step:>5}  loss {loss:.4f}  |g| {norm:.3f}  lr {current_lr:.2e}")

    return report


def evaluate(
    model: TransformerLM,
    dataset: Dataset,
    rng: np.random.Generator,
    batch_size: int = 8,
    n_batches: int = 4,
    split: str = "val",
) -> float:
    """Mean loss over a few batches, with dropout off.

    Restoring the previous mode afterwards matters: leaving the model in eval mode after a
    mid-training evaluation silently disables dropout for the rest of the run.
    """
    was_training = model.training
    model.eval()
    try:
        losses = []
        for _ in range(n_batches):
            x, y = dataset.batch(rng, batch_size, model.config.block_size, split)
            losses.append(model.loss(x, y))
        return float(np.mean(losses))
    finally:
        if was_training:
            model.train()


def overfit_batch(
    model: TransformerLM,
    x: np.ndarray,
    y: np.ndarray,
    steps: int = 200,
    lr: float = 3e-3,
) -> TrainReport:
    """Drive the loss on one fixed batch towards zero.

    The sharpest correctness test available for a model implementation. No schedule, no clipping, no
    dropout: nothing that could mask a broken gradient by slowing everything down.
    """
    parameters = dict(model.named_parameters())
    gradients = dict(model.named_gradients())
    optimizer = AdamW(parameters, lr=lr, weight_decay=0.0)
    report = TrainReport()

    model.eval()  # dropout off: this is a determinism test, not a regularisation test
    for _ in range(steps):
        model.zero_grad()
        report.losses.append(model.loss_and_backward(x, y))
        optimizer.step(gradients)
    return report
