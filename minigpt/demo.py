"""Fixtures for the demonstrations and the tests.

The corpus is short, original, and highly structured on purpose. A byte-level model of a few
thousand parameters trained for a few hundred steps in NumPy cannot learn English; it can learn the
regularities of a small, repetitive text, which is exactly what is needed to show that the gradients
and the optimiser work. Claiming more than that from a demo of this size would be dishonest.
"""

from __future__ import annotations

import numpy as np

from .model import ModelConfig, TransformerLM
from .tokenizer import ByteTokenizer
from .train import Dataset

PARAGRAPH = (
    "the gradient of the loss flows backward through every layer. "
    "the mask keeps the past from seeing the future. "
    "the softmax turns scores into probabilities. "
    "the residual path carries the signal to the first layer. "
)

#: About four kilobytes: enough for a train/validation split, small enough to run in seconds.
DEMO_TEXT = PARAGRAPH * 16


def tiny_config(vocab_size: int = 256, **overrides) -> ModelConfig:
    """A model small enough to gradient-check exhaustively.

    Correctness of the chain rule does not depend on width, so the checks run on two layers of two
    heads and 16 channels. Checking a large model would cost thousands of forward passes and prove
    nothing further.
    """
    settings = dict(vocab_size=vocab_size, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    settings.update(overrides)
    return ModelConfig(**settings)


def tiny_model(seed: int = 0, **overrides) -> TransformerLM:
    return TransformerLM(tiny_config(**overrides), seed=seed)


def random_batch(
    config: ModelConfig, batch_size: int = 2, seed: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Random ids and targets, for gradient checks that need no semantics."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, config.vocab_size, size=(batch_size, config.block_size))
    y = rng.integers(0, config.vocab_size, size=(batch_size, config.block_size))
    return x, y


def demo_dataset(text: str = DEMO_TEXT) -> tuple[ByteTokenizer, Dataset]:
    tokenizer = ByteTokenizer()
    return tokenizer, Dataset(np.array(tokenizer.encode(text), dtype=np.int64), train_fraction=0.9)
