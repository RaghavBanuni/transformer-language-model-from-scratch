"""Decoding: temperature, top-k, nucleus, and why the order of operations matters.

Sampling is where a model's quality is most often misjudged, because the same weights produce
incoherent or repetitive text depending on three numbers. Each knob here does something specific.

**Temperature** divides the logits before the softmax. Below 1 it sharpens the distribution,
above 1 it flattens it. It is applied to *logits*, not to probabilities: dividing probabilities
and renormalising is a different function, and a much less useful one, because temperature on
logits is monotone in log-odds and so preserves the model's relative confidences.

**Top-k** keeps the k most likely tokens and renormalises. It answers the failure mode where the
tail -- tens of thousands of tokens each with probability 1e-6 -- collectively holds enough mass
that a long sample eventually draws from it, and one absurd token derails everything after it.

**Nucleus (top-p)** keeps the smallest set whose cumulative probability reaches p. The argument
against fixed k is that the right cut varies by context: after "the capital of France is" the
distribution is nearly a point mass and k=50 admits 49 wrong answers, while mid-sentence it is
genuinely broad and k=50 truncates real options. Top-p adapts to the shape (Holtzman et al., 2020).

Order of operations: temperature first, then top-k, then top-p. Truncation after temperature is
not the same as before it -- temperature changes the cumulative mass, so top-p on pre-temperature
probabilities would cut a different set. Doing it in the other order is a real and silent
difference in behaviour, so the order is fixed here rather than left to the caller.

Greedy decoding is ``temperature=0``, handled as an explicit branch rather than by dividing by
something tiny, which would overflow.
"""

from __future__ import annotations

import numpy as np

from .layers import softmax


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive; use greedy decoding for 0")
    return logits / temperature


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the ``k`` largest logits per row, mask the rest with ``-inf``.

    ``-inf`` rather than a large negative number so the masked tokens have probability exactly
    zero; "almost zero" is still sampleable given enough draws.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= logits.shape[-1]:
        return logits
    kth_largest = np.partition(logits, -k, axis=-1)[..., -k, None]
    return np.where(logits < kth_largest, -np.inf, logits)


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Keep the smallest set of tokens whose cumulative probability reaches ``p``.

    The token that crosses the threshold is *kept*, which is what makes the nucleus always
    non-empty: with a distribution whose top token already exceeds p, the set is exactly that one
    token. Shifting the comparison by one is the standard off-by-one here and produces an empty
    nucleus, i.e. a crash or a uniform sample over nothing.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    if p == 1.0:
        return logits

    probabilities = softmax(logits, axis=-1)
    order = np.argsort(-probabilities, axis=-1)
    sorted_probabilities = np.take_along_axis(probabilities, order, axis=-1)
    cumulative = np.cumsum(sorted_probabilities, axis=-1)

    # Drop everything strictly after the first index where the cumulative mass reaches p.
    keep_sorted = cumulative - sorted_probabilities < p
    keep = np.empty_like(keep_sorted)
    np.put_along_axis(keep, order, keep_sorted, axis=-1)
    return np.where(keep, logits, -np.inf)


def sample_from_logits(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> np.ndarray:
    """Draw one token id per row. ``temperature=0`` means greedy.

    Returns an array of shape ``(batch,)``.
    """
    logits = np.atleast_2d(np.asarray(logits, dtype=float))
    if temperature == 0.0:
        return np.argmax(logits, axis=-1)

    logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)

    probabilities = softmax(logits, axis=-1)
    # Sample per row. np.random.Generator.choice has no batched form, so this loop is explicit
    # rather than a vectorised trick that would be harder to read for a handful of rows.
    return np.array(
        [rng.choice(probabilities.shape[-1], p=row) for row in probabilities], dtype=np.int64
    )


def generate(
    model,
    prompt_ids: np.ndarray,
    max_new_tokens: int,
    rng: np.random.Generator | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = True,
) -> np.ndarray:
    """Autoregressive generation, with or without a KV cache.

    The model is switched to eval mode first. Generating with dropout active is a subtle and very
    common mistake: it produces a different (noisier) model than the one that was trained, and
    nothing errors.

    With ``use_cache=True`` each new token costs one attention over the accumulated keys, rather
    than a full re-encode of the whole prefix -- ``O(T)`` against ``O(T^2)`` for the sequence. The
    uncached path is kept because it is the reference: a test asserts the two produce identical
    token sequences from the same seed.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    was_training = model.training
    model.eval()
    try:
        ids = np.asarray(prompt_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.shape[1] == 0:
            raise ValueError("need at least one prompt token")
        block_size = model.config.block_size
        if ids.shape[1] + max_new_tokens > block_size:
            raise ValueError(
                f"prompt ({ids.shape[1]}) plus {max_new_tokens} new tokens exceeds the block "
                f"size ({block_size}); this model has no sliding window"
            )

        if not use_cache:
            for _ in range(max_new_tokens):
                logits = model.forward(ids)[:, -1, :]
                next_ids = sample_from_logits(logits, rng, temperature, top_k, top_p)
                ids = np.concatenate((ids, next_ids[:, None]), axis=1)
            return ids

        # Prime the cache one token at a time, so a single code path fills it.
        caches = None
        logits = None
        for position in range(ids.shape[1]):
            logits, caches = model.forward_step(ids[:, position : position + 1], position, caches)

        for step in range(max_new_tokens):
            next_ids = sample_from_logits(logits[:, -1, :], rng, temperature, top_k, top_p)
            ids = np.concatenate((ids, next_ids[:, None]), axis=1)
            logits, caches = model.forward_step(next_ids[:, None], ids.shape[1] - 1, caches)
        return ids
    finally:
        if was_training:
            model.train()
