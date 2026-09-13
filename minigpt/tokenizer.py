"""Byte-level BPE: no unknown tokens, ever.

Why byte level
--------------
A word-level vocabulary has to answer "what about a word I have never seen?", and the usual answer
is an ``<UNK>`` token -- which throws away the information and makes the model's output unable to
spell anything new. A character-level vocabulary has no unknowns but produces sequences several
times longer, and attention cost grows with the square of length.

Byte-level BPE starts from the 256 possible byte values, so *every* string over any alphabet is
representable by construction. Emoji, Cyrillic, a Base64 blob, a corrupted download: all encodable,
none of them unknown. Merges then buy back the sequence length by learning that frequent byte pairs
deserve single ids.

The two rules that make it correct
----------------------------------
**Merges are applied in training order, not by frequency at encode time.** Each merge is learned on
the text as rewritten by all previous merges, so applying merge 300 before merge 12 produces a
tokenisation the model was never trained on. :meth:`BPETokenizer.encode` therefore repeatedly finds
the *lowest-ranked* applicable pair, which reconstructs the training order exactly.

**Ties are broken deterministically.** When two pairs are equally frequent, taking whichever the
hash order happens to yield makes the tokenizer irreproducible: the same corpus trains two different
vocabularies, and a checkpoint no longer matches its tokenizer. Here the tie goes to the
lexicographically smaller pair, and a test asserts that two training runs agree exactly.

Pre-tokenisation
----------------
Text is first split into chunks of ``optional whitespace + non-whitespace``, GPT-2 style, and merges
never cross a chunk boundary. Without that split, BPE happily learns a single token for a frequent
word *pair*, which wastes vocabulary on a bigram that only helps in one context and makes the
tokenisation of a word depend on what precedes it. Leading whitespace stays attached to its word,
which is why real tokenizers distinguish ``"cat"`` from ``" cat"``.

What this is not
----------------
The training loop is O(corpus) per merge, so it is fine for the megabyte-scale corpora in this
repository and inappropriate for gigabytes -- production implementations maintain incremental pair
counts and a priority queue. Correctness is identical; only the constant differs.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

CHUNK_PATTERN = re.compile(r"\s*\S+|\s+")


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` with ``new_id``.

    Non-overlapping matters: in ``[a, a, a]`` merging ``(a, a)`` yields ``[new, a]``, not
    ``[new, new]``. Scanning with an explicit index rather than a regex over the list is what keeps
    that unambiguous.
    """
    merged: list[int] = []
    index = 0
    while index < len(ids):
        if index + 1 < len(ids) and (ids[index], ids[index + 1]) == pair:
            merged.append(new_id)
            index += 2
        else:
            merged.append(ids[index])
            index += 1
    return merged


@dataclass
class BPETokenizer:
    """A trained byte-level BPE tokenizer.

    ``merges`` maps an ordered byte-id pair to the id it becomes; insertion order *is* the merge
    rank. ``vocab`` maps every id to the byte string it expands to.
    """

    merges: dict[tuple[int, int], int] = field(default_factory=dict)
    vocab: dict[int, bytes] = field(default_factory=lambda: {i: bytes([i]) for i in range(256)})

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # -- training ------------------------------------------------------------------------

    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        """Learn merges until the vocabulary reaches ``vocab_size``.

        Chunks are counted rather than listed, so a corpus with heavy repetition costs no more than
        its distinct vocabulary. Training stops early and silently if no pair remains -- a corpus
        can simply run out of structure before the requested size, and inventing merges to hit a
        target would produce ids the encoder can never emit.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256: the byte alphabet is not optional")
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        chunk_counts = Counter(CHUNK_PATTERN.findall(text))
        sequences: list[tuple[list[int], int]] = [
            (list(chunk.encode("utf-8")), count) for chunk, count in chunk_counts.items()
        ]

        merges: dict[tuple[int, int], int] = {}
        vocab: dict[int, bytes] = {index: bytes([index]) for index in range(256)}

        for new_id in range(256, vocab_size):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for ids, count in sequences:
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += count
            if not pair_counts:
                break

            # Highest count wins; ties go to the lexicographically smaller pair, so training is
            # reproducible rather than dependent on dictionary iteration order.
            best_pair = min(pair_counts, key=lambda pair: (-pair_counts[pair], pair))
            if pair_counts[best_pair] < 2:
                break  # a pair seen once buys nothing; stop rather than pad the vocabulary

            merges[best_pair] = new_id
            vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
            sequences = [(_merge_pair(ids, best_pair, new_id), count) for ids, count in sequences]
            if verbose:
                print(f"merge {new_id}: {best_pair} -> {vocab[new_id]!r} ({pair_counts[best_pair]}x)")

        return cls(merges=merges, vocab=vocab)

    # -- encoding and decoding -----------------------------------------------------------

    def encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pairs = zip(ids, ids[1:])
            best_pair = min(pairs, key=lambda pair: self.merges.get(pair, float("inf")))
            if best_pair not in self.merges:
                break
            ids = _merge_pair(ids, best_pair, self.merges[best_pair])
        return ids

    def encode(self, text: str) -> list[int]:
        """Text -> token ids. Never fails and never emits an unknown token."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        out: list[int] = []
        for chunk in CHUNK_PATTERN.findall(text):
            out.extend(self.encode_chunk(chunk))
        return out

    def decode(self, ids) -> str:
        """Token ids -> text.

        ``errors="replace"`` is required rather than defensive. A model can emit any id sequence,
        including one that splits a multi-byte UTF-8 character across a truncation boundary, and
        strict decoding would raise on output that is otherwise perfectly usable. The replacement
        character makes the damage visible instead of fatal.
        """
        pieces = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id not in self.vocab:
                raise ValueError(f"token id {token_id} is not in this vocabulary")
            pieces.append(self.vocab[token_id])
        return b"".join(pieces).decode("utf-8", errors="replace")

    # -- reporting and persistence -------------------------------------------------------

    def compression_ratio(self, text: str) -> float:
        """Bytes per token. Higher is better; 1.0 means the merges bought nothing."""
        n_bytes = len(text.encode("utf-8"))
        n_tokens = len(self.encode(text))
        return n_bytes / n_tokens if n_tokens else 0.0

    def to_dict(self) -> dict:
        """JSON-friendly form. Merge order is preserved as a list, because it is the rank."""
        return {
            "merges": [[list(pair), new_id] for pair, new_id in self.merges.items()],
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        merges = {(int(a), int(b)): int(new_id) for (a, b), new_id in data["merges"]}
        vocab: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
        for (first, second), new_id in merges.items():
            vocab[new_id] = vocab[first] + vocab[second]
        return cls(merges=merges, vocab=vocab)


class ByteTokenizer:
    """The degenerate case: one token per byte, no merges.

    Useful as a control. If a model trains with this and not with BPE, the bug is in the tokenizer,
    not the model -- and the demos use it to keep the vocabulary small enough that a NumPy model can
    be trained in seconds.
    """

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(token_id) & 0xFF for token_id in ids).decode("utf-8", errors="replace")
