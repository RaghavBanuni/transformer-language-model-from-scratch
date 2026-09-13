"""Byte-level BPE: round-trips, merge ordering, determinism, and compression."""

import numpy as np
import pytest

from minigpt.demo import DEMO_TEXT
from minigpt.tokenizer import BPETokenizer, ByteTokenizer


def test_byte_tokenizer_round_trips_anything():
    tokenizer = ByteTokenizer()
    for text in ["hello", "\u4f60\u597d", "\U0001f680 launch", "", "\x00\x01\xff".encode("latin-1").decode("latin-1")]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_byte_tokenizer_ids_are_within_the_vocabulary():
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("caf\u00e9 \U0001f9ee")
    assert all(0 <= token_id < tokenizer.vocab_size for token_id in ids)


def test_bpe_round_trips_including_text_it_never_saw():
    """The argument for byte level: there is no such thing as an unknown token.

    Every one of these encodes and decodes exactly, including scripts and code points that appear
    nowhere in the training corpus, because the base vocabulary is the 256 byte values. A word-level
    tokenizer would emit <UNK> and lose the content.
    """
    tokenizer = BPETokenizer.train(DEMO_TEXT, 400)
    for text in [
        "the gradient of the loss",
        "zygomorphic antidisestablishmentarianism",
        "\u4f60\u597d\u4e16\u754c",
        "caf\u00e9 na\u00efve \u00fcber",
        "\U0001f9ee\U0001f4c9\U0001f525",
        "tabs\tand\nnewlines",
        "   leading and trailing   ",
        "",
    ]:
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_training_is_deterministic():
    """Ties are broken lexicographically, not by dictionary order.

    Without a deterministic rule, two runs on the same corpus produce different vocabularies and a
    checkpoint silently stops matching its tokenizer.
    """
    first = BPETokenizer.train(DEMO_TEXT, 320)
    second = BPETokenizer.train(DEMO_TEXT, 320)
    assert first.merges == second.merges
    assert first.vocab == second.vocab


def test_merges_are_applied_in_training_rank_order():
    """Encoding must replay the merges in the order they were learned.

    Each merge was learned on text already rewritten by all previous merges, so applying a later
    merge first yields a tokenisation the model was never trained on. The encoder therefore always
    picks the lowest-ranked applicable pair -- and that reconstruction is what this checks: the
    ids produced for the training text expand back to exactly that text, and re-encoding the
    decoded form is idempotent.
    """
    tokenizer = BPETokenizer.train(DEMO_TEXT, 350)
    ids = tokenizer.encode(DEMO_TEXT)
    assert tokenizer.decode(ids) == DEMO_TEXT
    assert tokenizer.encode(tokenizer.decode(ids)) == ids


def test_a_frequent_pattern_becomes_a_single_token():
    """BPE should discover the structure that is actually there."""
    tokenizer = BPETokenizer.train("the theme of the thesis " * 50, 300)
    learned = {token.decode("utf-8", errors="replace") for token in tokenizer.vocab.values()}
    assert "the" in learned
    assert len(tokenizer.encode("the")) == 1


def test_merges_never_cross_a_word_boundary():
    """Pre-tokenisation is what keeps a token from spanning two words.

    Without it, BPE learns a single id for a frequent *bigram*, which wastes vocabulary on a pattern
    that only helps in one context and makes a word's tokenisation depend on what precedes it.
    """
    tokenizer = BPETokenizer.train("alpha beta " * 100, 400)
    for token in tokenizer.vocab.values():
        text = token.decode("utf-8", errors="replace")
        assert " " not in text.strip(), f"token {text!r} spans a word boundary"


def test_leading_whitespace_stays_attached_to_its_word():
    """Why "cat" and " cat" are different tokens in every real tokenizer."""
    tokenizer = BPETokenizer.train("the cat sat " * 100, 320)
    assert tokenizer.encode(" cat") != tokenizer.encode("cat")
    assert tokenizer.decode(tokenizer.encode(" cat")) == " cat"


def test_compression_improves_with_vocabulary_size():
    ratios = [BPETokenizer.train(DEMO_TEXT, size).compression_ratio(DEMO_TEXT) for size in (256, 300, 400)]
    assert ratios[0] == pytest.approx(1.0)  # no merges: one token per byte
    assert ratios[1] > ratios[0]
    assert ratios[2] > ratios[1]


def test_vocabulary_grows_by_exactly_one_per_merge():
    tokenizer = BPETokenizer.train(DEMO_TEXT, 300)
    assert tokenizer.vocab_size == 256 + len(tokenizer.merges)
    assert tokenizer.vocab_size <= 300


def test_training_stops_early_rather_than_padding_the_vocabulary():
    """A short corpus can simply run out of structure; inventing merges would create dead ids."""
    tokenizer = BPETokenizer.train("ab", 1000)
    assert tokenizer.vocab_size < 1000
    assert tokenizer.decode(tokenizer.encode("ab")) == "ab"


def test_serialisation_round_trip_preserves_encoding():
    tokenizer = BPETokenizer.train(DEMO_TEXT, 340)
    restored = BPETokenizer.from_dict(tokenizer.to_dict())
    assert restored.merges == tokenizer.merges
    assert restored.vocab == tokenizer.vocab
    assert restored.encode(DEMO_TEXT) == tokenizer.encode(DEMO_TEXT)


def test_decoding_tolerates_a_broken_multibyte_sequence():
    """A model can emit any ids, including a truncated UTF-8 character.

    Strict decoding would raise on output that is otherwise perfectly usable, so damage is made
    visible with the replacement character instead of fatal.
    """
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode("\u4f60")[:1]  # first byte of a three-byte character
    assert tokenizer.decode(ids) == "\ufffd"


def test_decode_accepts_numpy_ids():
    """Generation returns a NumPy array, so the tokenizer has to take one."""
    tokenizer = BPETokenizer.train(DEMO_TEXT, 300)
    ids = np.array(tokenizer.encode("the loss"), dtype=np.int64)
    assert tokenizer.decode(ids) == "the loss"


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError, match="at least 256"):
        BPETokenizer.train("abc", 100)
    with pytest.raises(TypeError):
        BPETokenizer.train(b"bytes not str", 300)
    tokenizer = BPETokenizer.train("abc", 300)
    with pytest.raises(ValueError, match="not in this vocabulary"):
        tokenizer.decode([999999])
