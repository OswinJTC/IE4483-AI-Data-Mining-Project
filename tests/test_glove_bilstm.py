import json

import pytest

from src.glove_bilstm import build_vocab, encode, load_split, tokenize


def test_tokenize_keeps_negation_and_punct():
    assert tokenize("Not good!") == ["not", "good", "!"]


def test_build_vocab_specials_and_min_freq():
    vocab = build_vocab(["good good movie", "good film"], min_freq=2)
    assert vocab["<pad>"] == 0
    assert vocab["<unk>"] == 1
    assert "good" in vocab
    assert "movie" not in vocab  # seen once


def test_encode_pads_and_truncates():
    vocab = build_vocab(["a b c d e f"], min_freq=1)
    assert len(encode("a b", vocab, 5)) == 5
    assert encode("a b", vocab, 5)[2:] == [0, 0, 0]
    long = encode("a b c d e f a b c d", vocab, 4)
    assert len(long) == 4


def test_encode_empty_and_all_oov():
    vocab = build_vocab(["a b"], min_freq=1)
    assert encode("", vocab, 4) == [0, 0, 0, 0]
    assert encode("zzz yyy", vocab, 4) == [1, 1, 0, 0]


def test_load_split_returns_texts_and_int_labels(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"reviews": "good", "sentiments": 1}, {"reviews": "bad", "sentiments": 0}]))
    texts, labels = load_split(str(p))
    assert texts == ["good", "bad"]
    assert labels == [1, 0]


def test_load_split_without_labels_raises(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"reviews": "good"}]))
    with pytest.raises(KeyError):
        load_split(str(p))
