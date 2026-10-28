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


# ---- Task 2: GloVe and embedding matrix ----
import numpy as np

from src.glove_bilstm import build_embedding_matrix, load_glove


def _tiny_glove(tmp_path):
    p = tmp_path / "glove.txt"
    p.write_text(
        "good 1 2 3 4\n"
        "bad -1 -2 -3 -4\n"
        "broken line without numbers\n"
        "short 1 2 3\n",
        encoding="utf-8",
    )
    return str(p)


def test_load_glove_skips_malformed_and_wrong_dim(tmp_path):
    glove = load_glove(_tiny_glove(tmp_path), dim=4)
    assert set(glove) == {"good", "bad"}
    assert glove["good"].shape == (4,)


def test_embedding_matrix_rows(tmp_path):
    glove = load_glove(_tiny_glove(tmp_path), dim=4)
    vocab = {"<pad>": 0, "<unk>": 1, "good": 2, "zzz": 3}
    m = build_embedding_matrix(vocab, glove, dim=4, seed=42)
    assert m.shape == (4, 4)
    assert m.dtype == np.float32
    assert np.array_equal(m[2], np.array([1, 2, 3, 4], dtype=np.float32))
    assert not m[0].any()  # pad row is zeros
    assert m[3].any() and np.abs(m[3]).max() < 1.0  # OOV: small random vector


def test_embedding_matrix_deterministic(tmp_path):
    glove = load_glove(_tiny_glove(tmp_path), dim=4)
    vocab = {"<pad>": 0, "<unk>": 1, "zzz": 2}
    a = build_embedding_matrix(vocab, glove, dim=4, seed=7)
    b = build_embedding_matrix(vocab, glove, dim=4, seed=7)
    assert np.array_equal(a, b)


# ---- Task 3: BiLSTM model ----
import torch

from src.glove_bilstm import BiLSTMClassifier


def _matrix(vocab_size=20, dim=8):
    rng = np.random.default_rng(0)
    m = rng.normal(size=(vocab_size, dim)).astype(np.float32)
    m[0] = 0
    return m


@pytest.mark.parametrize("pooling", ["attention", "max", "last"])
@pytest.mark.parametrize("bidirectional", [True, False])
def test_forward_shape(pooling, bidirectional):
    model = BiLSTMClassifier(_matrix(), hidden=6, pooling=pooling, bidirectional=bidirectional)
    ids = torch.tensor([[2, 3, 4, 0, 0], [5, 6, 7, 8, 9]])
    assert model(ids).shape == (2, 2)


@pytest.mark.parametrize("pooling", ["attention", "max"])
def test_padding_does_not_change_logits(pooling):
    torch.manual_seed(0)
    model = BiLSTMClassifier(_matrix(), hidden=6, pooling=pooling).eval()
    short = torch.tensor([[2, 3, 4] + [0] * 7])
    long = torch.tensor([[2, 3, 4] + [0] * 17])
    with torch.no_grad():
        assert torch.allclose(model(short), model(long), atol=1e-5)


def test_frozen_embeddings_have_no_grad():
    frozen = BiLSTMClassifier(_matrix(), freeze_embeddings=True)
    tuned = BiLSTMClassifier(_matrix(), freeze_embeddings=False)
    assert not frozen.embedding.weight.requires_grad
    assert tuned.embedding.weight.requires_grad


@pytest.mark.parametrize("pooling", ["attention", "max", "last"])
def test_all_pad_row_no_nan(pooling):
    model = BiLSTMClassifier(_matrix(), hidden=6, pooling=pooling).eval()
    out = model(torch.zeros(2, 5, dtype=torch.long))
    assert torch.isfinite(out).all()
