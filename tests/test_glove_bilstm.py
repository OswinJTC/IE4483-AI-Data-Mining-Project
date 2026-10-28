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


# ---- Task 4: metrics and training ----
from src.glove_bilstm import compute_metrics, predict_proba, set_seed, train_model


def test_metrics_all_positive_prediction():
    y_true = [1, 1, 1, 0]
    m = compute_metrics(y_true, [1, 1, 1, 1], [0.9, 0.8, 0.7, 0.6])
    assert m["neg_f1"] == 0.0
    assert m["accuracy"] == 0.75
    assert all(np.isfinite(m[k]) for k in ("precision", "recall", "f1", "pr_auc"))


def test_metrics_perfect():
    m = compute_metrics([0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0.1, 0.2, 0.8, 0.9, 0.7])
    for k in ("accuracy", "precision", "recall", "f1", "neg_f1", "pr_auc"):
        assert m[k] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 3]]


def _toy_data(n=60, max_len=6, seed=0):
    """Class 1 reviews contain token 2, class 0 reviews contain token 3; separable."""
    rng = np.random.default_rng(seed)
    ids, y = [], []
    for i in range(n):
        label = i % 2
        row = [2 if label else 3] + list(rng.integers(4, 9, size=3)) + [0] * (max_len - 4)
        ids.append([int(t) for t in row])
        y.append(label)
    return ids, y


def _run_toy(seed=42, **kw):
    set_seed(seed)
    ids, y = _toy_data()
    model = BiLSTMClassifier(_matrix(vocab_size=9), hidden=8, pooling="attention", freeze_embeddings=False)
    res = train_model(model, ids, y, ids, y, lr=0.01, batch_size=16, max_epochs=6, patience=6,
                      seed=seed, device="cpu", **kw)
    return model, ids, y, res


def test_train_model_reduces_loss_and_is_deterministic():
    _, _, _, a = _run_toy()
    _, _, _, b = _run_toy()
    assert a["history"][-1]["train_loss"] < a["history"][0]["train_loss"]
    assert a["metrics"]["accuracy"] == b["metrics"]["accuracy"]
    assert a["history"] == b["history"]


def test_early_stopping_restores_best():
    model, ids, y, res = _run_toy()
    best = max(h["val_neg_f1"] for h in res["history"])
    assert 1 <= res["best_epoch"] <= len(res["history"])
    probs = predict_proba(model, ids, device="cpu")
    restored = compute_metrics(y, (probs >= 0.5).astype(int), probs)
    assert restored["neg_f1"] == best


def test_train_model_class_weighted_runs():
    _, _, _, res = _run_toy(class_weighted=True)
    assert 0.0 <= res["metrics"]["neg_f1"] <= 1.0


def test_best_epoch_weights_are_restored_when_best_is_not_last(monkeypatch):
    """Epoch 1 scores best, later epochs score worse: training must stop early and
    the final model must hold the epoch-1 weights, not the last epoch's."""
    import copy

    import src.glove_bilstm as mod

    set_seed(42)
    ids, y = _toy_data()
    model = BiLSTMClassifier(_matrix(vocab_size=9), hidden=8, freeze_embeddings=False)
    snapshots, calls = [], {"n": 0}
    real = mod.compute_metrics

    def scripted(y_true, y_pred, y_prob):
        m = real(y_true, y_pred, y_prob)
        calls["n"] += 1
        if calls["n"] <= 3:  # one call per epoch while training
            snapshots.append(copy.deepcopy(model.state_dict()))
            m["neg_f1"] = 0.9 if calls["n"] == 1 else 0.1
        return m

    monkeypatch.setattr(mod, "compute_metrics", scripted)
    res = train_model(model, ids, y, ids, y, lr=0.05, batch_size=16, max_epochs=10, patience=2,
                      seed=42, device="cpu")

    assert res["best_epoch"] == 1
    assert len(res["history"]) == 3  # stopped early: epoch 1 best, then 2 stale epochs
    for key, value in model.state_dict().items():
        assert torch.equal(value, snapshots[0][key]), f"{key} not restored to epoch 1"


def test_finetuning_does_not_mutate_callers_embedding_matrix():
    """Fine-tuned embeddings must be a private copy: reusing one matrix for many runs
    must start each run from the same pretrained vectors."""
    emb = _matrix(vocab_size=9)
    original = emb.copy()
    ids, y = _toy_data()
    model = BiLSTMClassifier(emb, hidden=4, freeze_embeddings=False)
    train_model(model, ids, y, ids, y, lr=0.05, batch_size=16, max_epochs=2, patience=2,
                seed=1, device="cpu")
    assert np.array_equal(emb, original)


# ---- Stage 2 work: length bucketing, single validation pass, progress logging ----
from src.glove_bilstm import _bucketed_batches


def test_bucketed_batches_cover_every_index_once():
    set_seed(0)
    lengths = torch.randint(1, 150, (1000,))
    batches = _bucketed_batches(lengths, batch_size=64)
    seen = torch.cat(batches).sort().values
    assert torch.equal(seen, torch.arange(1000))
    assert all(len(b) <= 64 for b in batches)


def test_bucketed_batches_group_similar_lengths():
    set_seed(0)
    lengths = torch.randint(1, 150, (2000,))
    batches = _bucketed_batches(lengths, batch_size=32)
    bucketed = np.mean([int(lengths[b].max() - lengths[b].min()) for b in batches])
    random_batches = torch.randperm(2000).split(32)
    plain = np.mean([int(lengths[b].max() - lengths[b].min()) for b in random_batches])
    assert bucketed < plain / 2


def test_predict_proba_keeps_order_and_matches_single_reviews():
    set_seed(0)
    model = BiLSTMClassifier(_matrix(), hidden=6, pooling="attention").eval()
    ids = [[2, 3, 4] + [0] * 7, [5] * 10, [6, 7] + [0] * 8, [8, 9, 2, 3, 4, 5] + [0] * 4]
    batch = predict_proba(model, ids, batch_size=2, device="cpu")
    single = [predict_proba(model, [r], batch_size=1, device="cpu")[0] for r in ids]
    assert np.allclose(batch, single, atol=1e-5)


def test_train_model_verbose_prints_one_line_per_epoch(capsys):
    ids, y = _toy_data()
    model = BiLSTMClassifier(_matrix(vocab_size=9), hidden=4)
    res = train_model(model, ids, y, ids, y, lr=0.01, batch_size=16, max_epochs=3, patience=3,
                      seed=1, device="cpu", verbose=True)
    lines = [l for l in capsys.readouterr().out.splitlines() if "epoch" in l]
    assert len(lines) == len(res["history"])
