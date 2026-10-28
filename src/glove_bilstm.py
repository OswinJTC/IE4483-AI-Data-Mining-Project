"""GloVe + BiLSTM sentiment classifier: data, vocabulary, embeddings, model, training, metrics."""
import json
import copy
import os
import random
import re
import subprocess
import time
import zipfile
from collections import Counter

import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, UNK = "<pad>", "<unk>"
_TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9']")


def tokenize(text: str) -> list[str]:
    """Lowercase words plus single punctuation marks, so negations and '!' survive."""
    return _TOKEN_RE.findall(text.lower())


def build_vocab(texts: list[str], min_freq: int = 2) -> dict[str, int]:
    counts = Counter(tok for text in texts for tok in tokenize(text))
    vocab = {PAD: 0, UNK: 1}
    for tok, n in counts.most_common():
        if n >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(tok, vocab[UNK]) for tok in tokenize(text)][:max_len]
    return ids + [vocab[PAD]] * (max_len - len(ids))


def load_split(path: str) -> tuple[list[str], list[int]]:
    """Load a labeled split; raises KeyError for an unlabeled one (e.g. the test split)."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [r["reviews"] for r in records], [int(r["sentiments"]) for r in records]


GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.100d.txt"


def download_glove(dest_dir: str = "data/glove") -> str:
    """Fetch GloVe 6B and extract only the 100d vectors; skipped if already present."""
    path = os.path.join(dest_dir, GLOVE_FILE)
    if os.path.exists(path):
        return path
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "glove.6B.zip")
    # curl uses the system trust store; Python's urllib failed TLS verification on this machine.
    subprocess.run(["curl", "-fL", "-o", zip_path, GLOVE_URL], check=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(GLOVE_FILE, dest_dir)
    os.remove(zip_path)
    return path


def load_glove(path: str, dim: int = 100) -> dict[str, np.ndarray]:
    """Read a GloVe text file, skipping malformed lines and vectors of the wrong dimension."""
    glove = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            try:
                glove[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue
    return glove


def build_embedding_matrix(
    vocab: dict[str, int], glove: dict[str, np.ndarray], dim: int = 100, seed: int = 42
) -> np.ndarray:
    """Rows follow vocab ids: GloVe vector if known, small random vector if not, zeros for <pad>."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.1, size=(len(vocab), dim)).astype(np.float32)
    for tok, idx in vocab.items():
        if tok in glove:
            matrix[idx] = glove[tok]
    matrix[vocab[PAD]] = 0.0
    return matrix


class BiLSTMClassifier(nn.Module):
    """Embedding -> (Bi)LSTM -> pooling -> Linear. Pad id 0 is masked everywhere it matters."""

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        hidden: int = 128,
        layers: int = 1,
        dropout: float = 0.4,
        pooling: str = "attention",
        bidirectional: bool = True,
        freeze_embeddings: bool = True,
    ):
        super().__init__()
        if pooling not in {"attention", "max", "last"}:
            raise ValueError(f"unknown pooling: {pooling}")
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(
            torch.as_tensor(embedding_matrix), freeze=freeze_embeddings, padding_idx=0
        )
        self.lstm = nn.LSTM(
            embedding_matrix.shape[1],
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.attention = nn.Linear(out_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(out_dim, 2)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        seq_len = ids.shape[1]
        lengths = (ids != 0).sum(dim=1).clamp(min=1)  # all-pad rows behave as length 1
        mask = torch.arange(seq_len, device=ids.device)[None, :] < lengths[:, None]
        packed = pack_padded_sequence(
            self.embedding(ids), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, _) = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)

        if self.pooling == "attention":
            scores = self.attention(out).squeeze(-1).masked_fill(~mask, float("-inf"))
            pooled = (torch.softmax(scores, dim=1).unsqueeze(-1) * out).sum(dim=1)
        elif self.pooling == "max":
            pooled = out.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        else:  # last real position of each direction
            pooled = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        return self.classifier(self.dropout(pooled))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Positive class = 1. Negative F1 is the F1 of class 0; undefined scores count as 0."""
    y_true, y_pred, y_prob = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "neg_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


@torch.no_grad()
def predict_proba(model, ids, batch_size: int = 256, device: str | None = None) -> np.ndarray:
    """Positive-class probability for each encoded review."""
    dev = _pick_device(device)
    model.to(dev).eval()
    x = torch.as_tensor(ids, dtype=torch.long)
    probs = [
        torch.softmax(model(x[i : i + batch_size].to(dev)), dim=-1)[:, 1].cpu()
        for i in range(0, len(x), batch_size)
    ]
    return torch.cat(probs).numpy()


def train_model(
    model,
    train_ids,
    train_y,
    val_ids,
    val_y,
    *,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 15,
    patience: int = 3,
    clip: float = 1.0,
    class_weighted: bool = False,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    """Train with AdamW; keep the weights with the best validation Negative F1."""
    set_seed(seed)
    dev = _pick_device(device)
    model.to(dev)
    x = torch.as_tensor(train_ids, dtype=torch.long)
    y = torch.as_tensor(train_y, dtype=torch.long)
    weight = None
    if class_weighted:
        counts = torch.bincount(y, minlength=2).float()
        weight = (counts.sum() / (2.0 * counts.clamp(min=1))).to(dev)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    plain_loss_fn = nn.CrossEntropyLoss()  # val loss stays comparable across weighting choices
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    val_x = torch.as_tensor(val_ids, dtype=torch.long)
    val_t = torch.as_tensor(val_y, dtype=torch.long)

    history, best_f1, best_epoch, best_state, stale = [], -1.0, 0, None, 0
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), batch_size):
            idx = order[i : i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x[idx].to(dev)), y[idx].to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total += loss.item() * len(idx)
        probs = predict_proba(model, val_ids, device=str(dev))
        with torch.no_grad():
            model.eval()
            logits = torch.cat([model(val_x[i : i + 256].to(dev)).cpu() for i in range(0, len(val_x), 256)])
            val_loss = plain_loss_fn(logits, val_t).item()
        m = compute_metrics(val_y, (probs >= 0.5).astype(int), probs)
        history.append({
            "epoch": epoch,
            "train_loss": total / len(x),
            "val_loss": val_loss,
            "val_neg_f1": m["neg_f1"],
            "val_acc": m["accuracy"],
        })
        if m["neg_f1"] > best_f1:
            best_f1, best_epoch, stale = m["neg_f1"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break
    train_seconds = time.time() - start

    model.load_state_dict(best_state)
    probs = predict_proba(model, val_ids, device=str(dev))
    return {
        "history": history,
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
        "metrics": compute_metrics(val_y, (probs >= 0.5).astype(int), probs),
    }
