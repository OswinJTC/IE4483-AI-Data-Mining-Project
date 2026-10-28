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
    """Lowercase words plus single punctuation marks, so negations and '!' survive.

    Example: tokenize("I did NOT like it!") -> ["i", "did", "not", "like", "it", "!"]
    Apostrophes stay inside words: "don't" -> ["don't"] (GloVe splits it, so it has no GloVe vector).
    """
    return _TOKEN_RE.findall(text.lower())


def build_vocab(texts: list[str], min_freq: int = 2) -> dict[str, int]:
    """Map each frequent word to an integer id. Ids 0 and 1 are reserved for padding and unknown words.

    Example: build_vocab(["good good movie", "good film"], min_freq=2)
             -> {"<pad>": 0, "<unk>": 1, "good": 2}   ("movie" and "film" occur once, so they are dropped)
    Call it on TRAINING texts only, otherwise validation words leak into the vocabulary.
    """
    counts = Counter(tok for text in texts for tok in tokenize(text))  # word -> number of occurrences
    vocab = {PAD: 0, UNK: 1}
    for tok, n in counts.most_common():  # most frequent words get the smallest ids
        if n >= min_freq:
            vocab[tok] = len(vocab)      # next free id
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    """Turn a review into exactly max_len ids: unknown words -> 1, cut if too long, pad with 0 if too short.

    Example (vocab {"<pad>":0, "<unk>":1, "good":2, "movie":3}, max_len=5):
        encode("good movie xyz", vocab, 5) -> [2, 3, 1, 0, 0]   # "xyz" is unknown, then two pads
        encode("good " * 9, vocab, 5)      -> [2, 2, 2, 2, 2]   # truncated to 5
    """
    ids = [vocab.get(tok, vocab[UNK]) for tok in tokenize(text)][:max_len]  # lookup, then truncate
    return ids + [vocab[PAD]] * (max_len - len(ids))                        # right-pad with 0


def load_split(path: str) -> tuple[list[str], list[int]]:
    """Load a labeled split; raises KeyError for an unlabeled one (e.g. the test split).

    Example: texts, labels = load_split("data/processed/val_split.json")
             texts[0] -> "great boots, very comfortable", labels[0] -> 1   (1 = positive, 0 = negative)
    """
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [r["reviews"] for r in records], [int(r["sentiments"]) for r in records]


GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.100d.txt"


def download_glove(dest_dir: str = "data/glove") -> str:
    """Fetch GloVe 6B and extract only the 100d vectors; skipped if already present.

    Returns the path of glove.6B.100d.txt (about 347 MB; the 822 MB zip is deleted after extraction).
    """
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
    """Read a GloVe text file, skipping malformed lines and vectors of the wrong dimension.

    Each line is "word n1 n2 ... n100". Example: glove["good"] -> array([-0.031, 0.120, 0.539, ...]) (100 floats)
    """
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
    """Rows follow vocab ids: GloVe vector if known, small random vector if not, zeros for <pad>.

    Example: with 7,066 vocabulary words the result has shape (7066, 100);
             matrix[vocab["good"]] equals glove["good"], matrix[0] (the pad row) is all zeros.
    """
    rng = np.random.default_rng(seed)
    # Start every row as a small random vector (covers words GloVe lacks, e.g. "it's", and <unk>)...
    matrix = rng.normal(0.0, 0.1, size=(len(vocab), dim)).astype(np.float32)
    for tok, idx in vocab.items():
        if tok in glove:
            matrix[idx] = glove[tok]   # ...then overwrite with the pretrained vector where one exists
    matrix[vocab[PAD]] = 0.0           # padding carries no information
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
        # ids: (batch, seq_len). Real length of each review = number of non-pad ids,
        # e.g. [5, 9, 2, 0, 0] -> 3. Rows that are all padding are treated as length 1 to avoid NaNs.
        lengths = (ids != 0).sum(dim=1).clamp(min=1)
        # mask[b, t] is True for real positions: lengths 3 and 5 with seq_len 5 ->
        # [[T,T,T,F,F], [T,T,T,T,T]]. Pooling uses it so padding never influences the result.
        mask = torch.arange(seq_len, device=ids.device)[None, :] < lengths[:, None]
        # Packing tells the LSTM to stop at each review's real end instead of reading the padding.
        packed = pack_padded_sequence(
            self.embedding(ids), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, _) = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)

        # out: (batch, seq_len, hidden * 2) = one state per word, each seeing context from both sides.
        if self.pooling == "attention":
            # A learned score per word -> softmax -> weighted average, so sentiment-heavy words
            # ("terrible", "love") count more than filler words. Padding gets -inf, i.e. weight 0.
            scores = self.attention(out).squeeze(-1).masked_fill(~mask, float("-inf"))
            pooled = (torch.softmax(scores, dim=1).unsqueeze(-1) * out).sum(dim=1)
        elif self.pooling == "max":
            # Per feature, keep the largest value over the real words (padding set to -inf).
            pooled = out.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        else:  # last real position of each direction
            pooled = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        # pooled: (batch, hidden * 2) -> dropout (training only) -> linear -> (batch, 2) logits [negative, positive].
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
    """Positive class = 1. Negative F1 is the F1 of class 0; undefined scores count as 0.

    Example: compute_metrics([1,1,1,0], [1,1,1,1], [.9,.8,.7,.6]) -> accuracy 0.75, neg_f1 0.0
             (predicting only "positive" never finds a negative review, so Negative F1 is 0).
    confusion_matrix is [[tn, fp], [fn, tp]].
    """
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
    """Positive-class probability for each encoded review.

    Example: predict_proba(model, X_val) -> array([0.97, 0.03, 0.88, ...]) with one value per review.
    """
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
    """Train with AdamW; keep the weights with the best validation Negative F1.

    Each epoch: shuffle the training reviews, update the weights batch by batch (clipping gradients so
    updates stay stable), then score the validation set. If validation Negative F1 has not improved for
    `patience` epochs, stop and restore the best epoch's weights.
    Example result: {"best_epoch": 8, "history": [{"epoch": 1, "train_loss": 0.45, ...}, ...],
                     "metrics": {...}, "train_seconds": 417.0}
    class_weighted=True weights each class by n / (2 * class_count), so the rare negative class
    (about 15% of reviews) is penalised ~3x more per mistake than the positive class.
    """
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
