"""GloVe + BiLSTM sentiment classifier: data, vocabulary, embeddings, model, training, metrics."""
import json
import os
import re
import subprocess
import zipfile
from collections import Counter

import numpy as np

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
