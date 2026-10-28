"""GloVe + BiLSTM sentiment classifier: data, vocabulary, embeddings, model, training, metrics."""
import json
import re
from collections import Counter

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
