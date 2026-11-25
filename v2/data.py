"""Load labeled pairs and keep their connected question groups together."""

import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SPLIT_FRACTIONS = np.array([0.70, 0.10, 0.20])
SPLIT_NAMES = ("train", "validation", "test")
FIELDS = ("qid1", "qid2", "question1", "question2", "is_duplicate")

CONTRACTIONS = [
    (r"\bwon't\b", "will not"), (r"\bcan't\b", "can not"),
    (r"n't\b", " not"), (r"'re\b", " are"), (r"'ve\b", " have"),
    (r"'ll\b", " will"), (r"'d\b", " would"), (r"'m\b", " am"),
]
CONTRACTIONS = [(re.compile(pattern), replacement) for pattern, replacement in CONTRACTIONS]
SYMBOLS = str.maketrans({"%": " percent ", "$": " dollar ", "₹": " rupee ",
                         "€": " euro ", "@": " at ", "’": "'"})
HTML = re.compile(r"<[^>]+>")
PUNCTUATION = re.compile(r"[^\w]+")
SPACE = re.compile(r"\s+")


def preprocess(text):
    text = unicodedata.normalize("NFKC", text).casefold().translate(SYMBOLS)
    text = HTML.sub(" ", text.replace("[math]", " "))
    for pattern, replacement in CONTRACTIONS:
        text = pattern.sub(replacement, text)
    return SPACE.sub(" ", PUNCTUATION.sub(" ", text)).strip()


def load_pairs(path):
    frame = pd.read_csv(path, keep_default_na=False, dtype=dict.fromkeys(FIELDS, str))
    missing = set(FIELDS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if not frame["is_duplicate"].isin(["0", "1"]).all():
        raise ValueError("Expected a binary is_duplicate label on every row.")
    if frame[["qid1", "qid2"]].apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("Expected a question ID on each side of every pair.")
    frame["is_duplicate"] = frame["is_duplicate"].astype("int8")
    frame.insert(0, "source_row", np.arange(len(frame)))
    unique_text = pd.unique(pd.concat([frame.question1, frame.question2], ignore_index=True))
    cleaned = {text: preprocess(text) for text in unique_text}
    frame["clean1"] = frame.question1.map(cleaned)
    frame["clean2"] = frame.question2.map(cleaned)
    return frame


def question_groups(frame):
    """Link qids across pairs and across identical, nonempty model inputs."""
    qids, unique_ids = pd.factorize(np.concatenate([frame.qid1, frame.qid2]))
    text_codes, unique_text = pd.factorize(np.concatenate([frame.clean1, frame.clean2]))
    left, right = np.split(qids, 2)
    # Empty inputs still link their pair's IDs but do not all share one text node.
    nonempty = np.asarray(unique_text)[text_codes] != ""
    sources = np.concatenate([left, qids[nonempty]])
    targets = np.concatenate([right, len(unique_ids) + text_codes[nonempty]])
    size = len(unique_ids) + len(unique_text)
    graph = coo_matrix((np.ones(len(sources), dtype=np.int8), (sources, targets)),
                       shape=(size, size))
    _, labels = connected_components(graph, directed=False)
    return labels[left]


def split_pairs(frame, seed):
    """Allocate whole components to approximate the per-label row targets."""
    groups, rows = np.unique(frame.group, return_inverse=True)
    labels = frame.is_duplicate.to_numpy()
    counts = np.bincount(rows * 2 + labels, minlength=len(groups) * 2).reshape(-1, 2)
    totals = counts.sum(axis=0)
    if (totals == 0).any() or len(groups) < 3:
        raise ValueError("Need at least three components and both labels to split pairs.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    order = order[np.argsort(-counts[order].sum(axis=1), kind="stable")]
    targets = SPLIT_FRACTIONS[:, None] * totals
    deficits = targets / totals
    assigned = np.empty(len(groups), dtype=np.int8)
    for index in order:
        weight = counts[index] / totals
        # Prefer the largest reduction in squared per-label target error.
        split = int((deficits @ weight).argmax())
        assigned[index] = split
        deficits[split] -= weight

    splits = {
        name: frame.loc[assigned[rows] == index].reset_index(drop=True)
        for index, name in enumerate(SPLIT_NAMES)
    }
    validate_split(splits)
    return splits


def validate_split(splits):
    seen = [set(), set(), set()]
    for name, part in splits.items():
        if part.is_duplicate.nunique() != 2:
            raise ValueError(f"{name} lacks both classes. Use a larger sample or inspect the graph.")
        identifiers = [
            set(part.group),
            set(part.qid1) | set(part.qid2),
            (set(part.clean1) | set(part.clean2)) - {""},
        ]
        for previous, current in zip(seen, identifiers, strict=True):
            if not previous.isdisjoint(current):
                raise RuntimeError(f"Question leakage detected in {name}.")
            previous.update(current)
