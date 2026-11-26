"""Reuse the V2 split and index question text within each partition."""

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from v2.data import load_pairs, question_groups, split_pairs


@dataclass
class IndexedPairs:
    questions: list[str]
    left: np.ndarray
    right: np.ndarray
    labels: np.ndarray

    @classmethod
    def from_frame(cls, frame):
        ids, texts = pd.factorize(
            np.concatenate([frame.clean1, frame.clean2]), sort=True
        )
        left, right = np.split(ids.astype(np.int32), 2)
        return cls(texts.tolist(), left, right, frame.is_duplicate.to_numpy())


def prepare_splits(path, seed):
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    pairs = load_pairs(path)
    input_rows = len(pairs)
    pairs["group"] = question_groups(pairs)
    pairs = pairs[pairs.clean1.ne("") & pairs.clean2.ne("")]
    splits = split_pairs(pairs, seed)
    metadata = {
        "input_rows": input_rows, "input_sha256": checksum, "usable_rows": len(pairs),
        "excluded_empty_text": input_rows - len(pairs), "seed": seed,
        "splits": {
            name: {
                "rows": len(frame), "groups": frame.group.nunique(),
                "duplicates": int(frame.is_duplicate.sum()),
                "row_ids_sha256": hashlib.sha256(
                    frame.source_row.to_numpy(dtype="<i8").tobytes()
                ).hexdigest(),
            }
            for name, frame in splits.items()
        },
    }
    return splits, metadata
