"""Build reliable, bidirectional relationships from explicitly labeled pairs."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components


def adjacency(edges, size):
    left, right = edges.T
    graph = coo_matrix(
        (np.ones(2 * len(edges), dtype=np.int8),
         (np.concatenate([left, right]), np.concatenate([right, left]))),
        shape=(size, size),
    ).tocsr()
    return graph


@dataclass
class TripletPool:
    anchors: np.ndarray
    positives: csr_matrix
    negatives: csr_matrix
    stats: dict

    def candidates(self, graph, anchor):
        return graph.indices[graph.indptr[anchor]:graph.indptr[anchor + 1]]


def build_pool(pairs):
    size = len(pairs.questions)
    low = np.minimum(pairs.left, pairs.right).astype(np.int64)
    high = np.maximum(pairs.left, pairs.right).astype(np.int64)
    keys = low * size + high
    positive_keys = np.unique(keys[pairs.labels == 1])
    negative_keys = np.unique(keys[pairs.labels == 0])
    contradictions = np.intersect1d(positive_keys, negative_keys, assume_unique=True)
    positive_keys = np.setdiff1d(positive_keys, contradictions, assume_unique=True)
    negative_keys = np.setdiff1d(negative_keys, contradictions, assume_unique=True)

    positive_edges = np.column_stack(np.divmod(positive_keys, size))
    negative_edges = np.column_stack(np.divmod(negative_keys, size))
    positive_edges = positive_edges[positive_edges[:, 0] != positive_edges[:, 1]]
    positives = adjacency(positive_edges, size)
    _, positive_components = connected_components(positives, directed=False)
    # A positive path vetoes negative edges. Only explicitly labeled edges supply positives.
    within_positive = positive_components[negative_edges[:, 0]] == positive_components[
        negative_edges[:, 1]
    ]
    negatives = adjacency(negative_edges[~within_positive], size)
    has_positive = np.diff(positives.indptr) > 0
    has_negative = np.diff(negatives.indptr) > 0
    anchors = np.flatnonzero(has_positive & has_negative).astype(np.int32)
    stats = {
        "training_questions": size,
        "eligible_anchors": len(anchors),
        "skipped_anchors": size - len(anchors),
        "anchors_with_positive_but_no_negative": int((has_positive & ~has_negative).sum()),
        "contradictory_relationships": len(contradictions),
        "negatives_vetoed_by_identity_or_positive_path": int(within_positive.sum()),
        "positive_relationships": len(positive_edges),
        "negative_relationships": int((~within_positive).sum()),
    }
    return TripletPool(anchors, positives, negatives, stats)
