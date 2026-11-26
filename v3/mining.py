"""Sample one triplet per eligible anchor without an all-pairs matrix."""

import numpy as np


def choose_negative(candidates, similarities, positive_similarity, strategy, margin, fallback):
    if strategy == "hard":
        return candidates[similarities.argmax()], True
    allowed = ((similarities < positive_similarity) &
               (similarities > positive_similarity - margin))
    if not allowed.any():
        return fallback, False
    return candidates[np.where(allowed, similarities, -np.inf).argmax()], True


def sample_triplets(pool, rng, margin, strategy="random", embeddings=None):
    triplets = np.empty((len(pool.anchors), 3), dtype=np.int32)
    mined = 0
    for index, anchor in enumerate(pool.anchors):
        positive = rng.choice(pool.candidates(pool.positives, anchor))
        candidates = pool.candidates(pool.negatives, anchor)
        negative = rng.choice(candidates)
        if embeddings is not None and strategy != "random":
            anchor_embedding = embeddings[anchor]
            similarities = embeddings[candidates] @ anchor_embedding
            positive_similarity = embeddings[positive] @ anchor_embedding
            negative, changed = choose_negative(
                candidates, similarities, positive_similarity, strategy, margin, negative
            )
            mined += int(changed)
        triplets[index] = (anchor, positive, negative)
    return triplets, mined
