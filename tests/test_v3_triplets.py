import unittest

import numpy as np

from v3.data import IndexedPairs
from v3.mining import choose_negative, sample_triplets
from v3.triplets import build_pool


class TripletTests(unittest.TestCase):
    def setUp(self):
        pairs = IndexedPairs(
            list("abcdefg"),
            np.array([0, 2, 0, 0, 3, 4, 1, 4, 4]),
            np.array([1, 1, 2, 3, 4, 3, 1, 5, 6]),
            np.array([1, 1, 0, 0, 0, 1, 0, 1, 0]),
        )
        self.pool = build_pool(pairs)

    def test_only_labeled_edges_and_no_conflicts_or_positive_paths(self):
        pool = self.pool
        np.testing.assert_array_equal(pool.anchors, [0, 4])
        np.testing.assert_array_equal(pool.candidates(pool.positives, 1), [0, 2])
        np.testing.assert_array_equal(pool.candidates(pool.negatives, 0), [3])
        np.testing.assert_array_equal(pool.candidates(pool.negatives, 4), [6])
        self.assertEqual(pool.stats["contradictory_relationships"], 1)
        self.assertEqual(pool.stats["negatives_vetoed_by_identity_or_positive_path"], 2)
        expected = np.array([[0, 1, 3], [4, 5, 6]])
        sampled, mined = sample_triplets(pool, np.random.default_rng(42), 0.2)
        np.testing.assert_array_equal(sampled, expected)
        self.assertEqual(mined, 0)

    def test_hard_and_semi_hard_select_different_labeled_candidates(self):
        candidates = np.array([3, 4, 5])
        scores = np.array([0.93, 0.72, 0.40])
        self.assertEqual(choose_negative(candidates, scores, 0.8, "hard", 0.2, 5), (3, True))
        self.assertEqual(choose_negative(candidates, scores, 0.8, "semi-hard", 0.2, 5), (4, True))
        self.assertEqual(choose_negative(candidates, scores, 0.8, "semi-hard", 0.02, 5), (5, False))

    def test_mined_triplet_uses_current_embedding_scores(self):
        pairs = IndexedPairs(list("abcd"), np.array([0, 0, 0]), np.array([1, 2, 3]),
                             np.array([1, 0, 0]))
        vectors = np.array([[1, 0], [0.8, 0.6], [0.7, np.sqrt(0.51)], [0.95, np.sqrt(0.0975)]])
        pool = build_pool(pairs)
        hard, _ = sample_triplets(pool, np.random.default_rng(42), 0.2, "hard", vectors)
        semi, count = sample_triplets(pool, np.random.default_rng(42), 0.2, "semi-hard", vectors)
        np.testing.assert_array_equal(hard, [[0, 1, 3]])
        np.testing.assert_array_equal(semi, [[0, 1, 2]])
        self.assertEqual(count, 1)
