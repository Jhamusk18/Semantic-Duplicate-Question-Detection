import unittest

import numpy as np
import pandas as pd

from v2.data import preprocess, question_groups, split_pairs, validate_split
from v2.features import build_features, fit_context
from v2.inference import ARTIFACT_VERSION, predict_pair
from v2.reporting import best_threshold, evaluate


class CapturingModel:
    def predict_proba(self, features):
        self.last_features = features
        return np.array([[0.4, 0.6]])


class PipelineTests(unittest.TestCase):
    def test_grouping_is_transitive_across_ids_and_clean_text(self):
        frame = pd.DataFrame({
            "qid1": ["1", "2", "9", "6", "8"],
            "qid2": ["2", "3", "4", "7", "10"],
            "clean1": ["first", "second", "same normalized text", "other", ""],
            "clean2": ["second", "same normalized text", "fourth", "", "different"],
        })
        a, b, c, d, e = question_groups(frame)
        self.assertEqual(a, b)
        self.assertEqual(b, c)
        self.assertEqual(len({a, d, e}), 3)

    def test_splits_are_deterministic_and_disjoint(self):
        frame = pd.DataFrame([
            {"qid1": f"{i}a", "qid2": f"{i}b", "clean1": f"left {i}",
             "clean2": f"right {i}", "is_duplicate": i % 2}
            for i in range(100)
        ])
        frame["group"] = question_groups(frame)
        splits = split_pairs(frame, 42)
        self.assertEqual([len(split) for split in splits.values()], [70, 10, 20])
        self.assertTrue(splits["test"].equals(split_pairs(frame, 42)["test"]))
        with self.assertRaisesRegex(RuntimeError, "leakage"):
            validate_split({**splits, "test": splits["train"]})

    def test_all_feature_blocks_are_exactly_symmetric_and_train_fitted(self):
        train = pd.DataFrame({"clean1": ["pair word", "second item", "pair pair"],
                              "clean2": ["word item", "third item", "last word"]})
        needed = {"basic", "hand", "bow", "tfidf", "embedding"}
        context = fit_context(train, needed, "lsa", 42)
        a = ["pair novelword", "a a b", "word item"]
        b = ["third item", "b a a a", "pair word"]
        forward = build_features(a, b, context, needed)
        reverse = build_features(b, a, context, needed)
        for name in needed:
            pd.testing.assert_frame_equal(forward[name], reverse[name], check_exact=True)
        self.assertNotIn("novelword", context["bow"].vocabulary_)
        self.assertNotIn("novelword", context["tfidf"][0].vocabulary_)
        self.assertNotIn("nov", context["tfidf"][1].vocabulary_)

    def test_inference_applies_the_same_features_and_inclusive_threshold(self):
        model = CapturingModel()
        art = {"version": ARTIFACT_VERSION, "model": model, "threshold": 0.6,
               "blocks": ("hand",), "context": {}}
        self.assertEqual(preprocess(" <b> WON’T </b>  "), "will not")
        self.assertEqual(predict_pair("How much?", "How many?", art), (0.6, True))
        forward = model.last_features
        self.assertEqual(predict_pair("How many?", "How much?", art), (0.6, True))
        pd.testing.assert_frame_equal(forward, model.last_features, check_exact=True)
        with self.assertRaisesRegex(ValueError, "word or number"):
            predict_pair("???", "How many?", art)

    def test_threshold_metrics_include_equal_scores(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.2, 0.8])
        threshold = best_threshold(y, p)
        result = evaluate(y, p, threshold)
        self.assertEqual(threshold, 0.2)
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (1, 1, 0, 2))
        self.assertEqual(result["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
