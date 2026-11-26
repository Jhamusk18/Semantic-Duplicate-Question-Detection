import importlib.util
import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

if importlib.util.find_spec("tensorflow") is None:
    raise unittest.SkipTest("Install requirements-v3.txt for TensorFlow checks.")

import tensorflow as tf
from streamlit.testing.v1 import AppTest

from v2.data import validate_split
from v3.config import Config
from v3.data import IndexedPairs, prepare_splits
from v3.inference import CosineScorer, Predictor
from v3.losses import TripletMarginLoss
from v3.model import build_encoder, build_trainer
from v3.tokenizer import fit_tokenizer, tokenize


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.keras.utils.set_random_seed(42)
        cls.config = Config(vocabulary_size=32, sequence_length=5, embedding_dim=4,
                            lstm_units=8, projection_dim=6, dropout=0.0, margin=2.0)
        cls.tokenizer = fit_tokenizer(["how learn python", "learn programming", "bake bread"],
                                      cls.config)

    def test_training_vocabulary_padding_truncation_and_unknown_words(self):
        known, unknown, long = tokenize(self.tokenizer, [
            "learn", "testonly", "learn learn learn learn learn testonly",
        ])
        self.assertGreater(known[0], 1)
        np.testing.assert_array_equal(known[1:], [0] * 4)
        np.testing.assert_array_equal(unknown, [1, 0, 0, 0, 0])
        np.testing.assert_array_equal(long, [known[0]] * 5)
        self.assertNotIn("testonly", self.tokenizer.get_vocabulary())

    def test_group_splits_use_disjoint_ids_and_texts(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "pairs.csv"
            pairs = [
                {"qid1": f"{i}a", "qid2": f"{i}b", "question1": f"left {i}",
                 "question2": f"right {i}", "is_duplicate": i % 2}
                for i in range(100)
            ]
            pd.DataFrame(pairs).to_csv(path, index=False)
            splits, summary = prepare_splits(path, 42)
            validate_split(splits)
            self.assertEqual([len(split) for split in splits.values()], [70, 10, 20])
            self.assertEqual(summary["usable_rows"], 100)

    def test_triplet_loss_monotonicity_and_margin(self):
        loss = TripletMarginLoss(margin=0.2)
        anchor = np.array([1.0, 0.0])

        def score(positive, negative):
            p = np.array([positive, np.sqrt(1 - positive**2)])
            n = np.array([negative, np.sqrt(1 - negative**2)])
            difference = np.array([[anchor @ n - anchor @ p]], np.float32)
            return float(loss(np.zeros_like(difference), difference))

        self.assertLess(score(0.7, 0.7), score(0.5, 0.7))
        self.assertGreater(score(0.7, 0.8), score(0.7, 0.5))
        self.assertEqual(score(0.9, 0.1), 0.0)

    def test_shared_encoder_masking_and_gradients(self):
        encoder = build_encoder(self.config, len(self.tokenizer.get_vocabulary()))
        trainer = build_trainer(encoder, self.config)
        self.assertIs(trainer.get_layer("question_encoder"), encoder)
        self.assertEqual(len(trainer.trainable_weights), len(encoder.trainable_weights))
        tokens = tokenize(self.tokenizer, ["learn python", "learn programming", "bake bread"])
        embeddings = encoder(tokens).numpy()
        self.assertEqual(embeddings.shape, (3, 6))
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(encoder(tokens[:, :2]), embeddings, atol=1e-6)
        with tf.GradientTape() as tape:
            gap = trainer(tuple(tokens[i:i + 1] for i in range(3)), training=True)
            loss = trainer.loss(tf.zeros_like(gap), gap)
        gradients = tape.gradient(loss, encoder.trainable_weights)
        self.assertTrue(all(g is not None for g in gradients))
        self.assertTrue(any(tf.reduce_any(tf.convert_to_tensor(g) != 0) for g in gradients))

    def test_scoring_is_symmetric_and_saved_model_works_in_streamlit(self):
        encoder = build_encoder(self.config, len(self.tokenizer.get_vocabulary()))
        tokens = tokenize(self.tokenizer, ["how learn python", "learn programming"])
        pairs = IndexedPairs([], np.array([0, 1]), np.array([1, 0]), np.array([0, 0]))
        scores = CosineScorer(encoder, 2).score(tokens, pairs)
        np.testing.assert_array_equal(scores[0], scores[1])
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            with patch.dict(os.environ, {"QUORA_V3_MODEL_DIR": root,
                                         "QUORA_V2_MODEL_PATH": str(directory / "absent")}):
                app = AppTest.from_file("v3/app.py", default_timeout=30).run()
                self.assertIn("Train V3", app.warning[0].value)
                encoder.save(directory / "encoder.keras")
                (directory / "inference.json").write_text(json.dumps({
                    "config": asdict(self.config), "threshold": 0.5,
                }))
                (directory / "vocabulary.json").write_text(json.dumps(
                    self.tokenizer.get_vocabulary()
                ))
                predictor = Predictor(directory)
                first = "<b>HOW</b> learn Python?"
                second = "Learn programming!"
                p, y, cleaned = predictor.predict(first, second)
                self.assertEqual(predictor.predict(second, first)[:2], (p, y))
                self.assertEqual(cleaned, ("how learn python", "learn programming"))
                self.assertAlmostEqual(p, float(scores[0]), places=6)
                app.run()
                app.button[0].click().run()
                self.assertIn("at least one word", app.warning[0].value)
                app.text_area[0].set_value(first)
                app.text_area[1].set_value(second)
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.metric[0].value, f"{p:.4f}")
                self.assertIn(cleaned[0], [line.value for line in app.text])
