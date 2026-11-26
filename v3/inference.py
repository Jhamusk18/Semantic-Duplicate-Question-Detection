"""Shared cosine scoring and loading of the selected local artifact."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from v2.data import preprocess
from v3.config import Config
from v3.tokenizer import make_tokenizer, tokenize


class CosineScorer:
    def __init__(self, encoder, batch_size):
        self.batch_size = batch_size

        @tf.function(input_signature=[tf.TensorSpec((None, None), tf.int32)])
        def encode(tokens):
            return encoder(tokens, training=False)

        self._encode = encode

    def embed(self, tokens):
        lengths = np.count_nonzero(tokens, axis=1)
        order = np.argsort(lengths, kind="stable")
        vectors = []
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            width = max(1, lengths[indices[-1]])
            vectors.append(self._encode(tokens[indices, :width]).numpy())
        result = np.empty((len(tokens), vectors[0].shape[1]), dtype=vectors[0].dtype)
        result[order] = np.concatenate(vectors)
        return result

    def score(self, tokens, pairs):
        return self.score_embeddings(self.embed(tokens), pairs)

    @staticmethod
    def score_embeddings(vectors, pairs):
        output = np.empty(len(pairs.left), dtype=np.float32)
        for start in range(0, len(output), 2048):
            stop = start + 2048
            output[start:stop] = np.einsum(
                "ij,ij->i", vectors[pairs.left[start:stop]], vectors[pairs.right[start:stop]]
            )
        return output.clip(-1, 1)


class Predictor:
    def __init__(self, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "inference.json").read_text(encoding="utf-8"))
        self.config = Config(**metadata["config"])
        self.threshold = metadata["threshold"]
        vocabulary = json.loads((directory / "vocabulary.json").read_text(encoding="utf-8"))
        self.tokenizer = make_tokenizer(self.config, vocabulary)
        self.scorer = CosineScorer(
            keras.models.load_model(directory / "encoder.keras", compile=False),
            self.config.batch_size,
        )

    def predict(self, first, second):
        left, right = preprocess(first), preprocess(second)
        if not left or not right:
            raise ValueError("Both questions need at least one word or number.")
        encoded = self.scorer.embed(tokenize(self.tokenizer, sorted([left, right])))
        similarity = float(np.clip(encoded[0] @ encoded[1], -1, 1))
        return similarity, similarity >= self.threshold, (left, right)
