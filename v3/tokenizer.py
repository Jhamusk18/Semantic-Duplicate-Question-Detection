"""Training-only Keras vocabulary and reusable padding/truncation."""

import numpy as np
import tensorflow as tf
from tensorflow import keras


def make_tokenizer(config, vocabulary=None):
    return keras.layers.TextVectorization(
        max_tokens=config.vocabulary_size, standardize=None, split="whitespace",
        output_mode="int", output_sequence_length=config.sequence_length, vocabulary=vocabulary,
    )


def fit_tokenizer(train_questions, config):
    layer = make_tokenizer(config)
    dataset = tf.data.Dataset.from_tensor_slices(train_questions).batch(1024)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    layer.adapt(dataset.with_options(options))
    return layer


def tokenize(layer, questions, batch_size=2048):
    return np.concatenate([
        np.asarray(layer(tf.constant(questions[start:start + batch_size])), dtype=np.int32)
        for start in range(0, len(questions), batch_size)
    ])
