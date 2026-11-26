"""Cosine margin objective. The encoder returns unit-length vectors."""

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="quora")
class TripletMarginLoss(keras.losses.Loss):
    def __init__(self, margin=0.2, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, distance_difference):
        return tf.nn.relu(distance_difference + self.margin)

    def get_config(self):
        return {**super().get_config(), "margin": self.margin}
