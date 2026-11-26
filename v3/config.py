"""Configuration shared by training and artifact metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    vocabulary_size: int = 30_000
    sequence_length: int = 64
    embedding_dim: int = 128
    lstm_units: int = 128
    projection_dim: int = 128
    dropout: float = 0.2
    batch_size: int = 128
    learning_rate: float = 0.001
    margin: float = 0.2
    seed: int = 42
