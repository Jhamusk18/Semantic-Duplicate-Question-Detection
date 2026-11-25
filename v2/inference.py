"""Prediction with a locally trained artifact."""

from v2.data import preprocess
from v2.features import build_features, feature_matrix

ARTIFACT_VERSION = 1


def predict_pair(question1, question2, artifact):
    if artifact["version"] != ARTIFACT_VERSION:
        raise ValueError("Unsupported model artifact version. Retrain with this checkout.")
    left, right = preprocess(question1), preprocess(question2)
    if not left or not right:
        raise ValueError("Both questions need at least one word or number.")
    blocks = build_features([left], [right], artifact["context"], artifact["blocks"])
    matrix = feature_matrix(blocks, artifact["blocks"])
    probability = float(artifact["model"].predict_proba(matrix)[0, 1])
    return probability, probability >= artifact["threshold"]
