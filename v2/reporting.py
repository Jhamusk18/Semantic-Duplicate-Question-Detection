"""Metrics and headless evaluation plots."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, log_loss,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)


def best_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    score = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[score[:-1].argmax()])


def evaluate(labels, probabilities, threshold):
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "log_loss": log_loss(labels, probabilities, labels=[0, 1]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def save_plots(output, labels, results, probabilities):
    figure, (roc, pr) = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for row in results:
        name = row["experiment"]
        p = probabilities[name]
        fpr, tpr, _ = roc_curve(labels, p)
        precision, recall, _ = precision_recall_curve(labels, p)
        line, = roc.plot(fpr, tpr, label=f"{name} ({row['roc_auc']:.3f})")
        pr.plot(recall, precision, label=name, color=line.get_color())
    roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
    pr.axhline(labels.mean(), linestyle="--", color="gray")
    roc.set(title="ROC · test set", xlabel="False positive rate", ylabel="True positive rate")
    pr.set(title="Precision–recall · test set", xlabel="Recall", ylabel="Precision")
    for axis in (roc, pr):
        axis.legend(fontsize=8)
    figure.savefig(output / "evaluation_curves.png", dpi=150)
    plt.close(figure)

    rows = (len(results) + 2) // 3
    figure, axes = plt.subplots(rows, 3, squeeze=False, figsize=(11, 3.5 * rows),
                                layout="constrained")
    for axis, row in zip(axes.flat, results):
        ConfusionMatrixDisplay(np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]]),
                               display_labels=["Distinct", "Duplicate"]).plot(
                                   ax=axis, cmap="Blues", colorbar=False, values_format="d")
        axis.set_title(row["experiment"])
    for axis in list(axes.flat)[len(results):]:
        axis.set_visible(False)
    figure.savefig(output / "confusion_matrices.png", dpi=150)
    plt.close(figure)
