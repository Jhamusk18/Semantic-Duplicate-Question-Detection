"""Threshold-based evaluation of cosine scores and experiment plots."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, auc, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)


def threshold_curve(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    p, r = precision[:-1], recall[:-1]
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return thresholds, f1


def metrics(labels, scores, threshold):
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "f1": f1_score(labels, pred, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores),
        "pr_auc": auc(recall, precision),
        "threshold": threshold,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def save_errors(output, test, scores, threshold):
    errors = test[["source_row", "question1", "question2", "is_duplicate"]].copy()
    errors["cosine_similarity"] = scores
    errors["prediction"] = (scores >= threshold).astype(int)
    errors["error_type"] = np.where(errors.prediction == 1, "false_positive", "false_negative")
    errors["threshold_distance"] = np.abs(scores - threshold)
    errors = errors[errors.prediction != errors.is_duplicate]
    errors.sort_values(["error_type", "threshold_distance", "source_row"],
                       ascending=[True, False, True]).groupby("error_type").head(100).to_csv(
                           output / "error_analysis.csv", index=False)


def save_plots(output, validation_labels, test_labels, runs):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    loss_ax, val_ax, roc_ax, pr_ax, threshold_ax, hist_ax = axes.flat
    for name, run in runs.items():
        history, test_scores, validation_scores, result = run
        epochs = [row["epoch"] for row in history]
        line, = loss_ax.plot(epochs, [row["train_loss"] for row in history], label=name)
        style = {"label": name, "color": line.get_color()}
        val_ax.plot(epochs, [row["validation_f1"] for row in history], **style)
        fpr, tpr, _ = roc_curve(test_labels, test_scores)
        roc_ax.plot(fpr, tpr, **style)
        p, r, _ = precision_recall_curve(test_labels, test_scores)
        pr_ax.plot(r, p, **style)
        thresholds, f1 = threshold_curve(validation_labels, validation_scores)
        threshold_ax.plot(thresholds, f1, **style)
        threshold_ax.scatter(result["threshold"], result["best_validation_f1"],
                             color=line.get_color(), s=20)
        for label, scale in [(0, "--"), (1, "-")]:
            hist_ax.hist(test_scores[test_labels == label], bins=np.linspace(-1, 1, 42),
                         histtype="step", density=True, linestyle=scale, color=line.get_color(),
                         label=f"{name} · {'duplicate' if label else 'distinct'}")

    for axis, title, xlabel, ylabel in [
        (loss_ax, "Training loss", "Epoch", "Triplet loss"),
        (val_ax, "Validation F1", "Epoch", "F1"),
        (roc_ax, "Test ROC", "False positive rate", "True positive rate"),
        (pr_ax, "Test precision–recall", "Recall", "Precision"),
        (threshold_ax, "Validation threshold search", "Cosine threshold", "F1"),
        (hist_ax, "Test similarity distributions", "Cosine similarity", "Density"),
    ]:
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.legend(fontsize=8)
    roc_ax.plot([0, 1], [0, 1], linestyle=":", color="gray")
    pr_ax.axhline(test_labels.mean(), linestyle=":", color="gray")
    fig.savefig(output / "evaluation_curves.png", dpi=145)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(runs), squeeze=False, figsize=(5 * len(runs), 4),
                             layout="constrained")
    for axis, (name, (_, _, _, result)) in zip(axes.flat, runs.items(), strict=True):
        matrix = np.array([[result["tn"], result["fp"]], [result["fn"], result["tp"]]])
        ConfusionMatrixDisplay(matrix, display_labels=["Distinct", "Duplicate"]).plot(
            ax=axis, colorbar=False, values_format="d", cmap="Blues")
        axis.set_title(name)
    fig.savefig(output / "confusion_matrices.png", dpi=145)
    plt.close(fig)
