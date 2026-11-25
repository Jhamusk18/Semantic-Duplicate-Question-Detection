"""Run a component-disjoint ablation and export the validation-selected model."""

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from v2.data import load_pairs, question_groups, split_pairs
from v2.features import build_features, feature_matrix, fit_context
from v2.inference import ARTIFACT_VERSION
from v2.reporting import best_threshold, evaluate, save_plots

SEED = 42
V1_EXPERIMENTS = {
    "BoW": ("bow",),
    "BoW + basic": ("bow", "basic"),
    "BoW + handcrafted": ("bow", "hand"),
}
V2_EXPERIMENTS = {
    "Handcrafted": ("hand",),
    "TF-IDF": ("tfidf",),
    "Handcrafted + TF-IDF": ("hand", "tfidf"),
}


def parse_args(profile):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output", type=Path, default=Path(f"artifacts/{profile}"))
    parser.add_argument("--sample", type=int, help="Sample this many usable rows after grouping.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-trees", type=int, default=1_500)
    if profile == "v2":
        parser.add_argument("--embeddings", choices=["lsa", "minilm"], default="lsa",
                            nargs="?", const="minilm")
        parser.add_argument("--no-embeddings", action="store_const", const=None, dest="embeddings")
    else:
        parser.set_defaults(embeddings=None)
    options = parser.parse_args()
    for name in ("sample", "threads", "max_trees"):
        value = getattr(options, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return options


def train_models(train, validation, blocks, experiments, options):
    candidates = {}
    for name, keys in experiments.items():
        model = XGBClassifier(
            n_estimators=options.max_trees, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            tree_method="hist", eval_metric="logloss", early_stopping_rounds=40,
            random_state=options.seed, n_jobs=options.threads,
        )
        x_train = feature_matrix(blocks["train"], keys)
        x_val = feature_matrix(blocks["validation"], keys)
        model.fit(x_train, train.is_duplicate, eval_set=[(x_val, validation.is_duplicate)],
                  verbose=False)
        p_val = model.predict_proba(x_val)[:, 1]
        loss = log_loss(validation.is_duplicate, p_val)
        candidates[name] = {
            "model": model, "keys": keys, "threshold": best_threshold(validation.is_duplicate, p_val),
            "validation_log_loss": loss, "trees": model.best_iteration + 1, "features": x_train.shape[1],
        }
        print(f"{name:<32} validation log loss {loss:.5f}", flush=True)
    return candidates


def save_evaluation(output, splits, blocks, context, candidates, selected):
    test = splits["test"]
    train_prior = float(splits["train"].is_duplicate.mean())
    val_labels = splits["validation"].is_duplicate
    probabilities = {"Train prior": np.full(len(test), train_prior)}
    results = [{
        "experiment": "Train prior", "selected": False, "threshold": 0.5,
        "validation_log_loss": log_loss(val_labels, np.full(len(val_labels), train_prior)),
        "trees": 0, "features": 0,
        **evaluate(test.is_duplicate, probabilities["Train prior"], 0.5),
    }]

    sample = test.head(256)
    needed = set().union(*(item["keys"] for item in candidates.values()))
    forward = build_features(sample.clean1, sample.clean2, context, needed)
    reversed_blocks = build_features(sample.clean2, sample.clean1, context, needed)
    for key in needed:
        pd.testing.assert_frame_equal(forward[key], reversed_blocks[key], check_exact=True)

    for name, candidate in candidates.items():
        model, keys, threshold = (candidate[key] for key in ("model", "keys", "threshold"))
        p_test = model.predict_proba(feature_matrix(blocks, keys))[:, 1]
        p_forward = model.predict_proba(feature_matrix(forward, keys))[:, 1]
        p_reverse = model.predict_proba(feature_matrix(reversed_blocks, keys))[:, 1]
        np.testing.assert_array_equal(p_forward, p_reverse)
        probabilities[name] = p_test
        results.append({
            "experiment": name, "selected": name == selected,
            **{key: candidate[key] for key in ("threshold", "validation_log_loss", "trees", "features")},
            **evaluate(test.is_duplicate, p_test, threshold),
        })

    winner = candidates[selected]
    p = probabilities[selected]
    predictions = p >= winner["threshold"]
    errors = test[["source_row", "qid1", "qid2", "question1", "question2", "is_duplicate"]].copy()
    errors["prediction"] = predictions.astype(int)
    errors["p_duplicate"] = p
    errors["error_type"] = np.where(predictions, "false_positive", "false_negative")
    errors["confidence"] = np.where(predictions, p, 1 - p)
    errors = errors[errors.prediction != errors.is_duplicate]
    errors.sort_values(["confidence", "source_row"], ascending=[False, True]).head(200).to_csv(
        output / "error_analysis.csv", index=False)

    joblib.dump({
        "version": ARTIFACT_VERSION, "experiment": selected, "blocks": winner["keys"],
        "model": winner["model"], "threshold": winner["threshold"],
        "context": {key: context[key] for key in winner["keys"] if key in context},
    }, output / "model.joblib", compress=3)
    table = pd.DataFrame(results)
    table.to_csv(output / "ablation_results.csv", index=False, float_format="%.8f")
    save_plots(output, test.is_duplicate, results, probabilities)
    print(table.round(4).to_string(index=False), flush=True)
    return len(sample)


def run(options, profile, output):
    with options.data.open("rb") as stream:
        input_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    frame = load_pairs(options.data)
    input_rows = len(frame)
    frame["group"] = question_groups(frame)
    frame = frame[frame.clean1.ne("") & frame.clean2.ne("")]
    excluded = input_rows - len(frame)
    if options.sample is not None and options.sample < len(frame):
        frame = frame.sample(options.sample, random_state=options.seed).sort_values("source_row")
    splits = split_pairs(frame, options.seed)
    summary = {
        "source": options.data.name, "input_sha256": input_sha256, "input_rows": input_rows,
        "excluded_empty_text": excluded, "used_rows": len(frame), "sample": options.sample,
        "seed": options.seed, "max_trees": options.max_trees, "threads": options.threads,
        "profile": profile, "sentence_embeddings": options.embeddings,
        "largest_component_rows": int(frame.group.value_counts().max()),
        "splits": {
            name: {"rows": len(split), "groups": split.group.nunique(),
                   "duplicates": int(split.is_duplicate.sum())}
            for name, split in splits.items()
        },
    }
    print(f"Loaded {input_rows:,} rows, using {len(frame):,} after filtering/sampling.", flush=True)
    print(f"Component-disjoint split: {summary['splits']}", flush=True)

    experiments = V1_EXPERIMENTS.copy()
    if profile == "v2":
        experiments.update(V2_EXPERIMENTS)
    if options.embeddings:
        experiments[f"Handcrafted + TF-IDF + {options.embeddings.upper()}"] = (
            "hand", "tfidf", "embedding")
    needed = set().union(*experiments.values())
    print("Fitting feature transformers on training questions.", flush=True)
    context = fit_context(splits["train"], needed, options.embeddings, options.seed)

    blocks = {}
    for name in ("train", "validation"):
        print(f"Building {name} features.", flush=True)
        part = splits[name]
        blocks[name] = build_features(part.clean1, part.clean2, context, needed)
    candidates = train_models(splits["train"], splits["validation"], blocks, experiments, options)
    selected = min(candidates, key=lambda name: candidates[name]["validation_log_loss"])
    summary["selected_by_validation_log_loss"] = selected
    print(f"Selected {selected}. Building held-out test features.", flush=True)
    del blocks
    test = splits["test"]
    test_blocks = build_features(test.clean1, test.clean2, context, needed)
    summary["swap_test_rows_per_model"] = save_evaluation(
        output, splits, test_blocks, context, candidates, selected)
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main(profile="v2"):
    options = parse_args(profile)
    output = options.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise SystemExit(f"Output must be a new or empty directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".quora-", dir=output.parent) as temporary:
        staged = Path(temporary) / "result"
        staged.mkdir()
        with threadpool_limits(limits=options.threads):
            run(options, profile, staged)
        staged.rename(output)
    print(f"Saved model and evaluation to {output}", flush=True)


if __name__ == "__main__":
    main()
