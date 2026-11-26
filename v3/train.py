"""Train and compare Siamese LSTM triplet objectives on grouped Quora pairs."""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from v3.config import Config
from v3.data import IndexedPairs, prepare_splits
from v3.evaluate import metrics, save_errors, save_plots, threshold_curve
from v3.inference import CosineScorer
from v3.mining import sample_triplets
from v3.model import build_encoder, build_trainer
from v3.tokenizer import fit_tokenizer, make_tokenizer, tokenize
from v3.triplets import build_pool


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/v3"))
    parser.add_argument("--strategies", nargs="+", default=["random", "semi-hard"],
                        choices=["random", "hard", "semi-hard"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--refresh", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--train-limit", type=int, help="Subsample training rows only for a pilot.")
    parser.add_argument("--eval-limit", type=int, help="Subsample validation and test for a smoke run.")
    parser.add_argument("--resume", action="store_true")
    for name, value in asdict(Config()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    positive = ["epochs", "refresh", "patience", "lr_patience", "threads", "batch_size",
                "sequence_length", "embedding_dim", "lstm_units", "projection_dim"]
    if any(getattr(args, name) < 1 for name in positive):
        parser.error("Epoch counts, dimensions, thread counts, and batch size must be positive.")
    if args.vocabulary_size < 3 or args.warmup < 0 or not 0 <= args.dropout < 1:
        parser.error("Use a vocabulary of at least three, nonnegative warmup, and dropout in [0, 1).")
    if not 0 < args.margin <= 2 or not 0 < args.learning_rate:
        parser.error("Use a positive learning rate and a margin in (0, 2].")
    if any(limit is not None and limit < 2 for limit in (args.train_limit, args.eval_limit)):
        parser.error("Sample limits must be at least two.")
    if len(set(args.strategies)) != len(args.strategies):
        parser.error("Provide each strategy once.")
    return args


def write_json(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def epoch_dataset(tokens, triplets, batch_size):
    values = tuple(tokens[triplets[:, index]] for index in range(3))
    labels = np.zeros((len(triplets), 1), dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((values, labels)).batch(batch_size)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return dataset.with_options(options).prefetch(1)


def train_strategy(path, strategy, args, config, tokens, indexed, vocabulary, pool):
    path.mkdir(exist_ok=True)
    keras.backend.clear_session()
    keras.utils.set_random_seed(config.seed)
    history_path = path / "training_history.json"
    history = json.loads(history_path.read_text()) if history_path.is_file() else []
    if len(history) > args.epochs:
        raise SystemExit("--epochs cannot be smaller than the number of saved epochs.")
    if history:
        trainer = keras.models.load_model(path / f"epoch-{history[-1]['epoch']}.keras")
        encoder = trainer.get_layer("question_encoder")
    else:
        encoder = build_encoder(config, len(vocabulary))
        trainer = build_trainer(encoder, config)

    scorer = CosineScorer(encoder, config.batch_size)
    best = max(history, key=lambda row: row["validation_f1"]) if history else None
    embeddings = None
    mining_ids = np.unique(np.concatenate([
        pool.anchors, pool.positives[pool.anchors].indices, pool.negatives[pool.anchors].indices,
    ]))
    validation = indexed["validation"]
    for epoch in range(len(history) + 1, args.epochs + 1):
        if best and epoch - 1 - best["epoch"] >= args.patience:
            break
        rng = np.random.default_rng([config.seed, epoch])
        start = time.perf_counter()
        if strategy != "random" and epoch > args.warmup:
            if embeddings is None or (epoch - args.warmup - 1) % args.refresh == 0:
                embeddings = np.zeros((len(tokens["train"]), config.projection_dim), np.float32)
                embeddings[mining_ids] = scorer.embed(tokens["train"][mining_ids])

        examples, mined = sample_triplets(pool, rng, config.margin, strategy, embeddings)
        rng.shuffle(examples)
        data = epoch_dataset(tokens["train"], examples, config.batch_size)
        current_lr = float(trainer.optimizer.learning_rate.numpy())
        loss = trainer.fit(data, epochs=1, verbose=0).history["loss"][0]
        validation_scores = scorer.score(tokens["validation"], validation)
        thresholds, f1 = threshold_curve(validation.labels, validation_scores)
        pick = int(f1.argmax())
        row = {
            "epoch": epoch, "train_loss": float(loss), "validation_f1": float(f1[pick]),
            "threshold": float(thresholds[pick]), "triplets": len(examples),
            "mined_triplets": mined, "learning_rate": current_lr,
            "elapsed_seconds": time.perf_counter() - start,
        }
        if best is None or row["validation_f1"] > best["validation_f1"]:
            best = row
        else:
            wait = epoch - best["epoch"]
            if wait % args.lr_patience == 0:
                trainer.optimizer.learning_rate.assign(max(current_lr * 0.5, 1e-6))
        history.append(row)
        pending = path / "checkpoint.tmp.keras"
        trainer.save(pending)
        pending.replace(path / f"epoch-{epoch}.keras")
        write_json(history_path, history)
        print(f"{strategy} epoch {epoch}: loss={loss:.4f}, validation F1={f1[pick]:.4f}, "
              f"mined={mined:,}/{len(examples):,}, {row['elapsed_seconds']:.1f}s", flush=True)

    selected = keras.models.load_model(path / f"epoch-{best['epoch']}.keras", compile=False)
    encoder = selected.get_layer("question_encoder")
    encoder.save(path / "encoder.keras")
    write_json(path / "vocabulary.json", vocabulary)
    write_json(path / "inference.json", {"config": asdict(config), "threshold": best["threshold"]})
    return best, history, encoder


def limited(frame, limit, seed):
    if limit is None or limit >= len(frame):
        return frame
    return frame.sample(limit, random_state=seed).sort_values("source_row")


def main():
    args = arguments()
    config = Config(**{key: getattr(args, key) for key in asdict(Config())})
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise SystemExit("Use a new output directory or pass --resume.")
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    keras.utils.set_random_seed(config.seed)
    splits, summary = prepare_splits(args.data, config.seed)
    splits = {
        name: limited(frame, args.train_limit if name == "train" else args.eval_limit, config.seed)
        for name, frame in splits.items()
    }
    indexed = {name: IndexedPairs.from_frame(frame) for name, frame in splits.items()}
    if any(np.unique(part.labels).size != 2 for part in indexed.values()):
        raise SystemExit("Each used split needs both labels. Increase the sample sizes.")
    pool = build_pool(indexed["train"])
    if not len(pool.anchors):
        raise SystemExit("No eligible anchors. Increase the number of training rows.")

    summary.update({
        "config": asdict(config), "strategies": args.strategies, "train_limit": args.train_limit,
        "eval_limit": args.eval_limit, "used_rows_by_split": {k: len(v) for k, v in splits.items()},
        "triplet_pool": pool.stats, "threads": args.threads,
        "warmup": args.warmup, "refresh": args.refresh, "patience": args.patience,
        "lr_patience": args.lr_patience, "tensorflow": tf.__version__, "keras": keras.__version__,
        "devices": [device.device_type for device in tf.config.list_physical_devices()],
    })
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "run_summary.json"
    if args.resume:
        previous = json.loads(manifest.read_text()) if manifest.is_file() else {}
        previous.pop("requested_epochs", None)
        if previous != summary:
            raise SystemExit("Resume requires the same dataset and options except --epochs.")
        root_vocabulary = args.output / "vocabulary.json"
        vocabulary = json.loads(root_vocabulary.read_text())
        tokenizer = make_tokenizer(config, vocabulary)
    else:
        tokenizer = fit_tokenizer(indexed["train"].questions, config)
        vocabulary = tokenizer.get_vocabulary()
        write_json(args.output / "vocabulary.json", vocabulary)

    write_json(manifest, {**summary, "requested_epochs": args.epochs})
    tokens = {name: tokenize(tokenizer, part.questions) for name, part in indexed.items()}
    print(f"V3 split rows: {summary['used_rows_by_split']}. Triplet pool: {pool.stats}", flush=True)
    trained = {}
    for strategy in args.strategies:
        path = args.output / strategy
        trained[strategy] = train_strategy(path, strategy, args, config, tokens, indexed,
                                           vocabulary, pool)

    # Freeze every strategy's checkpoint and threshold before using test predictions.
    selected = max(trained, key=lambda name: trained[name][0]["validation_f1"])
    write_json(args.output / "selected.json", {"strategy": selected})
    runs = {}
    for strategy, (best, history, encoder) in trained.items():
        scorer = CosineScorer(encoder, config.batch_size)
        validation_scores = scorer.score(tokens["validation"], indexed["validation"])
        start = time.perf_counter()
        test_vectors = scorer.embed(tokens["test"])
        test_scores = scorer.score_embeddings(test_vectors, indexed["test"])
        inference_seconds = time.perf_counter() - start
        count = min(256, len(test_scores))
        test = indexed["test"]
        reversed_pairs = IndexedPairs(test.questions, test.right[:count], test.left[:count],
                                      test.labels[:count])
        np.testing.assert_array_equal(
            test_scores[:count], scorer.score_embeddings(test_vectors, reversed_pairs)
        )
        pair_tokens = tokens["test"][[test.left[0], test.right[0]]]
        trials = []
        for _ in range(30):
            start = time.perf_counter()
            scorer.embed(pair_tokens)
            trials.append((time.perf_counter() - start) * 1e3)
        result = {
            "strategy": strategy, "best_epoch": best["epoch"], "epochs_run": len(history),
            "best_validation_f1": best["validation_f1"], "pairs": len(test_scores),
            "training_seconds": sum(row["elapsed_seconds"] for row in history),
            "test_inference_seconds": inference_seconds,
            "median_single_pair_encoding_ms": float(np.median(trials)),
            "trainable_parameters": int(sum(np.prod(w.shape) for w in encoder.trainable_weights)),
            "triplets_generated": sum(row["triplets"] for row in history),
            "triplets_mined": sum(row["mined_triplets"] for row in history),
            "swap_check_pairs": count,
            **metrics(test.labels, test_scores, best["threshold"]),
        }
        runs[strategy] = (history, test_scores, validation_scores, result)
        save_errors(args.output / strategy, splits["test"], test_scores, best["threshold"])
    table = pd.DataFrame([run[-1] for run in runs.values()])
    table.to_csv(args.output / "metrics.csv", index=False, float_format="%.8f")
    save_plots(args.output, indexed["validation"].labels, indexed["test"].labels, runs)
    print(table.round(5).to_string(index=False), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
