# Held-out test results

These aggregate reports come from runs on the full usable Quora Question Pairs training file. See the [project methodology](../README.md#architecture-and-evaluation) and `run_summary.json` for the split sizes, input SHA-256, exact run options, and model selection rule.

- [Ablation metrics and confusion counts](ablation_results.csv)
- [Confusion matrices](confusion_matrices.png)
- [ROC and precision–recall curves](evaluation_curves.png)
- [Run metadata](run_summary.json)

The training command also writes a local `error_analysis.csv` with source row numbers, raw questions, labels, and probabilities. The CSV and the fitted model stay in the output directory outside version control.

## MiniLM pilot

This separate comparison used 4,000 usable pairs from the same labeled file, with seed 42 and the same group-based splitting code. Both pilot runs used the same 2,800 training, 400 validation, and 800 test pairs, with up to 50 trees per model. The pretrained weights came from `@lat.md/embed-minilm-fp16@0.1.1` and were converted to FP32 for inference.

On the 800 test pairs, adding MiniLM to handcrafted features and TF-IDF yielded 82.50% accuracy, 0.7819 F1, and 0.9014 ROC-AUC. Adding LSA to the same feature set yielded 73.25% accuracy, 0.6899 F1, and 0.8207 ROC-AUC under the same pilot conditions. The table on the main project page reports a separate full-dataset run. [Pilot metrics](minilm_pilot.csv) · [MiniLM setup](../v2/README.md#use-a-local-copy-of-minilm)


## V3 full-data experiment

All three Siamese LSTM strategies and the V1/V2 baselines in the matched comparison used the same
80,854 held-out pairs. V3 chooses checkpoints and thresholds by validation F1. V1/V2 select
the classifier by validation log loss and tune its threshold on validation F1. The earlier
V1/V2 reports include ROC-AUC and log loss. V3 reports ROC-AUC and PR-AUC for its cosine scores.
[Matched comparison](v3/comparison.csv) · [Full MiniLM ablations](minilm_full.csv) · [V3 metrics](v3/metrics.csv) · [V3 explanation and commands](../v3/README.md)
