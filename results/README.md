# Held-out test results

These aggregate reports come from runs on the full usable Quora Question Pairs training file. See the [project methodology](../README.md#architecture-and-evaluation) and `run_summary.json` for the split sizes, input SHA-256, exact run options, and model selection rule.

- [Ablation metrics and confusion counts](ablation_results.csv)
- [Confusion matrices](confusion_matrices.png)
- [ROC and precision–recall curves](evaluation_curves.png)
- [Run metadata](run_summary.json)

The training command also writes a local `error_analysis.csv` with source row numbers, raw questions, labels, and probabilities. The CSV and the fitted model stay in the output directory outside version control.
