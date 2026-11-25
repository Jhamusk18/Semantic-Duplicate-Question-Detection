# V2 evaluation and serving

From the repository root, follow the [project setup](../README.md) and run

```bash
python -m v2.quora_v2_pipeline
python -m streamlit run v2/app_v2.py
```

The default full run compares the V1 baselines, handcrafted features, word/character TF-IDF, their combination, and their combination with training-only LSA sentence embeddings. `--no-embeddings` removes the last ablation. `--sample`, `--seed`, `--threads`, and `--max-trees` control run size and reproducibility. `--output` selects a new or empty directory.

## Optional MiniLM run

Install the CPU PyTorch wheel first when using a CPU-only Linux host. Loading the pinned model on first use requires access to Hugging Face and its model download endpoints, or an existing local Hugging Face cache.

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-minilm.txt
python -m v2.quora_v2_pipeline --embeddings minilm --output artifacts/minilm
```

## Run outputs

| File | Contents |
|---|---|
| `ablation_results.csv` | Held-out metrics and confusion counts for each experiment, alongside validation loss, threshold, and tree count |
| `run_summary.json` | Dataset SHA-256, filtering counts, split sizes and label counts, training options, and selected experiment |
| `evaluation_curves.png` | Test ROC and precision–recall curves |
| `confusion_matrices.png` | Test confusion matrices using each experiment's classification threshold |
| `error_analysis.csv` | Up to 200 actual misclassifications with labels, probability, error type, and confidence |
| `model.joblib` | Selected model, threshold, and required fitted feature transformers |

The app loads `artifacts/v2/model.joblib` by default. Only load model artifacts you trust. Keep the data and text-bearing error report local under the dataset's terms.
