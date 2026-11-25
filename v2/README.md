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

### Use a local copy of MiniLM

After installing the same Python dependencies, pass a local model directory with
`--minilm-model-dir`. The directory needs its tokenizer, model configuration, and
weights. The app also needs access to this directory if the selected model uses MiniLM.

The Vercel Labs npm package below contains MiniLM weights stored in FP16. The pipeline
converts them to FP32 for CPU inference and uses mean pooling over up to 256 tokens.
The commands verify the archive hash and extract the model files without executing npm
package code.

```bash
mkdir -p artifacts/minilm-weights
curl -fsSL https://registry.npmjs.org/@lat.md/embed-minilm-fp16/-/embed-minilm-fp16-0.1.1.tgz \
    -o artifacts/minilm-weights/model.tgz
printf '%s  %s\n' \
    3a72901e5442be668701b468c764f83a27ca6af8a04b4ea85635d522e462f127 \
    artifacts/minilm-weights/model.tgz | sha256sum -c -
tar -xzf artifacts/minilm-weights/model.tgz -C artifacts/minilm-weights package/model
ln -s model.fp16.safetensors artifacts/minilm-weights/package/model/model.safetensors
HF_HUB_OFFLINE=1 python -m v2.quora_v2_pipeline --embeddings minilm \
    --minilm-model-dir artifacts/minilm-weights/package/model --output artifacts/minilm
```

[Vercel Labs model package](https://www.npmjs.com/package/@lat.md/embed-minilm-fp16)

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
