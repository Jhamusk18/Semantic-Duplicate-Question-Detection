# Semantic Duplicate Question Detection

An NLP project for determining whether two questions express the same intent using the Quora Question Pairs dataset. This repository documents two iterations: a traditional machine-learning baseline and a revised pipeline emphasizing reliable evaluation, symmetric similarity features, and reproducibility.

## Dataset

[Quora Question Pairs — Kaggle](https://www.kaggle.com/c/quora-question-pairs): 404,290 labeled question pairs, approximately 37% duplicates. Download the data separately under the dataset's applicable terms.

## Project evolution

| Component | V1 — baseline | V2 — revised pipeline |
|---|---|---|
| Representation | Bag of words and handcrafted features | Word/character TF-IDF cosine similarity and optional sentence embeddings |
| Validation | Random pair split | Question-graph connected-component split |
| Features | Stacked question vectors and individual question features | Symmetric similarity and min/max features |
| Evaluation | Accuracy | Log loss, ROC-AUC, F1 and baseline comparisons |
| Experiments | Incremental feature engineering | Feature ablations and error analysis |
| Serving | Streamlit demo | Shared feature extraction for training and inference |

## Reported V1 results

The supplied baseline documentation reports the following accuracy on a 30,000-pair sample with a random 80/20 split. These values have not been independently reproduced here and are not directly comparable with V2's group-aware evaluation.

| Features | Random Forest | XGBoost |
|---|---:|---:|
| Bag of words | 74.2% | 73.3% |
| + Basic features | 76.8% | 76.5% |
| + Preprocessing and advanced features | 78.5% | 79.3% |

The majority-class accuracy baseline is approximately 63%.

## V2 evaluation

V2 is designed to evaluate generalization using a leakage-aware split, training-only vectorizer fitting, validation-based threshold selection, and an ablation study. Numerical V2 results are pending a completed training run; no results are claimed here.

## Intended repository layout

- `v1/`: EDA, baseline notebooks, handcrafted features, and original demo.
- `v2/`: revised training pipeline, Streamlit application, dependencies, and evaluation outputs.

## Quick start (once source files are uploaded)

```bash
cd v2
python -m pip install -r requirements.txt
python quora_v2_pipeline.py --data train.csv --sample 100000
streamlit run app_v2.py
```

Use `--embeddings` to include the optional sentence-embedding features.

## Limitations

Human-generated labels can be noisy; subsampling affects results; and a group-aware split may change the evaluation distribution. V2 is not a fine-tuned transformer cross-encoder or a Siamese LSTM implementation.

## Acknowledgments

The traditional NLP baseline was inspired by [CampusX's Quora Question Pairs tutorial](https://www.youtube.com/watch?v=1fvQU5yPjFs) and its [companion repository](https://github.com/campusx-official/quora-question-pairs). Preserve original attribution and applicable licensing when redistributing baseline code.
