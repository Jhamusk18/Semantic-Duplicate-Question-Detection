# Dataset: Quora Question Pairs

Source: [Quora Question Pairs competition on Kaggle](https://www.kaggle.com/competitions/quora-question-pairs/data).

Download the competition files after accepting the applicable rules. The raw competition data is not stored in this repository.

## Local setup

Extract `train.csv.zip` into this directory so that `data/train.csv` exists. The original competition training data contains 404,290 labeled question pairs. The competition test file is unlabeled and must not be used to calculate held-out accuracy or other labeled metrics.

Optional Kaggle CLI download (requires your own authenticated Kaggle setup):

```bash
kaggle competitions download -c quora-question-pairs -f train.csv.zip -p data
unzip data/train.csv.zip -d data
```

After the V2 pipeline is uploaded, run from the repository root using its documented command-line arguments, e.g. `python v2/quora_v2_pipeline.py --data data/train.csv`. Confirm the actual script options before execution.

Keep raw datasets and credentials out of Git. If you need to redistribute the competition files, first verify the applicable data redistribution terms; large files require Git LFS or a separate storage mechanism.

Attribution: Quora, Quora Question Pairs (2017), hosted by Kaggle.
