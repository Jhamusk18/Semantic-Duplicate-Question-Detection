# Quora Question Pairs dataset

Download the labeled `train.csv.zip` after accepting the competition rules on Kaggle. Extract it to `data/train.csv`. The original labeled file contains 404,290 pairs. [Quora Question Pairs data](https://www.kaggle.com/competitions/quora-question-pairs/data)

```bash
unzip -p /path/to/train.csv.zip train.csv > data/train.csv
```

The CSV needs `qid1`, `qid2`, `question1`, `question2`, and `is_duplicate`. The training pipeline uses the original row position for error analysis. It filters out pairs whose normalized question text is empty. The separate competition test file is unlabeled.

Keep the raw data and any Kaggle credentials out of Git. See the competition terms for applicable data use and redistribution rules.
