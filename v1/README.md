# V1 baseline

CampusX's progression from bag-of-words to basic and advanced handcrafted features is available through one maintained training command. It replaces order-dependent concatenated vectors and the original random pair split with symmetric cosine similarities, train-only fitting, and component-disjoint splits. The app loads the same exported model and uses the same feature code as training.

```bash
python -m v1.baseline
python -m streamlit run v1/app.py
```

Run these commands from the repository root after following the [project setup](../README.md). The outputs go to `artifacts/v1/`. The [notebooks](notebooks/) cover training-set exploration and the three V1 experiments. Open them in a notebook environment using the same Python interpreter. Run the baseline before opening its result notebooks.

The original tutorial used a random 80/20 split on 30,000 pairs. Its reported accuracy figures use a different evaluation protocol. See [CampusX's original implementation](https://github.com/campusx-official/quora-question-pairs) and the [measured results for this repository](../results/).
