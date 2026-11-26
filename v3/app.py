"""Streamlit comparison backed by a trained Siamese LSTM."""

import json
import os
from pathlib import Path

import streamlit as st

from v3.inference import Predictor

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("encoder.keras", "inference.json", "vocabulary.json")


@st.cache_resource
def load_predictor(path, fingerprints):
    return Predictor(path)


def render():
    st.set_page_config(page_title="Siamese LSTM Question Comparison")
    st.title("Siamese LSTM Question Comparison")
    try:
        override = os.environ.get("QUORA_V3_MODEL_DIR")
        if override:
            directory = Path(override)
        else:
            store = ROOT / "artifacts" / "v3"
            selection = json.loads((store / "selected.json").read_text(encoding="utf-8"))
            strategy = selection["strategy"]
            if strategy not in {"random", "hard", "semi-hard"}:
                raise ValueError("Unknown strategy in the V3 selection file.")
            directory = store / strategy
        fingerprints = tuple((directory / filename).stat().st_mtime_ns for filename in REQUIRED)
        predictor = load_predictor(directory, fingerprints)
    except FileNotFoundError:
        st.warning("Train V3 with `python -m v3.train` or set `QUORA_V3_MODEL_DIR`.")
        return
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not load the trained V3 model. {exc}")
        return

    compare_path = Path(os.environ.get("QUORA_V2_MODEL_PATH", ROOT / "artifacts/v2/model.joblib"))
    compare = compare_path.is_file() and st.checkbox("Compare with V2")
    with st.form("questions"):
        first = st.text_area("Question 1", max_chars=5_000)
        second = st.text_area("Question 2", max_chars=5_000)
        submitted = st.form_submit_button("Compare questions")
    if not submitted:
        return
    try:
        similarity, duplicate, normalized = predictor.predict(first, second)
    except ValueError as exc:
        st.warning(str(exc))
        return

    st.subheader("Duplicate" if duplicate else "Distinct")
    st.metric("Cosine similarity", f"{similarity:.4f}")
    st.caption(f"Validation decision threshold {predictor.threshold:.4f}")
    for index, text in enumerate(normalized, start=1):
        st.write(f"Normalized question {index}")
        st.text(text)

    if compare:
        from v2.app_v2 import load_artifact
        from v2.inference import predict_pair

        try:
            artifact = load_artifact(compare_path, compare_path.stat().st_mtime_ns)
            probability, v2_duplicate = predict_pair(first, second, artifact)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            st.warning(f"Could not run the V2 comparison. {exc}")
        else:
            st.write(f"V2 prediction · {'Duplicate' if v2_duplicate else 'Distinct'}")
            st.metric("V2 estimated duplicate probability", f"{probability:.2%}")


if __name__ == "__main__":
    render()
