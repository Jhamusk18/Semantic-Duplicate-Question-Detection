"""Streamlit interface for pair classification."""

import os
from pathlib import Path

import joblib
import streamlit as st

from v2.inference import ARTIFACT_VERSION, predict_pair

ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def load_artifact(path, modified_ns):
    artifact = joblib.load(path)
    if artifact["version"] != ARTIFACT_VERSION:
        raise ValueError("Retrain the artifact with the current checkout.")
    return artifact


def render(profile="v2"):
    st.set_page_config(page_title="Duplicate Question Detection")
    st.title("Duplicate Question Detection")
    st.write("Compare two questions to see whether they ask for the same thing.")
    model_path = Path(os.environ.get("QUORA_MODEL_PATH", ROOT / "artifacts" / profile / "model.joblib"))
    try:
        artifact = load_artifact(model_path, model_path.stat().st_mtime_ns)
    except FileNotFoundError:
        command = "python -m v1.baseline" if profile == "v1" else "python -m v2.quora_v2_pipeline"
        st.warning(f"Train a model with `{command}` from the repository root to use this app.")
        return
    except Exception as exc:
        st.error(f"Could not load the model artifact. Retrain it with this checkout. {exc}")
        return

    with st.form("compare"):
        left = st.text_area("Question 1", max_chars=5_000)
        right = st.text_area("Question 2", max_chars=5_000)
        submitted = st.form_submit_button("Compare questions")
    if submitted:
        try:
            probability, duplicate = predict_pair(left, right, artifact)
        except ValueError as exc:
            st.warning(str(exc))
            return
        except RuntimeError as exc:
            st.error(f"Could not run the model. {exc}")
            return
        st.subheader("Duplicate" if duplicate else "Distinct")
        st.metric("Estimated duplicate probability", f"{probability:.2%}")
        st.caption(f"{artifact['experiment']} · validation decision threshold {artifact['threshold']:.2%}")


if __name__ == "__main__":
    render()
