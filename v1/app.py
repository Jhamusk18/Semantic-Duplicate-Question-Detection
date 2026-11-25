"""Serve the maintained V1 model with the shared Streamlit interface."""

from v2.app_v2 import render

if __name__ == "__main__":
    render(profile="v1")
