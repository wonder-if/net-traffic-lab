"""
Streamlit-based explorer with two panes:
1) Semi-supervised USTC traffic experiments (reads outputs/ produced by semi_supervised_experiment.py).
2) Feature stability for CIC-IDS2017 (reads ids2017_feature_outputs/ produced by ids2017_feature_stability.py).

Run locally (needs streamlit, pandas, pillow):
  pip install streamlit pandas pillow
  streamlit run interactive_app.py
"""

from pathlib import Path
import json
import pandas as pd
import streamlit as st
from PIL import Image


def load_image_safe(path: Path):
    if path.exists():
        return Image.open(path)
    return None


def semi_supervised_view():
    st.header("Semi-Supervised Traffic Classification (USTC)")
    outputs_dir = Path("outputs")

    summary_path = outputs_dir / "summary_results.csv"
    if summary_path.exists():
        st.subheader("Aggregated sweep results")
        df = pd.read_csv(summary_path)
        st.dataframe(df)
    else:
        st.info("No summary_results.csv found in outputs/. Run semi_supervised_experiment.py first.")
        return

    run_dirs = sorted([p for p in outputs_dir.iterdir() if p.is_dir()])
    if not run_dirs:
        st.warning("No run subdirectories detected in outputs/.")
        return

    selected = st.selectbox("Choose a run folder to inspect", run_dirs, format_func=lambda p: p.name)
    metrics_path = selected / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        st.write("Metrics", metrics)
    else:
        st.warning(f"{metrics_path} missing.")

    col1, col2 = st.columns(2)
    with col1:
        img = load_image_safe(selected / "learning_curves.png")
        if img:
            st.image(img, caption="Learning curves")
    with col2:
        img = load_image_safe(selected / "confusion_matrix.png")
        if img:
            st.image(img, caption="Confusion matrix")

    tsne_img = load_image_safe(selected / "tsne.png")
    if tsne_img:
        st.image(tsne_img, caption="t-SNE of learned features", use_column_width=True)


def stability_view():
    st.header("Stable Feature Extraction (CIC-IDS2017)")
    out_dir = Path("ids2017_feature_outputs")

    feat_csv = out_dir / "feature_stability.csv"
    if feat_csv.exists():
        df = pd.read_csv(feat_csv)
        st.subheader("Per-feature stability")
        st.dataframe(df.sort_values("mean_importance", ascending=False).reset_index(drop=True))

        st.subheader("Top feature pairs (co-occurrence in top-10)")
        pair_csv = out_dir / "stable_pairs.csv"
        if pair_csv.exists():
            df_pairs = pd.read_csv(pair_csv)
            st.dataframe(df_pairs.head(30))
        else:
            st.info("stable_pairs.csv not found.")
    else:
        st.info("feature_stability.csv not found in ids2017_feature_outputs/. Run ids2017_feature_stability.py first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        img = load_image_safe(out_dir / "top_features.png")
        if img:
            st.image(img, caption="Top features by mean importance")
    with col2:
        img = load_image_safe(out_dir / "top_frequency.png")
        if img:
            st.image(img, caption="Most frequent features in top-20")

    corr_img = load_image_safe(out_dir / "rank_correlation.png")
    if corr_img:
        st.image(corr_img, caption="Rank correlation across seeds", use_column_width=True)


def main():
    st.set_page_config(page_title="Network Traffic Lab", layout="wide")
    st.title("Network Traffic Learning Lab")
    st.markdown(
        """
        *Two academic showcases built on your existing pipelines.*
        - **Semi-supervised classification** (USTC): explore sweeps, metrics, and visuals.
        - **Stable feature extraction** (CIC-IDS2017): per-feature stability tables and pair co-occurrence.
        """
    )

    tab1, tab2 = st.tabs(["Semi-Supervised", "Stable Features"])
    with tab1:
        semi_supervised_view()
    with tab2:
        stability_view()


if __name__ == "__main__":
    main()
