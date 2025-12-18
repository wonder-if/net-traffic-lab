"""
Streamlit-based explorer with two panes:
1) Semi-supervised USTC traffic experiments (reads outputs/ produced by semi_supervised_experiment.py).
2) Feature stability for CIC-IDS2017 (reads ids2017_feature_outputs/ produced by ids2017_feature_stability.py).

Run locally (needs streamlit, pandas, pillow):
  pip install streamlit pandas pillow
  streamlit run interactive_app.py
"""

import subprocess
import time
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from PIL import Image
import ids2017_explain_adv as iea


def load_image_safe(path: Path):
    if path.exists():
        return Image.open(path)
    return None


def run_command(cmd: str, label: str, expected_epochs: int | None = None):
    """
    Run a shell command with a progress indicator.
    If expected_epochs is provided, progress follows detected 'Epoch' lines.
    """
    import re

    prog = st.progress(0, text=f"Starting: {label}")
    last_pct = 0
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = []
        epoch_done = 0
        for line in proc.stdout:
            output.append(line.rstrip())
            if expected_epochs:
                if re.search(r"Epoch\\s+\\d+", line):
                    epoch_done += 1
                    pct = int(min(99, epoch_done / expected_epochs * 100))
                    prog.progress(
                        pct, text=f"{label}: {epoch_done}/{expected_epochs} epochs"
                    )
            else:
                last_pct = min(99, last_pct + 5)
                prog.progress(last_pct, text=f"{label}...")
        ret = proc.wait()
        prog.progress(100, text=f"Finished: {label} (exit {ret})")
        log_text = "\n".join(output[-200:])
        with st.expander(f"{label} 日志", expanded=False):
            st.code(log_text or "(no output)", language="bash")
        return ret == 0
    except Exception as e:
        prog.progress(0, text=f"Failed: {label}")
        st.error(f"Error running command: {e}")
        return False


def semi_supervised_view():
    st.header("Semi-Supervised Traffic Classification")
    outputs_dir = Path("outputs")

    mode = st.radio(
        "数据来源",
        ["使用已有结果", "立即运行并生成结果"],
        horizontal=True,
        key="semi_mode",
    )
    if "semi_loaded" not in st.session_state:
        st.session_state.semi_loaded = False

    if mode == "立即运行并生成结果":
        st.warning("将调用 semi_supervised_experiment.py 运行一次 sweep，可能耗时。")
        if st.button("立即生成 USTC 半监督结果", key="semi_run"):
            cmd = (
                "python semi_supervised_experiment.py "
                "--max-per-class 1000 --test-per-class 200 "
                "--unlabeled-cap-per-class 600 --epochs 8 "
                "--labeled-ratios 0.01,0.02,0.05,0.1,0.2,0.5"
            )
            ok = run_command(cmd, "运行半监督实验", expected_epochs=8 * 6)
            if not ok:
                st.error("运行失败，请检查日志或在终端单独执行。")
            else:
                st.success("生成完成，可以查看结果。")
    else:
        if st.button("加载已有结果", key="semi_load"):
            st.session_state.semi_loaded = True

    if not st.session_state.semi_loaded:
        st.info("点击上面的按钮后再显示已有结果。")
        return

    summary_path = outputs_dir / "summary_results.csv"
    if summary_path.exists():
        st.subheader("Aggregated sweep results")
        df = pd.read_csv(summary_path)
        st.dataframe(df)
        # Accuracy vs ratio
        if "ratio" in df.columns and "test_acc_best" in df.columns:
            try:
                df_plot = df.copy()
                df_plot["ratio"] = pd.to_numeric(df_plot["ratio"], errors="coerce")
                df_plot = df_plot.dropna(subset=["ratio"])
                st.line_chart(df_plot.set_index("ratio")["test_acc_best"], height=220)
            except Exception:
                pass
    else:
        st.info(
            "No summary_results.csv found in outputs/. Run semi_supervised_experiment.py first."
        )
        return

    run_dirs = sorted([p for p in outputs_dir.iterdir() if p.is_dir()])
    if not run_dirs:
        st.warning("No run subdirectories detected in outputs/.")
        return

    selected = st.selectbox(
        "Choose a run folder to inspect", run_dirs, format_func=lambda p: p.name
    )
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

    mode = st.radio(
        "数据来源",
        ["使用已有结果", "立即运行并生成结果"],
        horizontal=True,
        key="stable_mode",
    )
    if "stable_loaded" not in st.session_state:
        st.session_state.stable_loaded = False

    if mode == "立即运行并生成结果":
        st.warning(
            "将调用 ids2017_feature_stability.py，运行随机森林多种子分析，可能耗时。"
        )
        if st.button("立即生成 IDS2017 特征稳定性结果", key="stable_run"):
            cmd = (
                "python ids2017_feature_stability.py "
                "--data-path ids2017/data.csv "
                "--max-rows 50000 --n-estimators 150 --seeds 0,1,2 --test-size 0.25"
            )
            ok = run_command(cmd, "运行特征稳定性分析")
            if not ok:
                st.error("运行失败，请检查日志或在终端单独执行。")
            else:
                st.success("生成完成，可以查看结果。")
    else:
        if st.button("加载已有结果", key="stable_load"):
            st.session_state.stable_loaded = True

    if not st.session_state.stable_loaded:
        st.info("点击上面的按钮后再显示已有结果。")
        return

    feat_csv = out_dir / "feature_stability.csv"
    if feat_csv.exists():
        df = pd.read_csv(feat_csv)
        st.subheader("Per-feature stability")
        st.dataframe(
            df.sort_values("mean_importance", ascending=False).reset_index(drop=True)
        )

        st.subheader("Top feature pairs (co-occurrence in top-10)")
        pair_csv = out_dir / "stable_pairs.csv"
        if pair_csv.exists():
            df_pairs = pd.read_csv(pair_csv)
            st.dataframe(df_pairs.head(30))
        else:
            st.info("stable_pairs.csv not found.")
    else:
        st.info(
            "feature_stability.csv not found in ids2017_feature_outputs/. Run ids2017_feature_stability.py first."
        )
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
        st.image(
            corr_img, caption="Rank correlation across seeds", use_column_width=True
        )


def incremental_view():
    st.header("Incremental Learning (USTC)")
    data_dir = st.text_input("USTC 数据目录", value="ustc_all")
    model_path = st.text_input("增量模型 checkpoint", value="incremental/ustc_model")
    max_per_class = st.number_input(
        "每类最大样本（评估用）", min_value=200, max_value=5000, value=2000, step=100
    )

    if st.button("评估增量模型", key="inc_eval"):
        try:
            from incremental_eval import evaluate

            with st.spinner("加载并评估增量模型..."):
                acc, y_true, y_pred = evaluate(
                    Path(model_path),
                    Path(data_dir),
                    max_per_class=max_per_class,
                )
            st.success(f"增量模型精度: {acc:.4f}")
            st.write("样本数:", len(y_true))
        except Exception as e:
            st.error(f"评估失败: {e}")


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

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Semi-Supervised", "Stable Features", "Incremental (USTC)", "Explainability & Adversarial"]
    )
    with tab1:
        semi_supervised_view()
    with tab2:
        stability_view()
    with tab3:
        incremental_view()
    with tab4:
        explain_adv_view()


def explain_adv_view():
    st.header("Explainability & Adversarial (IDS2017)")
    data_path = st.text_input("数据路径", value="ids2017/data.csv")
    ckpt_path = st.text_input("模型 checkpoint 路径", value="ids2017/11ids2017.pt")
    max_rows = st.number_input(
        "最多加载行数", min_value=1000, max_value=80000, value=8000, step=1000
    )

    if "iea_loaded" not in st.session_state:
        st.session_state.iea_loaded = False
        st.session_state.iea_data = None
        st.session_state.iea_model = None
        st.session_state.iea_labels = None
        st.session_state.iea_feat_names = None

    if st.button("加载数据与模型", key="iea_load"):
        try:
            X, y, feat_names, label_names = iea.load_dataset(
                Path(data_path), max_rows=max_rows
            )
            model = iea.load_model(
                Path(ckpt_path), input_dim=X.shape[1], num_classes=len(label_names)
            )
            st.session_state.iea_data = (X, y)
            st.session_state.iea_model = model
            st.session_state.iea_labels = label_names
            st.session_state.iea_feat_names = feat_names
            st.session_state.iea_loaded = True
            st.success(
                f"已加载: 样本 {len(y)}, 特征 {len(feat_names)}, 类别 {len(label_names)}"
            )
        except Exception as e:
            st.error(f"加载失败: {e}")
            st.session_state.iea_loaded = False

    if not st.session_state.iea_loaded:
        st.info("请先点击上方按钮加载数据和模型。")
        return

    X, y = st.session_state.iea_data
    label_names = st.session_state.iea_labels
    feat_names = st.session_state.iea_feat_names
    model = st.session_state.iea_model

    idx = st.slider("选择样本索引", min_value=0, max_value=len(y) - 1, value=0, step=1)
    epsilon = st.slider(
        "FGSM epsilon", min_value=0.0, max_value=0.3, value=0.05, step=0.01
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("生成可解释性", key="iea_explain"):
            sample = X[idx]
            result = iea.predict_and_explain(model, sample, target_label=None)
            probs = iea.probs_to_table(result["probs"], label_names, top=5)
            topk = iea.topk_to_table(result["attribution"], feat_names, k=12)
            out_dir = Path("ids2017_explain_outputs")
            out_dir.mkdir(exist_ok=True)
            chart_path = iea.bar_chart_topk(
                topk, "Top attributions (clean)", out_dir / "clean_topk.png"
            )
            st.subheader("预测概率 Top5")
            st.table(probs)
            st.subheader("特征归因 Top12")
            st.table(topk)
            st.image(Image.open(chart_path), caption="归因柱状图")

    with col2:
        if st.button("生成对抗样本并解释", key="iea_adv"):
            sample = X[idx]
            true_label = int(y[idx])
            x_adv, clean_res, adv_res = iea.fgsm_attack(
                model, sample, true_label=true_label, epsilon=epsilon
            )
            out_dir = Path("ids2017_explain_outputs")
            out_dir.mkdir(exist_ok=True)
            clean_topk = iea.topk_to_table(clean_res["attribution"], feat_names, k=8)
            adv_topk = iea.topk_to_table(adv_res["attribution"], feat_names, k=8)
            chart_clean = iea.bar_chart_topk(
                clean_topk, "Clean top attributions", out_dir / "clean_adv.png"
            )
            chart_adv = iea.bar_chart_topk(
                adv_topk, "Adversarial top attributions", out_dir / "adv_adv.png"
            )

            st.subheader("干净样本预测 Top5")
            st.table(iea.probs_to_table(clean_res["probs"], label_names, top=5))
            st.subheader("对抗样本预测 Top5")
            st.table(iea.probs_to_table(adv_res["probs"], label_names, top=5))

            st.subheader("归因对比")
            st.image(Image.open(chart_clean), caption="干净样本归因")
            st.image(Image.open(chart_adv), caption="对抗样本归因")

            st.info(
                f"原标签: {label_names[true_label] if true_label < len(label_names) else true_label}; "
                f"干净预测: {label_names[clean_res['pred']] if clean_res['pred'] < len(label_names) else clean_res['pred']}; "
                f"对抗预测: {label_names[adv_res['pred']] if adv_res['pred'] < len(label_names) else adv_res['pred']}"
            )

    st.markdown("---")
    st.subheader("防御评估（批量检测率）")
    max_eval = st.slider("评估样本数", min_value=50, max_value=500, value=200, step=50)
    conf_drop = st.slider(
        "置信度下降阈值", min_value=0.0, max_value=1.0, value=0.3, step=0.05
    )
    linf_thresh = st.slider(
        "L∞ 阈值", min_value=0.0, max_value=0.3, value=0.05, step=0.01
    )
    if st.button("运行防御评估", key="iea_defense"):
        with st.spinner("批量生成对抗并检测..."):
            stats = iea.batch_attack_and_detect(
                model,
                X,
                y,
                epsilon=epsilon,
                max_samples=max_eval,
                conf_drop_thresh=conf_drop,
                linf_thresh=linf_thresh,
            )
        st.write(
            f"测试样本: {stats['tested']}, 攻击成功率: {stats['attack_success_rate']:.3f}, "
            f"检测率: {stats['detection_rate']:.3f}"
        )
        df_rec = pd.DataFrame(stats["records"])
        st.dataframe(df_rec.head(50))


if __name__ == "__main__":
    main()
