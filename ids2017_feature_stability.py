import os
import argparse
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

# Keep threads low to avoid shared-memory issues.
for env_var in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(env_var, "1")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(path: Path, max_rows: int, drop_labels: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(path, encoding="latin1")
    if drop_labels:
        df = df[~df["Label"].isin(drop_labels)]

    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=0)

    labels_raw, label_names = pd.factorize(df["Label"])
    X_df = df.drop(columns=["Label"])

    # Fill missing values with column medians
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X_df.values.astype(np.float32))
    y = labels_raw.astype(np.int64)
    feature_names = list(X_df.columns)
    return X, y, feature_names


def run_rf_importance(
    X: np.ndarray,
    y: np.ndarray,
    seeds: List[int],
    n_estimators: int,
    test_size: float,
) -> Dict:
    runs, accs = [], []
    for seed in seeds:
        set_seed(seed)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            n_jobs=1,
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        accs.append(acc)
        runs.append({"seed": seed, "importance": clf.feature_importances_})
        print(f"Seed {seed}: val acc={acc:.4f}")
    return {"runs": runs, "val_accs": accs}


def summarize_importance(runs: List[Dict]) -> Dict[str, np.ndarray]:
    imp = np.stack([r["importance"] for r in runs], axis=0)
    mean = imp.mean(axis=0)
    std = imp.std(axis=0)
    ranks = np.argsort(np.argsort(-imp, axis=1), axis=1) + 1
    mean_rank = ranks.mean(axis=0)

    topk_freq = {}
    for k in [5, 10, 20]:
        freq = np.zeros(imp.shape[1], dtype=int)
        topk = np.argsort(-imp, axis=1)[:, :k]
        for row in topk:
            freq[row] += 1
        topk_freq[k] = freq

    pair_freq = {}
    for k in [10]:
        comb_counter: Counter = Counter()
        topk = np.argsort(-imp, axis=1)[:, :k]
        for row in topk:
            for a, b in combinations(sorted(row.tolist()), 2):
                comb_counter[(a, b)] += 1
        pair_freq[k] = comb_counter

    return {
        "mean": mean,
        "std": std,
        "mean_rank": mean_rank,
        "topk_freq": topk_freq,
        "pair_freq": pair_freq,
    }


def save_bar_chart(
    values: np.ndarray,
    errors: np.ndarray,
    labels: List[str],
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    width, height = 1100, 900  # extra height reserved for rotated labels
    margin = 120
    bar_width = max((width - 2 * margin) / max(len(values), 1), 10)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    max_val = float(values.max()) if len(values) else 1.0
    y_scale = (height - 2 * margin) / max(max_val, 1e-6)

    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)

    def draw_rotated_label(base_img: Image.Image, text: str, x: float, y: float, angle: int = 30):
        # Render label on a transparent canvas, then rotate and paste.
        canvas = Image.new("RGBA", (260, 120), (255, 255, 255, 0))
        d = ImageDraw.Draw(canvas)
        d.text((0, 0), text, fill="black", font=font)
        rotated = canvas.rotate(angle, expand=True)
        base_img.paste(rotated, (int(x), int(y)), rotated)

    for i, v in enumerate(values):
        x0 = margin + i * bar_width + 6
        x1 = x0 + bar_width - 12
        y1 = height - margin
        y0 = y1 - v * y_scale
        draw.rectangle([x0, y0, x1, y1], fill="#4C72B0")
        if errors is not None:
            err = errors[i]
            yerr_top = y1 - (v + err) * y_scale
            yerr_bot = y1 - max(v - err, 0) * y_scale
            draw.line(((x0 + x1) / 2, yerr_top, (x0 + x1) / 2, yerr_bot), fill="gray", width=2)
            draw.line((x0 + 5, yerr_top, x1 - 5, yerr_top), fill="gray", width=2)
            draw.line((x0 + 5, yerr_bot, x1 - 5, yerr_bot), fill="gray", width=2)
        # Tilt labels to avoid overlap; full text kept.
        draw_rotated_label(img, labels[i], x0 - 10, y1 + 20, angle=45)
        draw.text((x0, y0 - 14), f"{v:.3f}", fill="black", font=font)

    draw.text((margin, margin - 40), title, fill="black", font=font)
    draw.text((10, (height // 2)), ylabel, fill="black", font=font)
    img.save(path)


def save_heatmap(
    matrix: np.ndarray,
    labels: List[str],
    title: str,
    path: Path,
    vmin: float,
    vmax: float,
) -> None:
    n = matrix.shape[0]
    cell = 60
    margin = 120
    width = margin + n * cell + 50
    height = margin + n * cell + 90
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    def color_map(val: float) -> tuple:
        ratio = (val - vmin) / (vmax - vmin + 1e-8)
        r = int(255 * ratio)
        b = int(255 * (1 - ratio))
        g = int(255 * min(ratio, 1 - ratio) * 2)
        return (r, g, b)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = color_map(val)
            x0 = margin + j * cell
            y0 = margin + i * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=color, outline="black")
            draw.text((x0 + 10, y0 + 20), f"{val:.2f}", fill="black", font=font)

    for i, lbl in enumerate(labels):
        draw.text((margin + i * cell + 15, margin - 25), lbl, fill="black", font=font)
        draw.text((margin - 45, margin + i * cell + 20), lbl, fill="black", font=font)

    draw.text((margin, 35), title, fill="black", font=font)
    draw.text((margin, margin + n * cell + 25), f"Scale: [{vmin:.1f}, {vmax:.1f}]", fill="black", font=font)
    img.save(path)


def plot_rank_corr(runs: List[Dict], out_dir: Path) -> None:
    imp = np.stack([r["importance"] for r in runs], axis=0)
    order = np.argsort(-imp, axis=1)
    n = len(runs)
    corr = np.zeros((n, n))
    for i in range(n):
        corr[i, i] = 1.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rx = np.argsort(np.argsort(order[i])).astype(np.float64)
            ry = np.argsort(np.argsort(order[j])).astype(np.float64)
            mx, my = rx.mean(), ry.mean()
            num = ((rx - mx) * (ry - my)).sum()
            den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum() + 1e-8)
            corr[i, j] = float(num / den) if den > 0 else 0.0
    save_heatmap(
        corr,
        labels=[str(r["seed"]) for r in runs],
        title="Spearman rank correlation (importance ranks)",
        path=out_dir / "rank_correlation.png",
        vmin=-1,
        vmax=1,
    )


def save_pair_table(pair_counter: Counter, feature_names: List[str], k: int, path: Path) -> None:
    most_common = pair_counter.most_common(k)
    with path.open("w") as f:
        f.write("feature_a,feature_b,count\n")
        for (a, b), c in most_common:
            f.write(f"{feature_names[a]},{feature_names[b]},{c}\n")


def main():
    parser = argparse.ArgumentParser(description="Stability analysis of IDS2017 features (random forest importances).")
    parser.add_argument("--data-path", type=Path, default=Path("ids2017/data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("ids2017_feature_outputs"))
    parser.add_argument("--max-rows", type=int, default=50000, help="Cap rows for speed.")
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds.")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--drop-labels",
        type=str,
        default="Web Attack �XSS,Infiltration,Web Attack �Sql Injection,Heartbleed",
        help="Comma-separated labels to drop (rare/noisy).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    drop_labels = [s for s in args.drop_labels.split(",") if s.strip()]

    print("Loading dataset...")
    X, y, feat_names = load_dataset(args.data_path, args.max_rows, drop_labels)
    print(f"Loaded {len(y)} rows, {len(feat_names)} features.")

    exp = run_rf_importance(
        X,
        y,
        seeds=seeds,
        n_estimators=args.n_estimators,
        test_size=args.test_size,
    )
    stats = summarize_importance(exp["runs"])

    # Save per-feature stats
    feat_csv = args.output_dir / "feature_stability.csv"
    with feat_csv.open("w") as f:
        f.write("feature,mean_importance,std_importance,mean_rank,top5_freq,top10_freq,top20_freq\n")
        for i, name in enumerate(feat_names):
            f.write(
                f"{name},{stats['mean'][i]:.6f},{stats['std'][i]:.6f},{stats['mean_rank'][i]:.2f},"
                f"{stats['topk_freq'][5][i]},{stats['topk_freq'][10][i]},{stats['topk_freq'][20][i]}\n"
            )

    # Top features chart
    top_k = 20
    idx = np.argsort(-stats["mean"])[:top_k]
    save_bar_chart(
        values=stats["mean"][idx],
        errors=stats["std"][idx],
        labels=[feat_names[i] for i in idx],
        title=f"Top {top_k} stable features (mean ± std)",
        ylabel="Importance",
        path=args.output_dir / "top_features.png",
    )

    # Frequency chart
    freq_idx = np.argsort(-stats["topk_freq"][20])[:top_k]
    save_bar_chart(
        values=stats["topk_freq"][20][freq_idx],
        errors=None,
        labels=[feat_names[i] for i in freq_idx],
        title="Features most frequently in top-20 (across seeds)",
        ylabel="Count",
        path=args.output_dir / "top_frequency.png",
    )

    # Rank correlation across seeds
    plot_rank_corr(exp["runs"], args.output_dir)

    # Pair stability (co-occurrence in top-10)
    pair_counter = stats["pair_freq"][10]
    pair_csv = args.output_dir / "stable_pairs.csv"
    save_pair_table(pair_counter, feat_names, k=50, path=pair_csv)

    print(f"Saved feature stats to {feat_csv}")
    print(f"Validation accuracies: {[round(a,4) for a in exp['val_accs']]}")


if __name__ == "__main__":
    main()
