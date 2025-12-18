import argparse
import os
import random
from pathlib import Path
from typing import Dict, List

# Limit thread usage to avoid SHM issues in constrained environments.
for env_var in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(env_var, "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_class_file(path: Path, label: int, max_samples: int, seed: int) -> np.ndarray:
    rows = []
    rng = np.random.default_rng(seed)
    # Reservoir sampling to cap rows without loading everything if the file is large.
    with path.open("r") as f:
        for i, line in enumerate(f):
            cells = [c.strip() for c in line.split(",") if c.strip() != ""]
            vals = [int(c, 16) for c in cells]
            if len(rows) < max_samples:
                rows.append(vals)
            else:
                j = rng.integers(0, i + 1)
                if j < max_samples:
                    rows[j] = vals

    arr = np.array(rows, dtype=np.float32) / 255.0
    labels = np.full((arr.shape[0],), label, dtype=np.int64)
    return arr, labels


def load_dataset(data_dir: Path, max_per_class: int, seed: int) -> Dict[str, np.ndarray]:
    X_list, y_list = [], []
    class_names = {}
    for csv_path in sorted(data_dir.glob("*.csv")):
        class_id = int(csv_path.stem)
        class_names[class_id] = csv_path.stem
        Xi, yi = load_class_file(csv_path, class_id, max_per_class, seed)
        X_list.append(Xi)
        y_list.append(yi)
        print(f"Loaded {Xi.shape[0]} samples for class {class_id}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return {"X": X, "y": y, "class_names": class_names}


def run_importance_experiments(
    X: np.ndarray,
    y: np.ndarray,
    seeds: List[int],
    n_estimators: int,
    test_size: float,
) -> Dict:
    results = []
    accs = []
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
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        accs.append(acc)
        results.append({"seed": seed, "importance": clf.feature_importances_})
        print(f"Seed {seed}: val acc={acc:.4f}")

    return {"runs": results, "val_accs": accs}


def summarize_importance(runs: List[Dict]) -> Dict[str, np.ndarray]:
    imp_matrix = np.stack([r["importance"] for r in runs], axis=0)
    mean = imp_matrix.mean(axis=0)
    std = imp_matrix.std(axis=0)
    ranks = np.argsort(np.argsort(-imp_matrix, axis=1), axis=1) + 1  # rank 1 = most important
    mean_rank = ranks.mean(axis=0)
    topk_freq = {}
    for k in [5, 10, 20]:
        topk = np.argsort(-imp_matrix, axis=1)[:, :k]
        freq = np.zeros(imp_matrix.shape[1], dtype=int)
        for row in topk:
            freq[row] += 1
        topk_freq[k] = freq
    return {"mean": mean, "std": std, "mean_rank": mean_rank, "topk_freq": topk_freq}


def save_bar_chart(
    values: np.ndarray,
    errors: np.ndarray,
    labels: List[str],
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    width, height = 900, 500
    margin = 80
    bar_width = (width - 2 * margin) / max(len(values), 1)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    max_val = float(values.max()) if len(values) else 1.0
    y_scale = (height - 2 * margin) / max_val if max_val > 0 else 1.0

    # Axes
    draw.line((margin, margin, margin, height - margin), fill="black", width=2)
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)

    for i, v in enumerate(values):
        x0 = margin + i * bar_width + 10
        x1 = x0 + bar_width - 20
        y1 = height - margin
        y0 = y1 - v * y_scale
        draw.rectangle([x0, y0, x1, y1], fill="#4C72B0")
        # Error bars if provided
        if errors is not None:
            err = errors[i]
            yerr_top = y1 - (v + err) * y_scale
            yerr_bot = y1 - max(v - err, 0) * y_scale
            draw.line(( (x0+x1)/2, yerr_top, (x0+x1)/2, yerr_bot ), fill="gray", width=2)
            draw.line((x0+5, yerr_top, x1-5, yerr_top), fill="gray", width=2)
            draw.line((x0+5, yerr_bot, x1-5, yerr_bot), fill="gray", width=2)
        draw.text((x0, y1 + 5), labels[i], fill="black", font=font)
        draw.text((x0, y0 - 12), f"{v:.3f}", fill="black", font=font)

    draw.text((margin, margin - 30), title, fill="black", font=font)
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
    width = margin + n * cell + 40
    height = margin + n * cell + 80
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    def color_map(val: float) -> tuple:
        # map vmin..vmax to blue-white-red
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
            draw.text((x0 + 8, y0 + 20), f"{val:.2f}", fill="black", font=font)

    for i, lbl in enumerate(labels):
        draw.text((margin + i * cell + 15, margin - 20), lbl, fill="black", font=font)
        draw.text((margin - 40, margin + i * cell + 20), lbl, fill="black", font=font)

    draw.text((margin, 30), title, fill="black", font=font)
    draw.text((margin, margin + n * cell + 20), f"Scale: [{vmin:.1f}, {vmax:.1f}]", fill="black", font=font)
    img.save(path)


def plot_top_features(stats: Dict[str, np.ndarray], top_k: int, out_dir: Path) -> None:
    mean = stats["mean"]
    std = stats["std"]
    idx = np.argsort(-mean)[:top_k]
    labels = [f"f{j}" for j in idx]
    values = mean[idx]
    errors = std[idx]
    save_bar_chart(
        values=values,
        errors=errors,
        labels=labels,
        title=f"Top {top_k} stable features (mean ± std over seeds)",
        ylabel="Importance",
        path=out_dir / f"top_{top_k}_features.png",
    )


def plot_rank_corr(runs: List[Dict], out_dir: Path) -> None:
    imp_matrix = np.stack([r["importance"] for r in runs], axis=0)
    order = np.argsort(-imp_matrix, axis=1)
    n = len(runs)
    corr = np.zeros((n, n))
    for i in range(n):
        corr[i, i] = 1.0

    def spearman_manual(x: np.ndarray, y: np.ndarray) -> float:
        # Compute Spearman rho without heavy BLAS use.
        rx = np.argsort(np.argsort(x))  # ranks starting at 0
        ry = np.argsort(np.argsort(y))
        rx = rx.astype(np.float64)
        ry = ry.astype(np.float64)
        mx, my = rx.mean(), ry.mean()
        num = ((rx - mx) * (ry - my)).sum()
        den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum() + 1e-8)
        return float(num / den) if den > 0 else 0.0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rho = spearman_manual(order[i], order[j])
            corr[i, j] = rho
    save_heatmap(
        corr,
        labels=[str(r["seed"]) for r in runs],
        title="Spearman rank correlation (importance ranks)",
        path=out_dir / "rank_correlation.png",
        vmin=-1,
        vmax=1,
    )


def plot_topk_frequency(stats: Dict[str, np.ndarray], k: int, out_dir: Path) -> None:
    freq = stats["topk_freq"][k]
    idx = np.argsort(-freq)[:k]
    labels = [f"f{j}" for j in idx]
    values = freq[idx]
    save_bar_chart(
        values=values,
        errors=None,
        labels=labels,
        title=f"Features most frequently in top-{k} (across runs)",
        ylabel=f"Count in top-{k}",
        path=out_dir / f"topk_frequency_{k}.png",
    )


def main():
    parser = argparse.ArgumentParser(description="Stable feature extraction on USTC traffic data.")
    parser.add_argument("--data-dir", type=Path, default=Path("ustc_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("feature_outputs"))
    parser.add_argument("--max-per-class", type=int, default=800, help="Max samples per class to load.")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seeds for stability runs.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation fraction.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    set_seed(seeds[0] if seeds else 0)

    print("Loading data...")
    data = load_dataset(args.data_dir, args.max_per_class, seeds[0])
    X, y = data["X"], data["y"]
    print(f"Total samples: {len(y)}, features: {X.shape[1]}")

    exp = run_importance_experiments(
        X,
        y,
        seeds=seeds,
        n_estimators=args.n_estimators,
        test_size=args.test_size,
    )
    stats = summarize_importance(exp["runs"])

    # Save summary CSV
    mean, std, mean_rank = stats["mean"], stats["std"], stats["mean_rank"]
    top5 = np.argsort(-mean)[:5]
    summary_path = args.output_dir / "stable_features.csv"
    with summary_path.open("w") as f:
        f.write("feature,mean_importance,std_importance,mean_rank\n")
        for i in range(len(mean)):
            f.write(f"f{i},{mean[i]:.6f},{std[i]:.6f},{mean_rank[i]:.2f}\n")
    print(f"Saved detailed feature stats to {summary_path.resolve()}")

    # Plots
    plot_top_features(stats, top_k=20, out_dir=args.output_dir)
    plot_rank_corr(exp["runs"], out_dir=args.output_dir)
    plot_topk_frequency(stats, k=20, out_dir=args.output_dir)

    print("Top 5 stable features (by mean importance):")
    for j in top5:
        print(
            f"  f{j}: mean_importance={mean[j]:.4f}, std={std[j]:.4f}, mean_rank={mean_rank[j]:.2f}"
        )
    print(f"Validation accuracies across runs: {[round(a,4) for a in exp['val_accs']]}")


if __name__ == "__main__":
    main()
