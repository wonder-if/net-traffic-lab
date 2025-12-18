import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple

# Constrain OpenMP usage to avoid shared memory errors in restricted environments.
for env_var in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(env_var, "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_class_file(
    path: Path,
    label: int,
    max_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one CSV file and return a sample of rows with numeric features.
    Values are hex strings, so we parse base16 and normalize to [0, 1].
    """
    rows = []
    with path.open("r") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            # CSV values are hex strings separated by commas and padded with whitespace.
            cells = [c.strip() for c in line.split(",") if c.strip() != ""]
            vals = [int(c, 16) for c in cells]
            rows.append(vals)

    arr = np.array(rows, dtype=np.float32) / 255.0
    labels = np.full((arr.shape[0],), label, dtype=np.int64)
    return arr, labels


def build_dataset(
    data_dir: Path,
    max_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
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
    return X, y, class_names


def split_labeled_unlabeled(
    X: np.ndarray,
    y: np.ndarray,
    labeled_per_class: int,
    test_per_class: int,
    unlabeled_cap_per_class: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    parts = {
        "labeled_X": [],
        "labeled_y": [],
        "unlabeled_X": [],
        "unlabeled_y": [],
        "test_X": [],
        "test_y": [],
    }

    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)

        test_take = min(test_per_class, len(idx))
        test_idx = idx[:test_take]
        remaining = idx[test_take:]

        labeled_take = min(labeled_per_class, len(remaining))
        labeled_idx = remaining[:labeled_take]

        unlabeled_idx = remaining[labeled_take:]
        if unlabeled_cap_per_class is not None:
            unlabeled_idx = unlabeled_idx[:unlabeled_cap_per_class]

        parts["test_X"].append(X[test_idx])
        parts["test_y"].append(y[test_idx])
        parts["labeled_X"].append(X[labeled_idx])
        parts["labeled_y"].append(y[labeled_idx])
        parts["unlabeled_X"].append(X[unlabeled_idx])
        parts["unlabeled_y"].append(y[unlabeled_idx])

    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


class LabeledDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class UnlabeledDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class SimpleNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, return_features: bool = False):
        f = self.features(x)
        logits = self.classifier(f)
        if return_features:
            return logits, f
        return logits


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(yb.numpy())
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred


def train_semisupervised(
    model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    lambda_u: float,
    threshold: float,
) -> Dict[str, list]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"sup_loss": [], "unsup_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        sup_losses, unsup_losses = [], []
        labeled_iter = iter(labeled_loader)

        for xb_u in unlabeled_loader:
            try:
                xb_l, yb_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                xb_l, yb_l = next(labeled_iter)

            xb_l, yb_l = xb_l.to(device), yb_l.to(device)
            xb_u = xb_u.to(device)

            logits_l = model(xb_l)
            sup_loss = criterion(logits_l, yb_l)

            logits_u = model(xb_u)
            prob_u = torch.softmax(logits_u.detach(), dim=1)
            max_prob, pseudo = prob_u.max(dim=1)
            mask = max_prob.ge(threshold).float()

            if mask.sum() > 0:
                unsup_loss = (
                    F.cross_entropy(logits_u, pseudo, reduction="none") * mask
                ).sum() / mask.sum()
            else:
                unsup_loss = torch.tensor(0.0, device=device)

            loss = sup_loss + lambda_u * unsup_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sup_losses.append(sup_loss.item())
            unsup_losses.append(unsup_loss.item())

        test_acc, _, _ = evaluate(model, val_loader, device)
        history["sup_loss"].append(float(np.mean(sup_losses)))
        history["unsup_loss"].append(float(np.mean(unsup_losses) if unsup_losses else 0.0))
        history["test_acc"].append(test_acc)
        print(
            f"Epoch {epoch:02d} | sup_loss={history['sup_loss'][-1]:.4f} "
            f"unsup_loss={history['unsup_loss'][-1]:.4f} test_acc={test_acc:.4f}"
        )
    return history


def plot_learning_curves(history: Dict[str, list], output_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history["test_acc"]) + 1)
    ax1.plot(epochs, history["sup_loss"], label="Supervised loss")
    ax1.plot(epochs, history["unsup_loss"], label="Unsupervised loss", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["test_acc"], color="green", label="Test accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=200)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: Dict[int, str], output_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=sorted(class_names))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=[class_names[i] for i in sorted(class_names)],
        yticklabels=[class_names[i] for i in sorted(class_names)],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def plot_tsne(model: nn.Module, loader: DataLoader, class_names: Dict[int, str], device: torch.device, output_dir: Path) -> None:
    model.eval()
    embeddings, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            _, feat = model(xb, return_features=True)
            embeddings.append(feat.cpu().numpy())
            targets.append(yb.numpy())

    X_embed = np.concatenate(embeddings, axis=0)
    y = np.concatenate(targets, axis=0)

    # Use a manageable subset for TSNE
    if len(y) > 2000:
        idx = np.random.default_rng(0).choice(len(y), 2000, replace=False)
        X_embed = X_embed[idx]
        y = y[idx]

    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=0)
    coords = tsne.fit_transform(X_embed)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap="tab20", s=8, alpha=0.8)
    legend1 = ax.legend(*scatter.legend_elements(num=20), title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.add_artist(legend1)
    ax.set_title("t-SNE of learned features")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_dir / "tsne.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Semi-supervised experiment for USTC traffic data.")
    parser.add_argument("--data-dir", type=Path, default=Path("ustc_all"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-per-class", type=int, default=3000, help="Max samples to load per class.")
    parser.add_argument(
        "--labeled-per-class",
        type=int,
        default=80,
        help="Number of labeled samples per class (used if --labeled-ratios is not set).",
    )
    parser.add_argument(
        "--labeled-ratios",
        type=str,
        default="0.01,0.02,0.05,0.1,0.2,0.5",
        help="Comma-separated fractions of per-class data (post test split) to label, e.g. '0.05,0.1'. "
        "Overrides --labeled-per-class when provided.",
    )
    parser.add_argument("--test-per-class", type=int, default=200, help="Held-out test samples per class.")
    parser.add_argument("--unlabeled-cap-per-class", type=int, default=800, help="Cap on unlabeled samples per class.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-u", type=float, default=0.7, dest="lambda_u")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    X, y, class_names = build_dataset(args.data_dir, args.max_per_class, args.seed)
    ratios = None
    if args.labeled_ratios:
        ratios = [float(r.strip()) for r in args.labeled_ratios.split(",") if r.strip()]
    else:
        ratios = [None]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    summary = []
    for ratio in ratios:
        if ratio is None:
            labeled_per_class = args.labeled_per_class
            tag = f"fixed_{labeled_per_class}"
        else:
            labeled_per_class = max(
                1, int((args.max_per_class - args.test_per_class) * ratio)
            )
            tag = f"ratio_{ratio:.3f}"

        run_dir = args.output_dir / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Run {tag} | labeled_per_class={labeled_per_class} ===")

        splits = split_labeled_unlabeled(
            X,
            y,
            labeled_per_class=labeled_per_class,
            test_per_class=args.test_per_class,
            unlabeled_cap_per_class=args.unlabeled_cap_per_class,
            seed=args.seed,
        )

        print(
            f"Labeled: {len(splits['labeled_y'])}, "
            f"Unlabeled: {len(splits['unlabeled_X'])}, "
            f"Test: {len(splits['test_y'])}"
        )

        labeled_ds = LabeledDataset(splits["labeled_X"], splits["labeled_y"])
        unlabeled_ds = UnlabeledDataset(splits["unlabeled_X"])
        test_ds = LabeledDataset(splits["test_X"], splits["test_y"])

        labeled_loader = DataLoader(labeled_ds, batch_size=128, shuffle=True, drop_last=True)
        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=256, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

        model = SimpleNet(input_dim=X.shape[1], num_classes=len(class_names)).to(device)

        history = train_semisupervised(
            model,
            labeled_loader,
            unlabeled_loader,
            test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            lambda_u=args.lambda_u,
            threshold=args.threshold,
        )

        test_acc, y_true, y_pred = evaluate(model, test_loader, device)
        best_acc = float(max(history["test_acc"])) if history["test_acc"] else float(test_acc)
        print(f"Final test accuracy ({tag}): {test_acc:.4f}")
        with open(run_dir / "metrics.json", "w") as f:
            json.dump({"test_accuracy": test_acc, **history}, f, indent=2)

        summary.append(
            {
                "tag": tag,
                "ratio": ratio,
                "labeled_per_class": labeled_per_class,
                "test_acc_final": float(test_acc),
                "test_acc_best": best_acc,
            }
        )

        plot_learning_curves(history, run_dir)
        plot_confusion(y_true, y_pred, class_names, run_dir)
        plot_tsne(model, test_loader, class_names, device, run_dir)
        print(f"Outputs written to {run_dir.resolve()}")

    # Summaries across ratios
    if len(summary) > 1:
        summary_sorted = sorted(
            summary,
            key=lambda r: r["ratio"] if r["ratio"] is not None else -1.0,
        )
        table_path = args.output_dir / "summary_results.csv"
        with table_path.open("w") as f:
            f.write("tag,ratio,labeled_per_class,test_acc_final,test_acc_best\n")
            for r in summary_sorted:
                f.write(
                    f"{r['tag']},{r['ratio']},{r['labeled_per_class']},"
                    f"{r['test_acc_final']:.4f},{r['test_acc_best']:.4f}\n"
                )

        # Accuracy vs labeled ratio plot
        xs = [r["ratio"] if r["ratio"] is not None else 0 for r in summary_sorted]
        ys = [r["test_acc_best"] for r in summary_sorted]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, marker="o")
        for xi, yi in zip(xs, ys):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center")
        ax.set_xlabel("Labeled ratio per class")
        ax.set_ylabel("Best test accuracy")
        ax.set_title("Semi-supervised performance vs labeled ratio")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(args.output_dir / "summary_accuracy.png", dpi=200)
        plt.close(fig)
        print(f"Summary table: {table_path.resolve()}")
        print(f"Summary plot: {(args.output_dir / 'summary_accuracy.png').resolve()}")


if __name__ == "__main__":
    main()
