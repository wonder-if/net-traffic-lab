import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from incremental.models_adapted import OpenEmbed

# Limit threads to avoid shared memory issues.
for env_var in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(env_var, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_AFFINITY", "none")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def load_ustc_raw(data_dir: Path, max_per_class: int = 2000, seed: int = 0):
    """
    Load USTC per-class CSVs as raw byte values (0-255), without normalization.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    csvs = sorted(data_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files under {data_dir}")

    X_list, y_list = [], []
    rng = np.random.default_rng(seed)
    for csv_path in csvs:
        class_id = int(csv_path.stem)
        rows = []
        with csv_path.open() as f:
            for i, line in enumerate(f):
                if i >= max_per_class:
                    break
                cells = [c.strip() for c in line.split(",") if c.strip() != ""]
                vals = [int(c, 16) for c in cells]
                rows.append(vals)
        arr = np.array(rows, dtype=np.int64)
        lbls = np.full((arr.shape[0],), class_id, dtype=np.int64)
        X_list.append(arr)
        y_list.append(lbls)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def evaluate(model_path: Path, data_dir: Path, max_per_class: int = 2000, batch_size: int = 256):
    if model_path.is_dir():
        pt_files = sorted(model_path.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files under {model_path}")
        model_path = pt_files[0]
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    X, y = load_ustc_raw(data_dir, max_per_class=max_per_class, seed=0)
    num_classes = len(np.unique(y))
    ds = TensorDataset(torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    state = torch.load(model_path, map_location="cpu")
    model = OpenEmbed(output=num_classes)
    model.load_state_dict(state, strict=False)
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            _, logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(yb.cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    acc = float((y_true == y_pred).mean())
    return acc, y_true, y_pred


def render_confusion(cm: np.ndarray, path: Path):
    # Simple heatmap using Pillow to avoid heavy GUI backends
    from PIL import Image, ImageDraw, ImageFont

    num_classes = cm.shape[0]
    cell = 30
    margin = 80
    width = margin * 2 + cell * num_classes
    height = margin * 2 + cell * num_classes
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    vmax = cm.max() if cm.max() > 0 else 1

    def color(val):
        ratio = val / vmax
        r = int(255 * ratio)
        b = int(255 * (1 - ratio))
        g = int(200 * (1 - abs(0.5 - ratio) * 2))
        return (r, g, b)

    for i in range(num_classes):
        for j in range(num_classes):
            v = int(cm[i, j])
            x0 = margin + j * cell
            y0 = margin + i * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=color(v), outline="black")
            draw.text((x0 + 6, y0 + 8), str(v), fill="black", font=font)
        draw.text((margin - 30, margin + i * cell + 8), str(i), fill="black", font=font)
        draw.text((margin + i * cell + 8, margin - 20), str(i), fill="black", font=font)

    draw.text((margin, 20), "Confusion Matrix", fill="black", font=font)
    img.save(path)


def save_metrics(acc: float, y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(np.unique(y_true)))
    metrics = {
        "accuracy": acc,
        "num_samples": int(len(y_true)),
        "per_class_acc": {
            str(c): float((y_pred[y_true == c] == c).mean()) for c in np.unique(y_true)
        },
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    render_confusion(cm, out_dir / "confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate incremental model on USTC data using provided OpenEmbed.")
    parser.add_argument("--data-dir", type=Path, default=Path("ustc_all"))
    parser.add_argument("--model-path", type=Path, default=Path("incremental/ustc_model/15_ustcALL_0.pt"))
    parser.add_argument("--max-per-class", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_incremental"))
    args = parser.parse_args()
    acc, y_true, y_pred = evaluate(args.model_path, args.data_dir, max_per_class=args.max_per_class)
    save_metrics(acc, y_true, y_pred, args.output_dir)
    print(f"Accuracy: {acc:.4f}")
