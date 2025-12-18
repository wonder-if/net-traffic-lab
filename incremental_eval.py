import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from incremental.models_adapted import OpenEmbed


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate incremental model on USTC data using provided OpenEmbed.")
    parser.add_argument("--data-dir", type=Path, default=Path("ustc_all"))
    parser.add_argument("--model-path", type=Path, default=Path("incremental/ustc_model"))
    parser.add_argument("--max-per-class", type=int, default=2000)
    args = parser.parse_args()
    acc, _, _ = evaluate(args.model_path, args.data_dir, max_per_class=args.max_per_class)
    print(f"Accuracy: {acc:.4f}")
