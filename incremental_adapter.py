import argparse
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from incremental.models_adapted import adapt_net


def load_ustc_incremental(data_dir: Path, max_per_class: int = 2000, seed: int = 0):
    # Reuse minimalist loader similar to semi_supervised_experiment but without labels per class
    X_list, y_list = [], []
    rng = np.random.default_rng(seed)
    for csv_path in sorted(data_dir.glob("*.csv")):
        class_id = int(csv_path.stem)
        with csv_path.open() as f:
            rows = []
            for i, line in enumerate(f):
                if i >= max_per_class:
                    break
                cells = [c.strip() for c in line.split(",") if c.strip() != ""]
                vals = [int(c, 16) for c in cells]
                rows.append(vals)
        arr = np.array(rows, dtype=np.float32) / 255.0
        lbls = np.full((arr.shape[0],), class_id, dtype=np.int64)
        X_list.append(arr)
        y_list.append(lbls)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def incremental_eval(model_path: Path, data_dir: Path, batch_size: int = 256, max_per_class: int = 2000):
    X, y = load_ustc_incremental(data_dir, max_per_class=max_per_class, seed=0)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    ckpt = torch.load(model_path, map_location="cpu")
    model = adapt_net(num_classes=len(np.unique(y)))
    model.load_state_dict(ckpt)
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(yb.cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    acc = (y_true == y_pred).mean()
    return acc, y_true, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run incremental model evaluation on USTC data.")
    parser.add_argument("--data-dir", type=Path, default=Path("ustc_all"))
    parser.add_argument("--model-path", type=Path, default=Path("incremental/ustc_model"))
    parser.add_argument("--max-per-class", type=int, default=2000)
    args = parser.parse_args()
    acc, _, _ = incremental_eval(args.model_path, args.data_dir, max_per_class=args.max_per_class)
    print(f"Accuracy: {acc:.4f}")
