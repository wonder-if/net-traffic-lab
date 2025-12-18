import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class AdaptNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        f = self.features(x)
        return self.classifier(f)


def load_ustc(data_dir: Path, max_per_class: int = 2000, seed: int = 0):
    X_list, y_list = [], []
    rng = np.random.default_rng(seed)
    for csv_path in sorted(data_dir.glob("*.csv")):
        class_id = int(csv_path.stem)
        rows = []
        with csv_path.open() as f:
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
    if not X_list:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


def evaluate(model_path: Path, data_dir: Path, max_per_class: int = 2000, batch_size: int = 256):
    X, y = load_ustc(data_dir, max_per_class=max_per_class, seed=0)
    num_classes = len(np.unique(y))
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    state = torch.load(model_path, map_location="cpu")
    model = AdaptNet(num_classes=num_classes)
    model.load_state_dict(state, strict=False)
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(yb.cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    acc = float((y_true == y_pred).mean())
    return acc, y_true, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate incremental/adapted model on USTC data.")
    parser.add_argument("--data-dir", type=Path, default=Path("ustc_all"))
    parser.add_argument("--model-path", type=Path, default=Path("incremental/ustc_model"))
    parser.add_argument("--max-per-class", type=int, default=2000)
    args = parser.parse_args()
    acc, _, _ = evaluate(args.model_path, args.data_dir, max_per_class=args.max_per_class)
    print(f"Accuracy: {acc:.4f}")
