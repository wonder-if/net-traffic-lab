import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from ids2017.model import bp_model_2017


def load_dataset(path: Path, max_rows: int = 10000) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path, encoding="latin1")
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=0).reset_index(drop=True)
    labels_raw, label_names = pd.factorize(df["Label"])
    X_df = df.drop(columns=["Label"])
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    # Min-max
    mins = X_df.min()
    maxs = X_df.max()
    X_norm = (X_df - mins) / (maxs - mins + 1e-8)
    X = X_norm.values.astype(np.float32)
    y = labels_raw.astype(np.int64)
    return X, y, list(X_df.columns), list(label_names)


def load_model(checkpoint: Path, input_dim: int, num_classes: int) -> bp_model_2017:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = bp_model_2017(input_dim, num_classes)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # handle possible keys
    cleaned = {k.replace("module.", "").replace("model.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model


def predict_and_explain(
    model: bp_model_2017,
    sample: np.ndarray,
    target_label: Optional[int] = None,
) -> Dict:
    x = torch.tensor(sample[None, :], dtype=torch.float32, requires_grad=True)
    logits = model(x)
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    pred = int(probs.argmax())
    target = target_label if target_label is not None else pred
    loss = F.nll_loss(torch.log_softmax(logits, dim=1), torch.tensor([target]))
    loss.backward()
    grad = x.grad.detach().cpu().numpy()[0]
    attribution = np.abs(grad * sample)
    return {"probs": probs, "pred": pred, "attribution": attribution}


def fgsm_attack(
    model: bp_model_2017,
    sample: np.ndarray,
    true_label: int,
    epsilon: float,
) -> Tuple[np.ndarray, Dict, Dict]:
    x = torch.tensor(sample[None, :], dtype=torch.float32, requires_grad=True)
    logits = model(x)
    loss = F.nll_loss(torch.log_softmax(logits, dim=1), torch.tensor([true_label]))
    loss.backward()
    x_adv = (x + epsilon * x.grad.sign()).clamp(0, 1).detach().cpu().numpy()[0]
    clean = predict_and_explain(model, sample)
    adv = predict_and_explain(model, x_adv)
    return x_adv, clean, adv


def topk_to_table(attribution: np.ndarray, feature_names: List[str], k: int = 10) -> List[Dict]:
    idx = np.argsort(-attribution)[:k]
    return [{"feature": feature_names[i], "score": float(attribution[i])} for i in idx]


def bar_chart_topk(top_items: List[Dict], title: str, path: Path) -> Path:
    width, height = 900, 600
    margin = 120
    bar_width = 30
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    values = np.array([item["score"] for item in top_items], dtype=np.float32)
    max_val = float(values.max()) if len(values) else 1.0
    scale = (height - 2 * margin) / max(max_val, 1e-8)
    for i, item in enumerate(top_items):
        x0 = margin + i * (bar_width + 18)
        x1 = x0 + bar_width
        y1 = height - margin
        y0 = y1 - item["score"] * scale
        draw.rectangle([x0, y0, x1, y1], fill="#4C72B0")
        draw.text((x0, y0 - 14), f"{item['score']:.3f}", fill="black", font=font)
        # rotated label
        lbl = item["feature"][:40]
        lbl_img = Image.new("RGBA", (260, 60), (255, 255, 255, 0))
        d2 = ImageDraw.Draw(lbl_img)
        d2.text((0, 0), lbl, fill="black", font=font)
        rotated = lbl_img.rotate(50, expand=True)
        img.paste(rotated, (int(x0 - 10), int(y1 + 10)), rotated)
    draw.text((margin, margin - 60), title, fill="black", font=font)
    img.save(path)
    return path


def probs_to_table(probs: np.ndarray, label_names: List[str], top: int = 5) -> List[Dict]:
    idx = np.argsort(-probs)[:top]
    return [{"label": label_names[i] if i < len(label_names) else str(i), "prob": float(probs[i])} for i in idx]
