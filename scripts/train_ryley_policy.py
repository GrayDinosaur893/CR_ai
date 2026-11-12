"""
Train a local PyTorch Ryley Elixir-Discipline Policy Network on CPU/GPU.

Loads state->action supervision pairs generated from Ryley CR coaching rules +
recorded CR_AI arena states (`data/derived/ryley_policy_samples.jsonl`), trains an MLP
policy with AdamW on an 80/20 stratified-style split, evaluates validation accuracy
and per-class precision/recall, and exports `models/ryley_elixir_policy.pt` +
`data/derived/ryley_training_report.json`.
"""

import json
import pathlib
import random
from collections import Counter
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
ACTIONS = [
    "wait",
    "cheap_cycle_defend",
    "knight_or_musk_counter",
    "cannon_pull_defend",
    "opposite_lane_punish",
    "slow_roll_back",
]
ACTION_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}


class RyleyElixirPolicyNet(nn.Module):
    def __init__(self, in_dim: int = 12, num_classes: int = 6) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> int:
    samples_path = ROOT_DIR / "data" / "derived" / "ryley_policy_samples.jsonl"
    if not samples_path.exists():
        print(f"[-] Missing samples file: {samples_path}. Run scripts/prepare_ryley_training.py first.")
        return 1

    raw: List[Dict] = [json.loads(line) for line in samples_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    random.Random(42).shuffle(raw)

    xs = torch.tensor([r["features"] for r in raw], dtype=torch.float32)
    ys = torch.tensor([ACTION_TO_IDX.get(r["target_action"], 0) for r in raw], dtype=torch.long)

    split = max(1, int(0.8 * len(raw)))
    x_train, y_train = xs[:split], ys[:split]
    x_val, y_val = xs[split:], ys[split:]

    counts = Counter(int(y) for y in y_train.tolist())
    weights = torch.tensor(
        [len(y_train) / (len(ACTIONS) * max(1, counts.get(i, 1))) for i in range(len(ACTIONS))],
        dtype=torch.float32,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RyleyElixirPolicyNet(in_dim=xs.shape[1], num_classes=len(ACTIONS)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)

    history = []
    epochs = 40
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(bx)

        if ep in (1, 10, 20, 30, 40):
            model.eval()
            with torch.no_grad():
                tr_pred = model(x_train.to(device)).argmax(dim=1).cpu()
                va_pred = model(x_val.to(device)).argmax(dim=1).cpu()
                tr_acc = float((tr_pred == y_train).float().mean().item())
                va_acc = float((va_pred == y_val).float().mean().item())
            avg_loss = total_loss / max(1, len(x_train))
            history.append({"epoch": ep, "train_loss": round(avg_loss, 4), "train_acc": round(tr_acc, 4), "val_acc": round(va_acc, 4)})
            print(f"Epoch {ep:2d}/{epochs} | loss={avg_loss:.4f} | train_acc={tr_acc*100:.2f}% | val_acc={va_acc*100:.2f}%")

    model.eval()
    with torch.no_grad():
        val_preds = model(x_val.to(device)).argmax(dim=1).cpu().tolist()
        val_true = y_val.tolist()

    per_class = {}
    for idx, name in enumerate(ACTIONS):
        tp = sum(1 for p, t in zip(val_preds, val_true) if p == idx and t == idx)
        fp = sum(1 for p, t in zip(val_preds, val_true) if p == idx and t != idx)
        fn = sum(1 for p, t in zip(val_preds, val_true) if p != idx and t == idx)
        total_t = tp + fn
        per_class[name] = {
            "support": total_t,
            "precision": round(tp / max(1, tp + fp), 4),
            "recall": round(tp / max(1, total_t), 4),
        }

    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "ryley_elixir_policy.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "actions": ACTIONS,
            "in_dim": int(xs.shape[1]),
            "val_acc": history[-1]["val_acc"],
        },
        ckpt_path,
    )

    report = {
        "device": str(device),
        "total_samples": len(raw),
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "epochs": epochs,
        "final_train_loss": history[-1]["train_loss"],
        "final_train_acc": history[-1]["train_acc"],
        "final_val_acc": history[-1]["val_acc"],
        "per_class_val": per_class,
        "history": history,
        "checkpoint": str(ckpt_path),
    }
    report_path = ROOT_DIR / "data" / "derived" / "ryley_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved policy checkpoint: {ckpt_path}")
    print(f"Saved training report:   {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
