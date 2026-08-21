"""Train the two arms: broken (pure only, [1,0]/[0,1]) and corrected
(pure + hybrid [1,1]).

Identical across arms: architecture (768 -> 256 -> ReLU -> Dropout -> 2),
optimizer (Adam), lr (0.000116), weight_decay (0.000027), batch (128),
epochs (30), seed (20260820), loss (BCEWithLogitsLoss),
WeightedRandomSampler on per-class inverse-frequency weights. The ONLY
difference is which manifest rows enter the training set and what labels
they carry.
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

DEFAULT_WORKDIR = Path.home() / "projects" / "xai-model-autopsy-workdir"

# Best-config values from Pipeline/temp_ZZZZZZ_trainer_final.py:540-547.
LR = 0.000116
WEIGHT_DECAY = 0.000027
HIDDEN_DIM = 256
DROPOUT = 0.273294
BATCH_SIZE = 128
EPOCHS = 30
SEED = 20260820
FEATURE_DIM = 768


def _seed_all(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = HIDDEN_DIM,
                 dropout: float = DROPOUT, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _load_embeddings(workdir: Path):
    emb = np.load(workdir / "embeddings.npy")
    idx: list[dict] = []
    with (workdir / "embeddings_index.csv").open() as f:
        r = csv.DictReader(f)
        for row in r:
            row["row_idx"] = int(row["row_idx"])
            idx.append(row)
    if emb.shape[0] != len(idx):
        raise AssertionError(
            f"embeddings.npy N={emb.shape[0]} != index.csv N={len(idx)}"
        )
    if emb.shape[1] != FEATURE_DIM:
        raise AssertionError(
            f"embeddings dim {emb.shape[1]} != expected {FEATURE_DIM}"
        )
    return emb, idx


def _arm_indices_and_labels(idx: list[dict], arm: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (row_idxs, labels) for a given arm on a given split.

    Arms differ only in which manifest rows enter and what labels they carry:
    - broken:    pure classes only, labels [1,0] / [0,1]. Hybrid rows dropped.
    - corrected: same pure + hybrid, hybrid labels are [1,1].
    """
    rows, labels = [], []
    for row in idx:
        if row["split"] != split:
            continue
        if row["class_label"] == "schwannoma":
            lab = [1.0, 0.0]
        elif row["class_label"] == "neurofibroma":
            lab = [0.0, 1.0]
        elif row["class_label"] == "hybrid":
            if arm == "broken":
                continue
            lab = [1.0, 1.0]
        else:
            raise AssertionError(f"unknown class {row['class_label']!r}")
        rows.append(row["row_idx"]); labels.append(lab)
    return np.asarray(rows, dtype=np.int64), np.asarray(labels, dtype=np.float32)


def _sample_weights_from_labels(labels: np.ndarray) -> np.ndarray:
    """Inverse-frequency per unique label vector, matching thesis intent."""
    keys = [tuple(l) for l in labels.tolist()]
    from collections import Counter
    counts = Counter(keys)
    inv = {k: 1.0 / c for k, c in counts.items()}
    return np.asarray([inv[k] for k in keys], dtype=np.float32)


@dataclass
class TrainResult:
    arm: str
    best_epoch: int
    best_val_loss: float
    train_losses: list[float]
    val_losses: list[float]
    checkpoint_path: Path


def train_one_arm(
    arm: str, workdir: Path, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE,
    device_str: str | None = None,
) -> TrainResult:
    _seed_all(SEED)
    if device_str is None:
        device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)

    emb, idx = _load_embeddings(workdir)
    tr_i, tr_y = _arm_indices_and_labels(idx, arm, "train")
    va_i, va_y = _arm_indices_and_labels(idx, arm, "val")
    print(f"[{arm}] train N={len(tr_i)}  val N={len(va_i)}  device={device}")

    tr_X = torch.from_numpy(emb[tr_i])
    tr_Y = torch.from_numpy(tr_y)
    va_X = torch.from_numpy(emb[va_i]).to(device)
    va_Y = torch.from_numpy(va_y).to(device)

    sample_weights = _sample_weights_from_labels(tr_y)
    sampler_gen = torch.Generator(); sampler_gen.manual_seed(SEED)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights), replacement=True, generator=sampler_gen,
    )
    dl = DataLoader(
        TensorDataset(tr_X, tr_Y), batch_size=batch_size,
        sampler=sampler, num_workers=0,
    )

    model = MLPHead().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss()

    ckpt_dir = workdir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"{arm}.pt"
    best_val = float("inf"); best_epoch = -1
    train_losses: list[float] = []; val_losses: list[float] = []

    for ep in range(epochs):
        model.train()
        tot = 0.0; n = 0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            tot += float(loss.item()) * xb.shape[0]; n += xb.shape[0]
        train_loss = tot / max(n, 1)

        model.eval()
        with torch.no_grad():
            va_loss = float(crit(model(va_X), va_Y).item())
        train_losses.append(train_loss); val_losses.append(va_loss)
        if va_loss < best_val:
            best_val = va_loss; best_epoch = ep
            torch.save({
                "arm": arm, "epoch": ep, "val_loss": va_loss,
                "state_dict": model.state_dict(),
                "hparams": {
                    "lr": LR, "weight_decay": WEIGHT_DECAY,
                    "hidden": HIDDEN_DIM, "dropout": DROPOUT,
                    "batch": batch_size, "seed": SEED,
                },
            }, ckpt_path)
        print(f"  ep {ep:02d}  train_loss={train_loss:.4f}  val_loss={va_loss:.4f}"
              f"  {'*best' if ep == best_epoch else ''}")

    print(f"[{arm}] best_epoch={best_epoch} best_val_loss={best_val:.4f} -> {ckpt_path}")
    return TrainResult(arm, best_epoch, best_val, train_losses, val_losses, ckpt_path)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    p.add_argument("--arm", choices=("broken", "corrected", "both"), default="both")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    arms = ["broken", "corrected"] if args.arm == "both" else [args.arm]
    t0 = time.time()
    for arm in arms:
        train_one_arm(arm, args.workdir, epochs=args.epochs,
                      batch_size=args.batch_size, device_str=args.device)
    print(f"\ntotal training time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
