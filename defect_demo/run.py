"""Synthetic reproduction of the two-neuron BCE coupling defect.

Same head architecture and loss the thesis used, on 2-D Gaussian-cluster
data. Two arms trained identically except for the label space:

  broken     -- pure classes only, labels [1,0] / [0,1]
  corrected  -- same + hybrid samples [1,1] and background samples [0,0]

The broken arm collapses to p0 + p1 approx 1 and mirrored FeatureAblation
attributions across the two output neurons. The corrected arm does not.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import numpy as np
import torch
import torch.nn as nn
from captum.attr import FeatureAblation
from torch.utils.data import DataLoader, TensorDataset


def _pearson(a, b):
    """Pearson r for two 1-D arrays; returns 0.0 for degenerate (zero-variance) inputs."""
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

SEED = 20260820
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures" / "synthetic"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_data(n_per_class: int = 1500, n_hybrid: int = 800, n_bg: int = 800, seed: int = SEED):
    """Four 2-D Gaussian clusters.

    Class-0 centered at (-2, 0), class-1 at (+2, 0), hybrid (both-present)
    at (0, 0), background (neither-present) at (0, 4).
    """
    rng = np.random.default_rng(seed)
    X0 = rng.normal(loc=(-2.0, 0.0), scale=1.0, size=(n_per_class, 2))
    X1 = rng.normal(loc=(+2.0, 0.0), scale=1.0, size=(n_per_class, 2))
    Xh = rng.normal(loc=(0.0, 0.0), scale=0.6, size=(n_hybrid, 2))
    Xb = rng.normal(loc=(0.0, 4.0), scale=0.6, size=(n_bg, 2))
    return (X0.astype(np.float32), X1.astype(np.float32),
            Xh.astype(np.float32), Xb.astype(np.float32))


class TwoNeuronHead(nn.Module):
    """MLP: in -> hidden -> hidden -> 2 raw logits. Same shape as the thesis head."""

    def __init__(self, in_dim: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def train(model: nn.Module, X: np.ndarray, Y: np.ndarray,
          epochs: int = 150, lr: float = 1e-2, batch: int = 64) -> nn.Module:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    return model


@dataclass
class ArmDiagnostics:
    arm: str
    pearson_r_probs: float
    mean_abs_p0_plus_p1_minus_1: float
    pearson_r_attr: float
    p0: np.ndarray
    p1: np.ndarray
    attr0: np.ndarray
    attr1: np.ndarray


def diagnose(model: nn.Module, X_grid: np.ndarray, X_attr: np.ndarray, arm: str) -> ArmDiagnostics:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X_grid))).numpy()
    p0, p1 = probs[:, 0], probs[:, 1]
    r_pp = _pearson(p0, p1)
    mean_off = float(np.abs(p0 + p1 - 1.0).mean())

    fa = FeatureAblation(model)
    x_attr = torch.from_numpy(X_attr)
    a0 = fa.attribute(x_attr, target=0).numpy()
    a1 = fa.attribute(x_attr, target=1).numpy()
    r_attr = _pearson(a0.ravel(), a1.ravel())

    return ArmDiagnostics(
        arm=arm,
        pearson_r_probs=float(r_pp),
        mean_abs_p0_plus_p1_minus_1=mean_off,
        pearson_r_attr=float(r_attr),
        p0=p0, p1=p1, attr0=a0, attr1=a1,
    )


def build_arm(arm: str, seed: int = SEED):
    set_seed(seed)
    X0, X1, Xh, Xb = make_data(seed=seed)
    if arm == "broken":
        X = np.concatenate([X0, X1])
        Y = np.concatenate([
            np.tile([1.0, 0.0], (len(X0), 1)),
            np.tile([0.0, 1.0], (len(X1), 1)),
        ]).astype(np.float32)
    elif arm == "corrected":
        X = np.concatenate([X0, X1, Xh, Xb])
        Y = np.concatenate([
            np.tile([1.0, 0.0], (len(X0), 1)),
            np.tile([0.0, 1.0], (len(X1), 1)),
            np.tile([1.0, 1.0], (len(Xh), 1)),
            np.tile([0.0, 0.0], (len(Xb), 1)),
        ]).astype(np.float32)
    else:
        raise ValueError(f"unknown arm: {arm!r}")
    model = TwoNeuronHead()
    train(model, X, Y)

    xs, ys = np.meshgrid(np.linspace(-5, 5, 80), np.linspace(-3, 7, 80))
    X_grid = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    X_attr = np.concatenate([X0[:150], X1[:150], Xh[:150], Xb[:150]])
    return model, diagnose(model, X_grid, X_attr, arm)


def _plot(diagB: ArmDiagnostics, diagC: ArmDiagnostics, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, d in zip(ax, (diagB, diagC)):
        a.hist(d.p0 + d.p1, bins=50, alpha=0.8,
               color="C0" if d.arm == "broken" else "C1")
        a.axvline(1.0, color="k", ls="--", lw=1)
        a.set_xlabel("p0 + p1 across test grid")
        a.set_title(f"{d.arm}: mean|p0+p1-1| = {d.mean_abs_p0_plus_p1_minus_1:.3f}")
    fig.suptitle("Synthetic defect demo: two-neuron BCE output coupling")
    fig.tight_layout()
    fig.savefig(out_dir / "synth_p0_plus_p1.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for a, d in zip(ax, (diagB, diagC)):
        a.scatter(d.p0, d.p1, s=6, alpha=0.5,
                  color="C0" if d.arm == "broken" else "C1")
        a.plot([0, 1], [1, 0], "k--", alpha=0.4, lw=1)
        a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.set_xlabel("p0"); a.set_ylabel("p1")
        a.set_title(f"{d.arm}: r(p0, p1) = {d.pearson_r_probs:+.3f}")
    fig.tight_layout()
    fig.savefig(out_dir / "synth_p0_vs_p1.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for a, d in zip(ax, (diagB, diagC)):
        a.scatter(d.attr0.ravel(), d.attr1.ravel(), s=8, alpha=0.6,
                  color="C0" if d.arm == "broken" else "C1")
        lim = float(max(np.abs(d.attr0).max(), np.abs(d.attr1).max())) * 1.05
        a.plot([-lim, lim], [lim, -lim], "k--", alpha=0.3, lw=1)
        a.axhline(0, color="k", lw=0.4); a.axvline(0, color="k", lw=0.4)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
        a.set_xlabel("attribution to class 0"); a.set_ylabel("attribution to class 1")
        a.set_title(f"{d.arm}: r(attr0, attr1) = {d.pearson_r_attr:+.3f}")
    fig.suptitle("FeatureAblation attributions per output neuron")
    fig.tight_layout()
    fig.savefig(out_dir / "synth_attributions.png", dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Synthetic BCE two-neuron coupling demo.")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--out", type=Path, default=FIGURES_DIR)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    _, dB = build_arm("broken", seed=args.seed)
    _, dC = build_arm("corrected", seed=args.seed)

    for d in (dB, dC):
        print(f"{d.arm:<10s}  r(p0,p1) = {d.pearson_r_probs:+.4f}"
              f"   mean|p0+p1-1| = {d.mean_abs_p0_plus_p1_minus_1:.4f}"
              f"   r(attr0,attr1) = {d.pearson_r_attr:+.4f}")

    if not args.no_plot:
        _plot(dB, dC, args.out)
        print(f"figures -> {args.out}")


if __name__ == "__main__":
    main()
