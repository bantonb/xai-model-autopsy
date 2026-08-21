"""Run diagnostics on both trained arms.

For each arm:
- On PURE test patches:
    - r(p0, p1) and mean|p0+p1-1| (the headline coupling metric).
    - Per-class precision/recall/F1 via argmax prediction.
    - Multilabel F1(num_labels=2, macro) with 0.5 threshold — the thesis's
      metric of choice, kept here so the "still looks fine on pure test"
      claim is quotable.
- On HYBRID test patches:
    - p0, p1 distribution and predicted-both-classes fraction.
- Captum Occlusion on N pure test patches + N hybrid test patches, through
  the frozen backbone + arm's head, per class-target. Saved to figures/
  under slide-code-free filenames.

Emits diagnostics.json + figures/. Prints a compact summary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from captum.attr import Occlusion
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support

from pipeline import embed as embed_mod
from pipeline import train as train_mod

DEFAULT_WORKDIR = Path.home() / "projects" / "xai-model-autopsy-workdir"
REPO_FIGURES = Path(__file__).resolve().parent.parent / "figures"
N_OCCLUSION_SAMPLES = 6   # per (arm, source-class) — kept small on purpose
SEED = 20260820


def _load_index(workdir: Path) -> list[dict]:
    with (workdir / "embeddings_index.csv").open() as f:
        return [{**r, "row_idx": int(r["row_idx"])} for r in csv.DictReader(f)]


def _load_head(ckpt_path: Path, device: torch.device) -> nn.Module:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    head = train_mod.MLPHead().to(device)
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _eval_arm(arm: str, workdir: Path, device: torch.device, idx: list[dict]) -> dict:
    emb = np.load(workdir / "embeddings.npy")
    head = _load_head(workdir / "checkpoints" / f"{arm}.pt", device)

    def slice_test(class_label: str) -> tuple[np.ndarray, list[dict]]:
        keep = [r for r in idx if r["split"] == "test" and r["class_label"] == class_label]
        return emb[[r["row_idx"] for r in keep]], keep

    pure_labels = ("schwannoma", "neurofibroma")
    all_pure = [r for r in idx if r["split"] == "test" and r["class_label"] in pure_labels]
    pure_X = emb[[r["row_idx"] for r in all_pure]]
    pure_y_class = np.asarray([0 if r["class_label"] == "schwannoma" else 1 for r in all_pure])
    pure_y_ml = np.zeros((len(all_pure), 2), dtype=np.float32)
    for i, c in enumerate(pure_y_class):
        pure_y_ml[i, c] = 1.0

    hy_X, hy_rows = slice_test("hybrid")

    with torch.no_grad():
        pure_probs = torch.sigmoid(head(torch.from_numpy(pure_X).to(device))).cpu().numpy()
        if len(hy_X):
            hy_probs = torch.sigmoid(head(torch.from_numpy(hy_X).to(device))).cpu().numpy()
        else:
            hy_probs = np.zeros((0, 2), dtype=np.float32)

    p0, p1 = pure_probs[:, 0], pure_probs[:, 1]
    r_pp = _pearson(p0, p1)
    mean_off = float(np.abs(p0 + p1 - 1.0).mean())

    # Argmax multi-class prediction
    argmax_pred = pure_probs.argmax(axis=1)
    prec, rec, f1, sup = precision_recall_fscore_support(
        pure_y_class, argmax_pred, labels=[0, 1], zero_division=0
    )

    # Threshold-0.5 multilabel (independent per position)
    ml_pred = (pure_probs >= 0.5).astype(int)
    prec_ml, rec_ml, f1_ml, _ = precision_recall_fscore_support(
        pure_y_ml.astype(int), ml_pred, average="macro", zero_division=0
    )

    # Hybrid behaviour
    hy = {}
    if len(hy_probs):
        hy = {
            "n": int(len(hy_probs)),
            "p0_mean": float(hy_probs[:, 0].mean()), "p0_std": float(hy_probs[:, 0].std()),
            "p1_mean": float(hy_probs[:, 1].mean()), "p1_std": float(hy_probs[:, 1].std()),
            "sum_mean": float((hy_probs[:, 0] + hy_probs[:, 1]).mean()),
            "sum_std": float((hy_probs[:, 0] + hy_probs[:, 1]).std()),
            "frac_both_gt_0p5": float(((hy_probs[:, 0] >= 0.5) & (hy_probs[:, 1] >= 0.5)).mean()),
        }

    return {
        "arm": arm,
        "pure_n": int(len(pure_probs)),
        "pearson_r_probs_pure": r_pp,
        "mean_abs_p0_plus_p1_minus_1_pure": mean_off,
        "argmax_per_class": {
            "schwannoma": {"precision": float(prec[0]), "recall": float(rec[0]),
                            "f1": float(f1[0]), "support": int(sup[0])},
            "neurofibroma": {"precision": float(prec[1]), "recall": float(rec[1]),
                              "f1": float(f1[1]), "support": int(sup[1])},
        },
        "multilabel_macro": {
            "precision": float(prec_ml), "recall": float(rec_ml), "f1": float(f1_ml),
        },
        "hybrid_test": hy,
        "_probs_pure": pure_probs, "_probs_hybrid": hy_probs,
    }


def _plot_coupling(diag_broken: dict, diag_corrected: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, d in zip(axes, (diag_broken, diag_corrected)):
        p = d["_probs_pure"]
        ax.scatter(p[:, 0], p[:, 1], s=4, alpha=0.4,
                   color="C0" if d["arm"] == "broken" else "C1")
        ax.plot([0, 1], [1, 0], "k--", alpha=0.35, lw=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("p(schwannoma)"); ax.set_ylabel("p(neurofibroma)")
        ax.set_title(f"{d['arm']}: r(p0, p1) = {d['pearson_r_probs_pure']:+.3f}"
                     f"\nmean|p0+p1-1| = {d['mean_abs_p0_plus_p1_minus_1_pure']:.3f}"
                     f"   N={d['pure_n']}")
    fig.suptitle("Real-data test set: BCE two-neuron head output coupling")
    fig.tight_layout()
    fig.savefig(out_dir / "real_p0_vs_p1.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, d in zip(axes, (diag_broken, diag_corrected)):
        p = d["_probs_pure"]
        ax.hist(p[:, 0] + p[:, 1], bins=60, alpha=0.85,
                color="C0" if d["arm"] == "broken" else "C1")
        ax.axvline(1.0, color="k", ls="--", lw=1)
        ax.set_xlabel("p0 + p1"); ax.set_ylabel("count")
        ax.set_title(f"{d['arm']}: mean|p0+p1-1| = {d['mean_abs_p0_plus_p1_minus_1_pure']:.3f}")
    fig.suptitle("p(schwannoma) + p(neurofibroma) on pure test patches")
    fig.tight_layout()
    fig.savefig(out_dir / "real_p0_plus_p1.png", dpi=140); plt.close(fig)


def _plot_hybrid(diag_broken: dict, diag_corrected: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    if not diag_broken["_probs_hybrid"].size:
        print("  (no hybrid test patches; skipping hybrid figure)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, d in zip(axes, (diag_broken, diag_corrected)):
        p = d["_probs_hybrid"]
        ax.scatter(p[:, 0], p[:, 1], s=12, alpha=0.75,
                   color="C0" if d["arm"] == "broken" else "C1")
        ax.plot([0, 1], [1, 0], "k--", alpha=0.35, lw=1)
        ax.axhline(0.5, color="k", lw=0.4); ax.axvline(0.5, color="k", lw=0.4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("p(schwannoma)"); ax.set_ylabel("p(neurofibroma)")
        f = d["hybrid_test"].get("frac_both_gt_0p5", 0.0)
        ax.set_title(f"{d['arm']} on HYBRID test\nfrac(p0>=.5 AND p1>=.5) = {f:.2f}"
                     f"   N={d['hybrid_test'].get('n', 0)}")
    fig.tight_layout()
    fig.savefig(out_dir / "real_hybrid_probs.png", dpi=140); plt.close(fig)


def _load_patch_tensor(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (embed_mod.IMG_SIZE, embed_mod.IMG_SIZE):
        img = img.resize((embed_mod.IMG_SIZE, embed_mod.IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device), img


def _occlusion_figures(
    workdir: Path, device: torch.device, out_dir: Path, seed: int = SEED
) -> None:
    """Captum Occlusion through frozen backbone + each head, on a small sample."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    idx = _load_index(workdir)
    rng = random.Random(seed)

    def sample_test(class_label: str, n: int):
        rows = [r for r in idx if r["split"] == "test" and r["class_label"] == class_label]
        rng.shuffle(rows)
        return rows[:n]

    samples = []
    for cls in ("schwannoma", "neurofibroma", "hybrid"):
        for r in sample_test(cls, N_OCCLUSION_SAMPLES):
            samples.append((cls, r))
    print(f"occlusion samples: {len(samples)} (per class: N={N_OCCLUSION_SAMPLES})")

    backbone = embed_mod.build_frozen_backbone(device=device)

    class Full(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone; self.head = head
        def forward(self, x):
            f = self.backbone(x)
            if f.ndim > 2:
                f = f.mean(dim=list(range(2, f.ndim)))
            return self.head(f)

    for arm in ("broken", "corrected"):
        head = _load_head(workdir / "checkpoints" / f"{arm}.pt", device)
        full = Full(backbone, head).to(device).eval()
        occ = Occlusion(full)

        for i, (src_cls, r) in enumerate(samples):
            x, pil = _load_patch_tensor(r["patch_path"], device)
            attrs = []
            for target in (0, 1):
                a = occ.attribute(
                    inputs=x, target=target,
                    strides=(3, 16, 16),
                    sliding_window_shapes=(3, 32, 32),
                    baselines=0.0,
                )
                attrs.append(a[0].sum(dim=0).cpu().numpy())

            fig, ax = plt.subplots(1, 3, figsize=(9, 3.4))
            ax[0].imshow(np.asarray(pil)); ax[0].set_title(f"patch\nsrc={src_cls}")
            ax[0].axis("off")
            vmax = float(max(np.abs(attrs[0]).max(), np.abs(attrs[1]).max()) + 1e-9)
            for j, a in enumerate(attrs):
                im = ax[j + 1].imshow(a, cmap="seismic", vmin=-vmax, vmax=vmax)
                ax[j + 1].set_title(f"attr -> class {j}")
                ax[j + 1].axis("off")
            fig.suptitle(f"Occlusion — {arm} arm — sample {i:02d}")
            fig.tight_layout()
            # slide-code-free filename
            uid = hashlib.sha256(r["patch_path"].encode()).hexdigest()[:8]
            fig.savefig(out_dir / f"occ_{arm}_sample{i:02d}_{src_cls}_{uid}.png", dpi=110)
            plt.close(fig)


def run(workdir: Path, device_str: str | None = None) -> None:
    if device_str is None:
        device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"diagnostics on {device}")
    idx = _load_index(workdir)

    dB = _eval_arm("broken", workdir, device, idx)
    dC = _eval_arm("corrected", workdir, device, idx)

    for d in (dB, dC):
        print(f"[{d['arm']}]  N_pure={d['pure_n']}"
              f"  r(p0,p1)={d['pearson_r_probs_pure']:+.4f}"
              f"  mean|p0+p1-1|={d['mean_abs_p0_plus_p1_minus_1_pure']:.4f}"
              f"  ML-F1(macro)={d['multilabel_macro']['f1']:.4f}"
              f"  schwannoma-F1={d['argmax_per_class']['schwannoma']['f1']:.4f}"
              f"  neurofibroma-F1={d['argmax_per_class']['neurofibroma']['f1']:.4f}")
        if d["hybrid_test"]:
            h = d["hybrid_test"]
            print(f"           hybrid-test N={h['n']} "
                  f"p0_mean={h['p0_mean']:.3f} p1_mean={h['p1_mean']:.3f} "
                  f"sum_mean={h['sum_mean']:.3f} "
                  f"frac_both>=0.5={h['frac_both_gt_0p5']:.3f}")

    fig_dir_synth = REPO_FIGURES / "real"
    _plot_coupling(dB, dC, fig_dir_synth)
    _plot_hybrid(dB, dC, fig_dir_synth)
    _occlusion_figures(workdir, device, REPO_FIGURES / "occlusion")

    out = workdir / "diagnostics.json"
    serialisable = []
    for d in (dB, dC):
        d2 = {k: v for k, v in d.items() if not k.startswith("_")}
        serialisable.append(d2)
    out.write_text(json.dumps(serialisable, indent=2))
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    run(args.workdir, device_str=args.device)


if __name__ == "__main__":
    main()
