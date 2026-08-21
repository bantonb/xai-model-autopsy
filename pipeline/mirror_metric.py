"""Numeric mirror metric for Occlusion attributions.

Closes PHASE_1 sub-claim 5d's qualitative-only status: for each of the 18
Occlusion samples (6 schwannoma + 6 neurofibroma + 6 hybrid) and each arm,
compute the Pearson r between the class-0-target and class-1-target
attribution maps (flattened).

Sample selection, seed, occlusion window/stride/baseline exactly match
pipeline.diagnose._occlusion_figures — the same 18 samples are used.

Writes:
- workdir/mirror_metric.json — per-sample values + per-arm summary.
Prints a compact table.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from captum.attr import Occlusion

from pipeline import diagnose as diag_mod
from pipeline import embed as embed_mod

DEFAULT_WORKDIR = Path.home() / "projects" / "xai-model-autopsy-workdir"
N_OCCLUSION_SAMPLES = diag_mod.N_OCCLUSION_SAMPLES
SEED = diag_mod.SEED


def _pearson_flat(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run(workdir: Path, device_str: str | None = None) -> dict:
    if device_str is None:
        device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"mirror_metric on {device}")

    idx = diag_mod._load_index(workdir)
    rng = random.Random(SEED)

    def sample_test(class_label: str, n: int):
        rows = [r for r in idx if r["split"] == "test" and r["class_label"] == class_label]
        rng.shuffle(rows)
        return rows[:n]

    samples: list[tuple[str, dict]] = []
    for cls in ("schwannoma", "neurofibroma", "hybrid"):
        for r in sample_test(cls, N_OCCLUSION_SAMPLES):
            samples.append((cls, r))
    print(f"samples: {len(samples)} (per class: N={N_OCCLUSION_SAMPLES})")

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

    per_arm: dict[str, list[dict]] = {}
    for arm in ("broken", "corrected"):
        head = diag_mod._load_head(workdir / "checkpoints" / f"{arm}.pt", device)
        full = Full(backbone, head).to(device).eval()
        occ = Occlusion(full)
        results: list[dict] = []
        for i, (src_cls, r) in enumerate(samples):
            x, _pil = diag_mod._load_patch_tensor(r["patch_path"], device)
            attrs = []
            for target in (0, 1):
                a = occ.attribute(
                    inputs=x, target=target,
                    strides=(3, 16, 16),
                    sliding_window_shapes=(3, 32, 32),
                    baselines=0.0,
                )
                attrs.append(a[0].sum(dim=0).cpu().numpy())
            r_val = _pearson_flat(attrs[0], attrs[1])
            results.append({"i": i, "src_class": src_cls, "r": r_val})
            print(f"  [{arm}] sample{i:02d} src={src_cls:<12s} r(attr0,attr1)={r_val:+.4f}")
        per_arm[arm] = results

    def summarize(vals: list[float]) -> dict:
        a = np.asarray(vals, dtype=np.float64)
        return {
            "n": int(a.size),
            "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()),
            "median": float(np.median(a)),
        }

    summary = {}
    for arm, recs in per_arm.items():
        all_r = [r["r"] for r in recs]
        by_cls: dict[str, list[float]] = {}
        for r in recs:
            by_cls.setdefault(r["src_class"], []).append(r["r"])
        summary[arm] = {
            "overall": summarize(all_r),
            "by_source_class": {k: summarize(v) for k, v in by_cls.items()},
        }

    print("\nsummary:")
    for arm in ("broken", "corrected"):
        s = summary[arm]["overall"]
        print(f"  {arm:<9s} r(attr0,attr1)  mean={s['mean']:+.4f}  "
              f"min={s['min']:+.4f}  max={s['max']:+.4f}  N={s['n']}")

    out = {
        "seed": SEED,
        "n_per_class": N_OCCLUSION_SAMPLES,
        "occlusion": {"window": [3, 32, 32], "stride": [3, 16, 16], "baseline": 0.0},
        "per_arm": per_arm,
        "summary": summary,
    }
    out_path = workdir / "mirror_metric.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    run(args.workdir, device_str=args.device)


if __name__ == "__main__":
    main()
