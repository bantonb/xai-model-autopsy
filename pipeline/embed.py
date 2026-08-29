"""Embed every manifest patch through a frozen ConvNeXt-Small backbone.

Writes:
- embeddings.npy in the workdir: float32 array of shape (N, 768).
- embeddings_index.csv in the workdir: N rows, columns
    (row_idx, patch_path, class_label, slide_id, patient_id, split)
  aligned 1-to-1 with embeddings.npy so a downstream stage can filter by
  class/split without re-reading images.

The backbone is `timm.create_model("convnext_small", pretrained=True,
num_classes=0)`. It is put in .eval() AND every parameter has
requires_grad_(False). A test asserts no backbone gradient after a forward
pass through the head.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from pipeline import manifest

DEFAULT_WORKDIR = Path.home() / "projects" / "xai-model-autopsy-workdir"
BACKBONE_NAME = "convnext_small"
FEATURE_DIM = 768
IMG_SIZE = 224


def build_frozen_backbone(device: torch.device | str = "cpu") -> nn.Module:
    """timm convnext_small, ImageNet weights, num_classes=0, frozen.

    Freezes in two ways: .eval() (BN/dropout in inference mode) AND
    requires_grad_(False) (gradients do not accumulate). The recon showed
    the thesis's trainer only did the first — this one does both, and the
    test in tests/test_freeze.py asserts it.
    """
    model = timm.create_model(BACKBONE_NAME, pretrained=True, num_classes=0)
    model.reset_classifier(num_classes=0)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


class _PatchDataset(Dataset):
    """Read patches lazily. Assumes 224x224 RGB PNGs; resizes if not."""

    def __init__(self, rows: list[manifest.ManifestRow]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> torch.Tensor:
        r = self.rows[i]
        img = Image.open(r.patch_path).convert("RGB")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0, 1]
        # ImageNet normalization (matches timm's default convnext preprocessing).
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return torch.from_numpy(arr.transpose(2, 0, 1))  # CHW


def run(workdir: Path, batch_size: int = 64, device_str: str | None = None) -> Path:
    workdir = Path(workdir)
    manifest_csv = workdir / "manifest.csv"
    rows = manifest.read(manifest_csv)
    manifest.validate(rows, check_paths_exist=True)

    if device_str is None:
        device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"device: {device}  N={len(rows)}  batch={batch_size}")

    model = build_frozen_backbone(device=device)

    ds = _PatchDataset(rows)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=4, persistent_workers=True,
        pin_memory=(device.type == "cuda"),
    )

    out = np.zeros((len(rows), FEATURE_DIM), dtype=np.float32)
    t0 = time.time(); done = 0
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device, non_blocking=True)
            feats = model(xb)
            if feats.ndim > 2:  # some timm models return spatial features
                feats = feats.mean(dim=list(range(2, feats.ndim)))
            out[done : done + feats.shape[0]] = feats.detach().cpu().numpy()
            done += feats.shape[0]
            if done % (batch_size * 10) == 0:
                elapsed = time.time() - t0
                print(f"  {done}/{len(rows)}  ({done/elapsed:.1f} patches/s)")

    elapsed = time.time() - t0
    print(f"embedded {done}/{len(rows)} patches in {elapsed:.1f}s "
          f"({done/max(elapsed, 1e-6):.1f} patches/s)")

    emb_path = workdir / "embeddings.npy"
    idx_path = workdir / "embeddings_index.csv"
    np.save(emb_path, out)
    with idx_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_idx", "patch_path", "class_label", "slide_id",
                    "patient_id", "split"])
        for i, r in enumerate(rows):
            w.writerow([i, r.patch_path, r.class_label, r.slide_id,
                        r.patient_id, r.split])
    print(f"wrote {emb_path} shape={out.shape}")
    print(f"wrote {idx_path}")
    return emb_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None,
                   help="mps | cpu (default: mps if available)")
    args = p.parse_args(argv)
    run(args.workdir, batch_size=args.batch_size, device_str=args.device)


if __name__ == "__main__":
    main()
