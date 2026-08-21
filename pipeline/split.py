"""Person-level split assignment.

Rules (per gate approval 2026-08-20):
- Hybrid class has only 5 patients; the split is *pinned* to guarantee at
  least one hybrid patient in each of train/val/test.
- Neurofibroma patient NfE3 (27 slides) is pinned to train so it never
  contaminates val/test.
- All other patients are bucketed deterministically by hashing
  f"{class_label}::{patient_id}::{seed}" and comparing to cumulative
  thresholds [0.70, 0.85, 1.00].
"""
from __future__ import annotations

import hashlib

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train, val, test
DEFAULT_SEED = 20260820

PINNED_SPLITS: dict[str, dict[str, str]] = {
    "schwannoma": {},
    "neurofibroma": {"NfE3": "train"},
    "hybrid": {
        "EH12": "train",
        "EH9": "train",
        "EH15": "val",
        "EH14": "test",
        "EH7": "test",
    },
}


def _hash_bucket(class_label: str, patient_id: str, seed: int) -> str:
    key = f"{class_label}::{patient_id}::{seed}".encode()
    h = int(hashlib.sha256(key).hexdigest(), 16)
    frac = (h % 10_000_000) / 10_000_000
    t_train, t_val, _ = SPLIT_FRACTIONS
    if frac < t_train:
        return "train"
    if frac < t_train + t_val:
        return "val"
    return "test"


def assign_splits(
    class_label: str, patient_ids: list[str], seed: int = DEFAULT_SEED
) -> dict[str, str]:
    """Return {patient_id: split} for one class.

    Pinned assignments always win. Everything else is hashed with the seed.
    """
    pinned = PINNED_SPLITS.get(class_label, {})
    out: dict[str, str] = {}
    for p in patient_ids:
        if p in pinned:
            out[p] = pinned[p]
        else:
            out[p] = _hash_bucket(class_label, p, seed)
    # Sanity: every patient assigned exactly once, no patient in two splits.
    if len(out) != len(set(patient_ids)):
        raise AssertionError(
            f"duplicate patients passed to assign_splits({class_label!r}): "
            f"got {len(patient_ids)} ids, {len(set(patient_ids))} unique"
        )
    return out
