"""Canonical patch-manifest schema.

One schema defined here is read and written by every pipeline stage
(extraction -> embedding -> training -> diagnostics). Two stages sharing
an implicit shape with nothing enforcing agreement is the anti-pattern
this repo documents in the thesis; do not reintroduce it.
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Iterable

CLASS_LABELS = ("schwannoma", "neurofibroma", "hybrid")
SPLITS = ("train", "val", "test")

# Patient prefix: leading letters, optional space, digits, optional .digits.
# Matches "S18", "NF 12.2", "Nf5", "NfE3", "EH12".
_PATIENT_RE = re.compile(r"^([A-Za-z]+\s*\d+(?:\.\d+)?)")


def patient_of_slide(slide_id: str) -> str:
    """Extract the patient identifier from a slide id.

    A slide id looks like the .ndpi stem: e.g. "S18 - 1_5_0", "NF 12.2 -1_1",
    "NfE3 - 1_0", "EH12 - 1_0". The patient identifier is the leading
    "class-letters + case-number" segment before the first ' - '/' -'/'-'
    separator.
    """
    m = _PATIENT_RE.match(slide_id)
    if not m:
        raise ValueError(f"cannot extract patient id from slide_id={slide_id!r}")
    return m.group(1).strip()


@dataclass(frozen=True)
class ManifestRow:
    patch_path: str      # absolute path to patch file in the workdir
    class_label: str     # one of CLASS_LABELS
    slide_id: str        # e.g. "S18 - 1_5_0"
    patient_id: str      # e.g. "S18"
    split: str           # one of SPLITS


FIELDNAMES = [f.name for f in fields(ManifestRow)]


def write(rows: Iterable[ManifestRow], path: Path) -> int:
    """Write a manifest to CSV. Returns the number of rows written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
            n += 1
    return n


def read(path: Path) -> list[ManifestRow]:
    """Read a manifest CSV. Raises if the header does not match FIELDNAMES."""
    with Path(path).open() as f:
        r = csv.DictReader(f)
        if r.fieldnames != FIELDNAMES:
            raise ValueError(
                f"manifest schema mismatch at {path}: "
                f"expected {FIELDNAMES}, got {r.fieldnames}"
            )
        return [
            ManifestRow(
                patch_path=row["patch_path"],
                class_label=row["class_label"],
                slide_id=row["slide_id"],
                patient_id=row["patient_id"],
                split=row["split"],
            )
            for row in r
        ]


def validate(rows: list[ManifestRow], check_paths_exist: bool = True) -> None:
    """Assert manifest invariants. Runs on every load; asserts, not comments.

    - class_label is in CLASS_LABELS
    - split is in SPLITS
    - patient_of_slide(slide_id) == patient_id (extractor and consumer agree)
    - no patient appears in more than one split (person-level split guarantee)
    - optionally, every patch_path exists on disk
    """
    if not rows:
        raise AssertionError("manifest is empty")

    for i, r in enumerate(rows):
        if r.class_label not in CLASS_LABELS:
            raise AssertionError(
                f"row {i}: class_label={r.class_label!r} not in {CLASS_LABELS}"
            )
        if r.split not in SPLITS:
            raise AssertionError(f"row {i}: split={r.split!r} not in {SPLITS}")
        derived = patient_of_slide(r.slide_id)
        if derived != r.patient_id:
            raise AssertionError(
                f"row {i}: patient_id={r.patient_id!r} but slide_id={r.slide_id!r} "
                f"derives patient={derived!r}"
            )

    # Person-level split guarantee — the assertion, not a comment.
    patient_to_splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        patient_to_splits[r.patient_id].add(r.split)
    leaks = {p: sorted(s) for p, s in patient_to_splits.items() if len(s) > 1}
    if leaks:
        raise AssertionError(
            f"patient-level split violated: {len(leaks)} patient(s) appear in "
            f"multiple splits: {leaks}"
        )

    if check_paths_exist:
        missing = [r.patch_path for r in rows if not Path(r.patch_path).exists()]
        if missing:
            raise AssertionError(
                f"{len(missing)} patch_path(s) do not exist on disk; first: "
                f"{missing[0]}"
            )


def summarize(rows: list[ManifestRow]) -> dict:
    """Return a dict of aggregate counts for reporting."""
    per_split_class = Counter((r.split, r.class_label) for r in rows)
    per_patient_split_class = Counter(
        (r.patient_id, r.split, r.class_label) for r in rows
    )
    return {
        "total": len(rows),
        "per_split_class": dict(per_split_class),
        "per_patient_split_class": dict(per_patient_split_class),
    }
