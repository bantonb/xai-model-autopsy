"""Assertions on the canonical manifest and the person-level split.

The person-level split guarantee is enforced by a running assertion in
`pipeline.manifest.validate`, not by a comment. This test exercises it on
a synthetic and on the actual archive's patient roster.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from pipeline import manifest, split


ARCHIVE = Path(
    os.environ.get(
        "XAI_AUTOPSY_ARCHIVE",
        str(Path.home() / "Documents" / "Projects" / "Anton BA Thesis Project" / "Data 2"),
    )
)

_PATIENT_RE = re.compile(r"^([A-Za-z]+\s*\d+(?:\.\d+)?)")


def _patient_of_filename(fn: str) -> str:
    m = _PATIENT_RE.match(fn)
    assert m, f"unparseable slide filename: {fn!r}"
    return m.group(1).strip()


def _patients_per_class() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for cls_dir, cls_label in (
        ("Schwannome", "schwannoma"),
        ("Neurofibrome", "neurofibroma"),
        ("Hybrid", "hybrid"),
    ):
        d = ARCHIVE / cls_dir
        if not d.exists():
            pytest.skip(f"archive class dir missing: {d}")
        files = sorted(f for f in os.listdir(d) if f.endswith(".ndpi"))
        patients = sorted({_patient_of_filename(f) for f in files})
        out[cls_label] = patients
    return out


def test_patient_of_slide_extraction():
    assert manifest.patient_of_slide("S18 - 1_5_0") == "S18"
    assert manifest.patient_of_slide("S12 -1_0") == "S12"
    assert manifest.patient_of_slide("NF 12.2 -1_1") == "NF 12.2"
    assert manifest.patient_of_slide("NfE3 - 1_0") == "NfE3"
    assert manifest.patient_of_slide("EH12 - 1_0") == "EH12"


def test_split_pins_hybrid_by_patient():
    assignments = split.assign_splits(
        "hybrid", ["EH7", "EH9", "EH12", "EH14", "EH15"]
    )
    # Every hybrid patient must be in the fixed slot from the gate approval.
    assert assignments["EH12"] == "train"
    assert assignments["EH9"] == "train"
    assert assignments["EH15"] == "val"
    assert assignments["EH14"] == "test"
    assert assignments["EH7"] == "test"


def test_split_pins_NfE3_to_train():
    patients = _patients_per_class()["neurofibroma"]
    assignments = split.assign_splits("neurofibroma", patients)
    assert assignments["NfE3"] == "train"


def test_split_is_deterministic_across_calls():
    patients = _patients_per_class()
    a = {c: split.assign_splits(c, patients[c]) for c in patients}
    b = {c: split.assign_splits(c, patients[c]) for c in patients}
    assert a == b


def test_all_splits_populated_per_class():
    patients = _patients_per_class()
    for cls, pts in patients.items():
        assignments = split.assign_splits(cls, pts)
        got = set(assignments.values())
        assert got == set(split.SPLIT_FRACTIONS.__class__(("train", "val", "test"))), (
            f"class {cls}: not every split populated, got {sorted(got)} "
            f"(assignments={assignments})"
        )


def test_validate_rejects_patient_leak(tmp_path):
    # Same patient in two splits -> validator must raise.
    p = str(tmp_path / "dummy.png")
    Path(p).write_bytes(b"")  # zero-byte placeholder
    rows = [
        manifest.ManifestRow(
            patch_path=p, class_label="schwannoma",
            slide_id="S18 - 1_5_0", patient_id="S18", split="train",
        ),
        manifest.ManifestRow(
            patch_path=p, class_label="schwannoma",
            slide_id="S18 - 1_5_1", patient_id="S18", split="test",
        ),
    ]
    with pytest.raises(AssertionError, match="patient-level split violated"):
        manifest.validate(rows, check_paths_exist=False)


def test_validate_rejects_bad_class_label(tmp_path):
    p = str(tmp_path / "dummy.png")
    Path(p).write_bytes(b"")
    rows = [
        manifest.ManifestRow(
            patch_path=p, class_label="tumor",  # not in CLASS_LABELS
            slide_id="S18 - 1_5_0", patient_id="S18", split="train",
        ),
    ]
    with pytest.raises(AssertionError, match="class_label"):
        manifest.validate(rows, check_paths_exist=False)


def test_validate_rejects_slide_patient_mismatch(tmp_path):
    p = str(tmp_path / "dummy.png")
    Path(p).write_bytes(b"")
    rows = [
        manifest.ManifestRow(
            patch_path=p, class_label="schwannoma",
            slide_id="S18 - 1_5_0", patient_id="S12",  # wrong
            split="train",
        ),
    ]
    with pytest.raises(AssertionError, match="derives patient"):
        manifest.validate(rows, check_paths_exist=False)


def test_roundtrip_csv(tmp_path):
    p = tmp_path / "patch.png"
    p.write_bytes(b"x")
    out = tmp_path / "manifest.csv"
    rows = [
        manifest.ManifestRow(
            patch_path=str(p), class_label="schwannoma",
            slide_id="S18 - 1_5_0", patient_id="S18", split="train",
        ),
    ]
    manifest.write(rows, out)
    loaded = manifest.read(out)
    assert loaded == rows
    manifest.validate(loaded, check_paths_exist=True)
