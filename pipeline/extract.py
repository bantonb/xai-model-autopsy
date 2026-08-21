"""Extract tissue patches from Data 2/*.ndpi WSIs into the workdir.

Deterministic:
- 1 slide per patient for pure classes, up to 2 for hybrid.
- Slides picked in sorted-filename order.
- RandomTiler seed derived from slide_id via sha256; per-slide cap = 1500;
  max_iter = 2x cap (avoid runaway on tissue-poor slides).
- Level 0, tile size 224x224, tissue_percent 50.

Post-hoc per-patient cap (35% of split-class total, per gate approval):
- For each (split, class), compute cap_per_patient such that the patient's
  share is at most cap_frac of the resulting group total; downsample if over.
- If a group has 1-2 patients (hybrid val/train/test), the cap is
  mathematically infeasible; the deviation is recorded in the report.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from histolab.slide import Slide
from histolab.tiler import RandomTiler

from pipeline import manifest
from pipeline import split as split_mod


ARCHIVE = Path(
    os.environ.get(
        "XAI_AUTOPSY_ARCHIVE",
        str(Path.home() / "Documents" / "Projects" / "Anton BA Thesis Project" / "Data 2"),
    )
)
DEFAULT_WORKDIR = Path.home() / "projects" / "xai-model-autopsy-workdir"
CLASS_DIR = {
    "schwannoma": "Schwannome",
    "neurofibroma": "Neurofibrome",
    "hybrid": "Hybrid",
}
SLIDES_PER_PATIENT = {"schwannoma": 1, "neurofibroma": 1, "hybrid": 2}

PER_SLIDE_CAP = 1500
MAX_ITER_MULTIPLIER = 2
TILE_SIZE = 224
TISSUE_PCT = 50.0
LEVEL = 0
SEED = 20260820


def _seed_for(slide_id: str) -> int:
    return int(hashlib.sha256(slide_id.encode()).hexdigest(), 16) % (2**31)


def _enumerate_slides(cls_label: str) -> list[str]:
    d = ARCHIVE / CLASS_DIR[cls_label]
    return sorted(f for f in os.listdir(d) if f.endswith(".ndpi"))


def _slides_by_patient(cls_label: str) -> dict[str, list[str]]:
    by: dict[str, list[str]] = defaultdict(list)
    for fn in _enumerate_slides(cls_label):
        by[manifest.patient_of_slide(fn[:-5])].append(fn)
    return dict(by)


def _select_slides(cls_label: str) -> list[tuple[str, str]]:
    """Return [(patient_id, ndpi_filename), ...] chosen deterministically."""
    picked: list[tuple[str, str]] = []
    for p, files in sorted(_slides_by_patient(cls_label).items()):
        for fn in sorted(files)[: SLIDES_PER_PATIENT[cls_label]]:
            picked.append((p, fn))
    return picked


def _extract_one(cls_label: str, filename: str, workdir: Path) -> tuple[list[Path], float, int]:
    slide_id = filename[:-5]
    src = ARCHIVE / CLASS_DIR[cls_label] / filename
    out_dir = workdir / "patches" / cls_label / slide_id
    if out_dir.exists():
        existing = sorted(out_dir.rglob("*.png"))
        if existing:
            return existing, 0.0, len(existing)
    out_dir.mkdir(parents=True, exist_ok=True)
    slide = Slide(str(src), processed_path=str(out_dir))
    tiler = RandomTiler(
        tile_size=(TILE_SIZE, TILE_SIZE),
        n_tiles=PER_SLIDE_CAP,
        level=LEVEL,
        seed=_seed_for(slide_id),
        check_tissue=True,
        tissue_percent=TISSUE_PCT,
        prefix="",
        suffix=".png",
        max_iter=PER_SLIDE_CAP * MAX_ITER_MULTIPLIER,
    )
    t0 = time.time()
    tiler.extract(slide)
    elapsed = time.time() - t0
    files = sorted(out_dir.rglob("*.png"))
    return files, elapsed, len(files)


def apply_per_patient_cap(
    rows: list[manifest.ManifestRow], cap_frac: float = 0.35, seed: int = SEED
) -> tuple[list[manifest.ManifestRow], Counter, dict]:
    """Downsample any patient whose share of its (split, class) exceeds cap_frac.

    Cap per patient x is chosen so that x / (x + others) <= cap_frac ==>
    x <= cap_frac / (1 - cap_frac) * others.

    If a group has only one patient, no cap is possible; recorded in `infeasible`.
    Returns (kept_rows, dropped_counter, infeasible_dict).
    """
    if not 0 < cap_frac < 1:
        raise ValueError(f"cap_frac must be in (0, 1); got {cap_frac}")
    rng = random.Random(seed)
    per_group: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        per_group[(r.split, r.class_label)].append(i)

    keep_all: set[int] = set()
    dropped: Counter = Counter()
    infeasible: dict = {}

    ratio = cap_frac / (1.0 - cap_frac)
    for (sp, cls), idxs in per_group.items():
        by_pt: dict[str, list[int]] = defaultdict(list)
        for i in idxs:
            by_pt[rows[i].patient_id].append(i)

        if len(by_pt) == 1:
            (only_p,) = by_pt.keys()
            infeasible[(sp, cls)] = {
                "patient": only_p,
                "share": 1.0,
                "reason": "single-patient group",
            }
            keep_all.update(idxs)
            continue

        counts = {p: len(v) for p, v in by_pt.items()}
        for p, ct in counts.items():
            others = sum(counts.values()) - ct
            max_x = int(ratio * others)
            if ct > max_x:
                pt_idx = list(by_pt[p]); rng.shuffle(pt_idx)
                keep_pt = pt_idx[:max_x]
                dropped[(sp, cls, p)] = ct - max_x
            else:
                keep_pt = list(by_pt[p])
            keep_all.update(keep_pt)

        # After capping, if the leading patient's share still >cap_frac, record it.
        final_counts = Counter(rows[i].patient_id for i in idxs if i in keep_all)
        top_p, top_ct = max(final_counts.items(), key=lambda kv: kv[1])
        total_after = sum(final_counts.values())
        share = top_ct / total_after if total_after else 0.0
        if share > cap_frac + 1e-9:
            infeasible[(sp, cls)] = {
                "patient": top_p,
                "share": share,
                "reason": "cap infeasible with current patient mix",
            }

    return [r for i, r in enumerate(rows) if i in keep_all], dropped, infeasible


def run(workdir: Path, only_class: str | None = None) -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    all_rows: list[manifest.ManifestRow] = []
    per_slide_stats: list[dict] = []

    classes = [only_class] if only_class else list(CLASS_DIR.keys())
    for cls in classes:
        patients = sorted(_slides_by_patient(cls).keys())
        splits = split_mod.assign_splits(cls, patients, seed=SEED)
        selected = _select_slides(cls)
        print(f"[{cls}] {len(selected)} slides across {len(patients)} patients")
        for pt, fn in selected:
            files, elapsed, count = _extract_one(cls, fn, workdir)
            print(f"  {cls}/{splits[pt]:<5s}  {fn}  -> {count} tiles in {elapsed:.1f}s")
            per_slide_stats.append(
                {"class": cls, "patient": pt, "slide": fn[:-5],
                 "split": splits[pt], "n_tiles": count, "elapsed_s": elapsed}
            )
            for f in files:
                all_rows.append(
                    manifest.ManifestRow(
                        patch_path=str(f.resolve()),
                        class_label=cls,
                        slide_id=fn[:-5],
                        patient_id=pt,
                        split=splits[pt],
                    )
                )

    print(f"\ntotal raw rows: {len(all_rows)}")
    kept, dropped, infeasible = apply_per_patient_cap(all_rows)
    print(f"per-patient cap: kept {len(kept)}, dropped {len(all_rows) - len(kept)}")
    for (sp, cls, p), n in sorted(dropped.items()):
        print(f"  capped {p} in ({sp}, {cls}): -{n} patches")
    for (sp, cls), info in sorted(infeasible.items()):
        print(f"  INFEASIBLE cap in ({sp}, {cls}): patient={info['patient']} "
              f"share={info['share']:.2%}  reason={info['reason']}")

    manifest.validate(kept, check_paths_exist=True)
    out = workdir / "manifest.csv"
    n = manifest.write(kept, out)
    print(f"\nwrote manifest {out} ({n} rows)")

    stats_out = workdir / "per_slide_stats.csv"
    import csv
    with stats_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "patient", "slide", "split", "n_tiles", "elapsed_s"])
        w.writeheader(); w.writerows(per_slide_stats)
    print(f"wrote per-slide stats {stats_out}")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    p.add_argument("--only-class", choices=list(CLASS_DIR.keys()), default=None)
    args = p.parse_args(argv)
    if not ARCHIVE.exists():
        sys.exit(f"archive not found: {ARCHIVE}")
    run(args.workdir, only_class=args.only_class)


if __name__ == "__main__":
    main()
