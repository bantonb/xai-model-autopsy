"""Emit the patches-per-split-per-class-per-patient table (Anton point 3).

Reads workdir/manifest.csv, writes a Markdown table to stdout suitable for
copy-pasting into docs/PHASE_1_RESULT.md.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from pipeline import manifest


DEFAULT_WORKDIR = Path.home() / "projects" / "xai-model-autopsy-workdir"


def build_table(rows: list[manifest.ManifestRow]) -> str:
    per_gp: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in rows:
        per_gp[(r.split, r.class_label)][r.patient_id] += 1

    lines = ["| split | class | total | patients | dominant patient (share) |",
             "|---|---|---:|---:|---|"]
    for (sp, cls), counter in sorted(per_gp.items()):
        total = sum(counter.values())
        top_p, top_ct = counter.most_common(1)[0]
        share = top_ct / total if total else 0.0
        lines.append(
            f"| {sp} | {cls} | {total} | {len(counter)} | "
            f"{top_p} ({top_ct} = {share:.1%}) |"
        )

    lines.append("")
    lines.append("## Per-patient breakdown")
    lines.append("")
    lines.append("| split | class | patient | patches |")
    lines.append("|---|---|---|---:|")
    for (sp, cls), counter in sorted(per_gp.items()):
        for p, n in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"| {sp} | {cls} | {p} | {n} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    args = p.parse_args(argv)
    rows = manifest.read(args.workdir / "manifest.csv")
    manifest.validate(rows, check_paths_exist=True)
    print(build_table(rows))


if __name__ == "__main__":
    main()
