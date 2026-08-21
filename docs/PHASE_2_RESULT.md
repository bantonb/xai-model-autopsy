# PHASE_2_RESULT — assembling the public-facing repo

**Date:** 2026-08-21 (Phase 2 session, ~1 h wall clock)
**Repo:** `~/projects/xai-model-autopsy/`
**Workdir:** `~/projects/xai-model-autopsy-workdir/`
**Archive:** untouched.
**Session:** interactive, Anton present, auto mode active.

Nothing pushed. No remote. Publication remains a Phase 3 decision.

---

## 0. Deliverables summary

- **2A — occlusion mirror-metric:** closed. Numeric result appended to
  `docs/PHASE_1_RESULT.md` §5d-quant, sub-claim 5d upgraded from
  QUALITATIVELY CONFIRMED to CONFIRMED (qualitative + quantitative).
- **2B — README:** rewritten from a stub to the full ~300-line
  portfolio-facing document per the brief's structure. Voice pass is
  Anton's.
- **2C — supporting docs:** `docs/THESIS_SUMMARY.md` (1 page) and
  `CITATION.cff` written. Design notes updated with the env-var change
  from 2E.
- **2D — figure curation:** occlusion set curated from 36 files down to
  6 (samples 02, 11, 13 × broken/corrected). Approved by Anton after
  visual tissue-content review. All figures well under 500 KB; no
  downscale required.
- **2E — pre-publication sweep dry-run:** results below. One path leak
  fixed; slide-code decision recorded (see §5).

## 1. Part 2A — Occlusion mirror-metric

Script: `pipeline/mirror_metric.py`. Same 18 samples as
`pipeline.diagnose` (same seed, same shuffle), same window/stride
(32×32/16, baseline 0). Runtime: ~20 min on MPS
(`workdir/logs/xai_mirror_metric.log`, backed by
`workdir/mirror_metric.json`, 5.9 KB).

`[run-verified workdir/mirror_metric.json]`

Per-arm distribution across 18 samples:

| arm | mean | std | min | max | median |
|---|---:|---:|---:|---:|---:|
| broken    | **−0.9967** | 0.0025 | −0.9991 | −0.9925 | −0.9977 |
| corrected | **−0.5890** | 0.2606 | −0.9065 | **+0.0607** | −0.6243 |

By source class (mean r):

| arm | schwannoma | neurofibroma | hybrid |
|---|---:|---:|---:|
| broken    | −0.9959 | −0.9976 | −0.9965 |
| corrected | −0.7582 | −0.6745 | **−0.3342** |

Every broken-arm sample has r < −0.99; the two class-target
attribution maps are near-perfect sign-inverses regardless of source
class. The corrected arm shifts every single sample toward
decoupling, most on hybrid patches (which are the patches the
corrected arm was actually trained on with `[1,1]` labels). One
hybrid sample (i=13) actually crosses the sign line to r = +0.061 —
the two class-target attribution maps are no longer inverses at all.

**Interpretation of the arm separation:** the qualitative claim of
sub-claim 5d is now quantitative. Broken and corrected distributions
do not overlap (broken max −0.992, corrected min −0.906) — a
clean separation on the metric that most directly probes the
mutual-exclusivity constraint. The **direction of the fix is
unambiguously right**; the **magnitude is class-dependent and
strongest where the fix trained on data (hybrid)**, which is the
expected pattern for a label-space intervention.

## 2. Part 2B — README

Replaced the stub. Sections landed per brief structure:

1. Title + tagline
2. Pitch (verbatim)
3. 60-second top fold: 3-sentence story, headline table, hero
   figure (`figures/real/real_hybrid_probs.png`), copy-pasteable
   demo command
4. "What this repo is about" (<200 words)
5. The defect (thesis paths + Sigmoid comment with the `freedome`
   typo preserved)
6. The diagnosis (per XAI family, one paragraph each, including
   embedded occlusion figure pair for sample 13)
7. Reproduction (this repo — synthetic + real, all headline tables)
8. What the fix does — and does not — do
9. Post-thesis code audit (both findings)
10. Limitations (all six sub-points)
11. Repo map + how to reproduce
12. License, author line, "full thesis PDF available on request"

Voice rules from brief followed: plain first-person, no
marketing adjectives, short varied sentences, numbers carry claims.
Voice pass remains Anton's.

**Numbers deviation from the brief's suggested top-fold text:** the
brief-suggested pitch mentioned "371 slides, ~27 GB, 3.1M patches".
Only the ~3.13M patches number is confirmed in
`Pipeline/PORTFOLIO_RECON_REPORT.md` §E; the slide count and total
volume are not verifiable from the archive (`Data 2/` on this machine
holds 282 `.ndpi` files / 4.1 GB, and the recon does not name a
global slide count). The README uses ~3.13M patches and describes the
data as "a private clinical archive of `.ndpi` whole-slide images"
without asserting a slide count. Flagging this as a **deviation for
Anton to correct in the voice pass** — if 371 / ~27 GB is the
canonical thesis-corpus number, plug it in; if not, the softer phrasing
is honest.

## 3. Part 2C — Supporting docs

- **`docs/THESIS_SUMMARY.md`** — one page, sections: Task, Data,
  Architecture, XAI method selection, Findings, Outlook. All numbers
  and paths sourced from `PORTFOLIO_RECON_REPORT.md` and
  `design_notes.md`. Ends with "full PDF on request".
- **`CITATION.cff`** — CFF v1.2.0, MIT-licensed, single author.
- **`docs/design_notes.md`** §8 updated to describe the
  `XAI_AUTOPSY_ARCHIVE` env var introduced in 2E.

## 4. Part 2D — Figure curation

Approach: use the 2A mirror-metric to pick occlusion samples
objectively; keep 3 samples × 2 arms = 6 files. Delta = corrected r −
broken r (larger positive delta = more improvement from fix).

Selected (approved by Anton after seeing all three pairs in-session):

| sample | source | broken r | corrected r | Δ |
|---:|---|---:|---:|---:|
| 02 | schwannoma   | −0.9926 | −0.5919 | +0.401 |
| 11 | neurofibroma | −0.9984 | −0.3403 | +0.658 |
| 13 | hybrid       | −0.9975 | +0.0607 | +1.058 |

One per class; sample 13 is the visually most-dramatic (only
sign-flip in the whole set). Every file well under 500 KB (max
~200 KB); no downscale needed.

Deleted from committed repo (still present in workdir if reneeded):
30 occlusion PNGs (`git rm` from `figures/occlusion/`, `git ls-files
figures/occlusion/` = 6). Real (3) and synthetic (3) figure sets
retained in full — all are plots, no tissue content, no size issues.

**Tissue-content check:** Anton reviewed sample 02 (schwannoma), 11
(neurofibroma), 13 (hybrid) patches directly in-session via
multimodal read. Content: H&E-stained stroma and nuclei, no
annotations, no visible slide codes. Filenames already carry
`sample{NN}_{class}_{hash}` — no slide identifier. Approved.

## 5. Part 2E — Pre-publication sweep + slide-code decision

### 5.1 Path leaks — fixed

- `pipeline/extract.py:35` and `tests/test_manifest_and_split.py:19`
  both hardcoded `/Users/antonburckhardt/...`. Replaced with an env
  var: `os.environ.get("XAI_AUTOPSY_ARCHIVE", str(Path.home() /
  "Documents" / "Projects" / "Anton BA Thesis Project" / "Data 2"))`.
  `git ls-files | xargs grep -In "/Users/antonburckhardt"` → **0
  hits**.

### 5.2 Token / secrets

- `hf_`, `sk-`, `api[_-]?key`, `token`, `secret` in committed files:
  only benign matches inside `.gitignore` comments.
- `burckhardt.anton|anton.burckhardt|@gmail` in committed files: no
  hits (git-config user email lives outside tracked files).
- `/home/user`: no hits.

### 5.3 Slide-code decision (the one gate this phase surfaces)

Two categories, decided separately with Anton in-session:

**In code + tests: KEEP AS-IS.**
- `pipeline/split.py:23-27` contains the six pinned patient IDs
  (EH7, EH9, EH12, EH14, EH15, NfE3). Reproducibility of the split
  requires them; without them, `pipeline.split` cannot regenerate
  the rules that guarantee ≥1 hybrid patient per split (only five
  hybrid patients exist in the archive).
- `pipeline/manifest.py` docstrings and `tests/test_manifest_and_split.py`
  use the same codes as parse examples and fixtures.
- Rationale (Anton): the codes are opaque identifiers to any reader
  without the clinical archive; redacting them breaks reproducibility
  for no publication-safety gain.
- `docs/design_notes.md` also references the same codes as
  documentation-of-code; left as-is for consistency with the code
  decision.

**In `docs/PHASE_1_RESULT.md`: STRIP §9 + HASH pinned rules in §2-§3.**
- Original PHASE_1_RESULT preserved in workdir at
  `PHASE_1_RESULT.unredacted.md` (20.5 KB), untouched.
- Committed copy: §9's 55-row per-patient breakdown removed;
  patient codes in §2 (pinned rules) and §3 (aggregate table + cap
  outcomes + infeasibility list) aliased to `H1..H5` (hybrid),
  `S1..S4` (schwannoma), `N1..N5` (neurofibroma) — sorted
  numerically within class. A "Note on patient identifiers" block
  at the top of the file makes the aliasing explicit.
- Rationale (Anton): the aggregate §3 table remains auditable in the
  committed copy; the per-patient granularity is workdir-only.

**Remaining post-strip sweep on docs:**
- `docs/PHASE_1_RESULT.md` still matches the pattern `\bS[0-9]{1,2}\b`
  on the aliases `S1`, `S2`, `S3`, `S4` — false positives, not the
  real S-codes (which look like `S18`, `S53`, `S10`). Judged safe.
- `docs/design_notes.md` retains real slide codes per the code
  decision above.

### 5.4 Verification after all changes

- `pytest tests/` → **16 passed in 143 s** (same as Phase 1;
  `test_freeze.py` still asserts backbone freeze).
- `git ls-files | xargs grep -InE "hf_|/Users/antonburckhardt|/home/user"`
  → 0 hits.

## 6. Deviations from the brief, reported not absorbed

- **README slide-count softening** — see §2 note above. Deviated
  from the brief's headline pitch because the specific numbers were
  not verifiable in the archive; flagged rather than fabricated.
- **Design notes not stripped** — the brief listed
  `docs/PHASE_1_RESULT.md` as the one doc named for slide-code
  scrubbing. `docs/design_notes.md` also contains codes; Anton's
  "keep in code" decision was interpreted to cover
  code-documentation too, so it is left as-is. If publication review
  wants design_notes hashed as well, that is a one-pass edit at
  Phase 3.
- **Mirror-metric runtime overshot** — 20 min actual vs the ~5 min
  gate estimate. Same root cause as Phase 1 embedding: single-worker
  data loading around Occlusion's forward passes. Numbers unaffected
  (deterministic on seed 20260820); did not fix because embeddings
  are cached and the mission is documenting the phenomenon, not
  throughput.

## 7. Definition of Done — checklist

- [x] Occlusion mirror-metric computed; §5d-quant added to
      `docs/PHASE_1_RESULT.md`; sub-claim 5d updated to CONFIRMED.
- [x] README complete, every figure it references present in the
      repo (`figures/real/real_hybrid_probs.png`,
      `figures/occlusion/occ_broken_sample13_hybrid_11c94f8a.png`,
      `figures/occlusion/occ_corrected_sample13_hybrid_11c94f8a.png`).
- [x] `docs/THESIS_SUMMARY.md` written.
- [x] `CITATION.cff` written.
- [x] Occlusion figure set curated (6 files kept, 30 removed);
      tissue content approved by Anton via in-session visual review.
- [x] Pre-publication sweep results recorded here (§5); slide-code
      decision documented per category.
- [x] `pytest tests/` = 16 passed after all edits.
- [x] Path leak fixed and re-verified.
- [x] Workdir preserves the unredacted `PHASE_1_RESULT.unredacted.md`.
- [x] Nothing pushed. No remote. `git remote -v` = empty.
- [x] Deviations reported (§2 slide count, §6).
- [ ] Anton's voice pass on the README (Phase 2 finishing step;
      then Phase 3 for review + publication).

## 8. Handover to Anton

Bring the README here for the voice pass. The three specific
places to look at first:
1. The 60-second top-fold — story sentences and hero figure pick.
2. §Data description — soften/tighten the "private clinical archive
   of `.ndpi`" phrasing if the exact scale is publishable.
3. §Post-thesis code audit — the tone should read as verification-of-
   own-work, not blame; verify it lands that way.

## 9. Handover addendum — 2026-08-21 close-out

Phase 2 is accepted. The design-chat voice pass produced a
consolidated edit list of **12 items** against README/docs, and
surfaced one open **patient-key question**: whether dotted
neurofibroma codes like `NF 12.1` / `NF 12.2` denote the same
underlying patient (indexing sub-slides) or distinct patients
sharing a case number. If they collapse to one patient, the
person-level-split claim needs re-verification because two rows the
manifest currently treats as separate patients would land in the
same person bucket. Both the 12-item edit list and the patient-key
question are **owned by PHASE_3_BRIEF**, not by this phase; nothing
in Phase 2 is being retouched. Phase 3 runs in a fresh session.
