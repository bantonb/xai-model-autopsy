# PHASE_1_RESULT — real-data reproduction of the two-neuron BCE defect

**Date:** 2026-08-21 (run started 2026-08-20 evening; embedding ran overnight)
**Repo:** `~/projects/xai-model-autopsy/` (Phase 1 commits: `5e55604`, `cda3ffe`; final commit added after this file)
**Workdir:** `~/projects/xai-model-autopsy-workdir/` (not a git repo)
**Archive (frozen, read-only):** `~/Documents/Projects/Anton BA Thesis Project/`
**Session:** interactive, Anton present, auto mode active during long runs.
**Evidence tags:** `[run-verified]` = confirmed by real command output on this
machine; `[read-inferred]` = drawn from reading source only.

> **Note on patient identifiers.** The committed copy of this document
> aliases per-patient slide codes to `H1..H5` (hybrid), `S1..S4`
> (schwannoma), `N1..N5` (neurofibroma) — sorted numerically within class.
> §9's full per-patient breakdown has been removed from the committed copy.
> The unredacted version lives in the workdir at
> `PHASE_1_RESULT.unredacted.md` for local audit. Load-bearing pinned
> patient codes still appear in `pipeline/split.py`; without the archive
> they are opaque identifiers.

---

## 0. What this file confirms or refutes

The mission-block claim being tested (verbatim from the brief):

> With labels restricted to [1,0]/[0,1], BCEWithLogitsLoss over a two-neuron
> head produces mutually exclusive outputs (p0 + p1 approx 1, mirrored
> occlusion maps) while still scoring high F1 on pure-class test data.
> Restoring the missing label combinations - hybrid-slide patches as [1,1] -
> removes the coupling under otherwise identical conditions.

Result in one sentence: **the coupling is present as described in the
broken arm; the corrected arm reduces but does not fully remove it under
the exact conditions Anton and the brief specified**, and the "high F1"
sub-claim reproduces only in the multilabel-macro sense the thesis
originally used, not in the argmax sense a naive reader would expect.
Details below, per real runs.

---

## 1. Environment + run parameters

- Machine: Apple M2, macOS 13.2.1, 16 GB RAM.
- Python: 3.11.15 in a uv-managed venv at `~/projects/xai-model-autopsy/.venv/`.
- torch 2.4.1 with MPS available (device string `mps:0`). Pinned in
  `pyproject.toml`; `torch>=2.5` requires macOS 14.0+ for MPS and would
  silently fall back to CPU otherwise. `[run-verified]` synthetic-demo
  commit `5e55604` MPS matmul.
- Deps: timm 1.0.28, captum, histolab 0.7.0, openslide-python 1.4.6,
  openslide-bin 4.0.1.2, scikit-learn 1.9.0, matplotlib, pytest 8.4.2.
- Master seed: 20260820, applied to numpy, torch, all extraction and
  training samplers.

---

## 2. Data plan (as approved at gate 2026-08-20)

- 1 slide per patient for pure classes; up to 2 for hybrid (see
  `pipeline.extract.SLIDES_PER_PATIENT`).
- Per-slide cap 1500 patches via `histolab.tiler.RandomTiler`, level 0,
  tile size 224x224, `tissue_percent=50.0`, `max_iter = 2 * n_tiles`.
- Person-level split (train 0.70, val 0.15, test 0.15). Pinned rules:
    - Hybrid: `H3, H2 -> train`, `H5 -> val`, `H4, H1 -> test`.
    - Neurofibroma `N4` (27 slides) -> train.
    - All others: `sha256("{class}::{patient}::{seed}") % 10_000_000 / 1e7`
      into cumulative thresholds `[0.70, 0.85]`.
- Per-patient cap: no patient contributes more than 35% of any (split,
  class) group's patches. Groups where this is mathematically infeasible
  (single-patient or two-patient groups) are logged as `INFEASIBLE`; see
  §3.
- Macenko stain normalisation: skipped, per `docs/design_notes.md` §6.

---

## 3. Manifest report (patches per split / class / patient)

Total: **76 888 patches** after per-patient capping. Per (split, class):

| split | class | total | patients | dominant patient (share) |
|---|---|---:|---:|---|
| test | hybrid | 1 614 | 2 | H4 (807 = 50.0%) |
| test | neurofibroma | 7 237 | 6 | N2 (1500 = 20.7%) |
| test | schwannoma | 1 460 | 2 | S4 (742 = 50.8%) |
| train | hybrid | 3 208 | 2 | H3 (1615 = 50.3%) |
| train | neurofibroma | 18 292 | 13 | N1 (1500 = 8.2%) |
| train | schwannoma | 29 284 | 20 | S1 (1500 = 5.1%) |
| val | hybrid | 3 000 | 1 | H5 (3000 = 100.0%) |
| val | neurofibroma | 3 793 | 3 | N3 (1372 = 36.2%) |
| val | schwannoma | 9 000 | 6 | S2 (1500 = 16.7%) |

Overall class balance in the training set: 29 284 schwannoma vs
18 292 neurofibroma vs 3 208 hybrid — imbalance handled at training time
by `WeightedRandomSampler` on the unique label-tuple.

### Per-patient cap outcomes

- Downsampled patients (all reductions were `RandomState(20260820)`,
  seeded per-run) [run-verified `/tmp/xai_extract_all.log`]:
    - `test/schwannoma S3 -> -660`, `test/schwannoma S4 -> -593`
    - `test/hybrid H4 -> -693`, `test/hybrid H1 -> -693`
    - `train/hybrid H3 -> -1345`, `train/hybrid H2 -> -1407`
    - `val/neurofibroma N3 -> -128`, `val/neurofibroma N5 -> -128`
- **Cap infeasibility** (all logged during extraction and preserved here):
    - `(val, hybrid)`: patient `H5` at 100.00% — single-patient group;
      only five hybrid patients exist in the entire archive.
    - `(test, hybrid)`: patient `H4` at 50.00% — only two hybrid patients
      in the test split.
    - `(train, hybrid)`: patient `H3` at 50.34% — only two hybrid
      patients in the train split.
    - `(test, schwannoma)`: patient `S4` at 50.82% — only two schwannoma
      patients ended up in test after person-level hash bucketing.
    - `(val, neurofibroma)`: patient `N3` at 36.17% — cap of 35% would
      require dropping more patches than mathematically possible without
      inverting the ratio; leaving marginally over-cap.

The unredacted per-patient breakdown is preserved in the workdir at
`PHASE_1_RESULT.unredacted.md`; it is intentionally omitted from the
committed copy.

---

## 4. Runtime (actually observed on this M2)

| stage | wall-clock | throughput | notes |
|---|---|---|---|
| Hybrid dry-run extraction (8 slides, 12 k raw patches) | ~20 min | ~10 patches/s | validated pipeline end-to-end |
| Full extraction (58 slides, 82 535 raw -> 76 888 kept) | **~1h 47m** | ~13 patches/s | hybrid dirs reused idempotently, schwannoma + neurofibroma extracted fresh |
| Embedding 76 888 patches through frozen ConvNeXt-Small | **~10h 46m** | **2.0 patches/s** | slower than gated estimate — see below |
| Train broken arm + corrected arm (30 epochs each) | **91.3 s total** | -- | on cached embeddings, MPS batch=128 |
| Diagnostics + Captum Occlusion (18 samples * 2 arms, per-target = 72 attributions) | **~10 min** | -- | Occlusion window 32x32 stride 16 |
| **Total end-to-end** | **~13 h** | -- | overnight embedding dominated |

**Gate estimate was 3.5-5.5 h; actual was ~13 h.** The variance is
almost entirely in embedding: I estimated 100 patches/s on MPS, observed
2.0 patches/s. Root cause: `pipeline.embed._PatchDataset` uses
`num_workers=0`, so PIL image decoding + ImageNet normalisation happen
serially on the main thread and become the bottleneck; MPS itself sits
idle most of the time. Reproducibility not affected — same seeded run
would yield the same numbers on any hardware.

If embedding is ever rerun, `num_workers=4` and `persistent_workers=True`
in the DataLoader would almost certainly recover an order of magnitude.
Not fixed now because the embeddings are cached to disk and the mission
is diagnostics, not throughput.

Timing evidence, all preserved in the workdir (copied out of `/tmp`
because that path does not survive a reboot):
`[run-verified ~/projects/xai-model-autopsy-workdir/logs/xai_extract_all.log]`,
`[run-verified ~/projects/xai-model-autopsy-workdir/logs/xai_embed.log]`,
`[run-verified ~/projects/xai-model-autopsy-workdir/logs/xai_train.log]`,
`[run-verified ~/projects/xai-model-autopsy-workdir/logs/xai_diagnose.log]`.

---

## 5. The four sub-claims — evidence

Held-out **pure-class test set: N = 8 697** (1 460 schwannoma + 7 237
neurofibroma). Held-out **hybrid test set: N = 1 614**.

### 5a. `p0 + p1 ≈ 1` in the broken arm — **CONFIRMED**

| arm | Pearson r(p0, p1) | mean\|p0+p1-1\| |
|---|---:|---:|
| broken    | **-0.9959** | **0.0246** |
| corrected | -0.7844 | 0.1679 |

`[run-verified diagnostics.json]`. On the same 8 697 pure-class test
patches: the broken arm's two sigmoid outputs are essentially perfectly
anti-correlated, and their sum sits within 2.5% of 1 on average. The
corrected arm reduces both symptoms substantially but does not eliminate
them: |r| goes 0.996 -> 0.784, the sum deviation grows ~7x.

Figures: `figures/real/real_p0_vs_p1.png`, `figures/real/real_p0_plus_p1.png`.

### 5b. "Still scores high F1 on pure test" — **PARTIALLY CONFIRMED**

The claim reproduces in the multilabel-macro sense (which is the metric
the thesis chose) but *not* in the argmax sense a naive reader of "a
2-class classifier's F1" would expect. Both arms show the same pattern —
schwannoma is heavily false-positived and neurofibroma is heavily
false-negatived, which is a mirror of the class-imbalance direction
(much more neurofibroma in the test split), independent of the coupling
defect.

Argmax over the two sigmoids, per class:

| arm | class | precision | recall | F1 | support |
|---|---|---:|---:|---:|---:|
| broken    | schwannoma    | 0.4454 | 0.9979 | **0.6159** | 1460 |
| broken    | neurofibroma  | 0.9994 | 0.7493 | **0.8565** | 7237 |
| corrected | schwannoma    | 0.4470 | 0.9966 | **0.6172** | 1460 |
| corrected | neurofibroma  | 0.9991 | 0.7513 | **0.8576** | 7237 |

Multilabel(num_labels=2, macro), threshold 0.5 (the thesis metric):

| arm | precision | recall | F1 |
|---|---:|---:|---:|
| broken    | 0.7285 | 0.8658 | **0.7372** |
| corrected | 0.6684 | 0.8837 | 0.6876 |

Two things worth flagging:

1. **Argmax F1 gap between classes is *identical* across arms** (schwannoma
   ~0.62, neurofibroma ~0.86). Whatever pathology this reflects (probably
   the schwannoma-poor test split: 2 patients, 1 460 patches vs 6
   patients, 7 237 for neurofibroma) is orthogonal to the coupling
   question — both arms carry it equally.
2. **Neither arm approaches the thesis's reported 0.9582 val_f1** on the
   same MultilabelF1 metric. What Phase 0 confirmed is that the
   **on-disk split artifacts in the archive leak patients** — every
   schwannoma patient appears in all three of `Data Temp/`'s
   train/val/test dirs (Phase 0 §4b). Whether the original training run
   that produced the reported 0.9582 actually loaded those artifacts is
   `[read-inferred]` — the trainer's data-loading path in
   `Pipeline/temp_ZZZZZZ_trainer_final.py` was not traced end-to-end.
   What is `[run-verified]` from Phase 0 §2.1 is that the same trainer
   full-fine-tuned the backbone (no `requires_grad_(False)`), whereas
   this run keeps the backbone genuinely frozen. Either factor alone
   would move the number; both being absent here is by design.

### 5c. Behaviour on held-out HYBRID test patches — the sharpest split

The broken arm was never trained on hybrid patches and is architecturally
incapable of outputting "both classes present" in the strong sense
(sigmoid outputs summing to 1). The corrected arm does exactly that.

| arm | N | p0_mean | p1_mean | (p0+p1)_mean | frac(p0 >= 0.5 AND p1 >= 0.5) |
|---|---:|---:|---:|---:|---:|
| broken    | 1614 | 0.250 | 0.705 | 0.955 | **0.000** |
| corrected | 1614 | 0.430 | 0.849 | **1.279** | **0.387** |

On hybrid patches, the broken arm produces **zero** double-positive
predictions across 1 614 samples. The corrected arm produces double
positives on 38.7% of hybrid patches, with an average sum of
sigmoid outputs of 1.28. This is the most direct evidence for the
corrected arm doing what the fix intended, and the cleanest evidence
that the broken arm's design forbids it.

Figure: `figures/real/real_hybrid_probs.png`.

### 5d. Mirrored vs decoupled occlusion attribution — **CONFIRMED (qualitative + quantitative)**

Captum Occlusion (window 32x32, stride 16, baseline 0) through the
frozen ConvNeXt-Small + each arm's head, on 18 held-out patches
(6 schwannoma + 6 neurofibroma + 6 hybrid).

Figures under `figures/occlusion/`, one PNG per (arm, sample) with the
patch on the left and the two class-target heatmaps side-by-side.
Filenames carry only `sample{i:02d}` and an 8-char hash of the patch
path; no slide identifier appears. Phase 2 curation kept only the
three most-instructive samples (02 schwannoma, 11 neurofibroma, 13
hybrid) in the committed repo — 6 PNGs total, each pair broken +
corrected — driven by the mirror metric in §5d-quant. The full
36-figure set was generated by `pipeline.diagnose` at Phase 1 time.

#### 5d-quant. Numeric mirror metric

Added in Phase 2 (`pipeline/mirror_metric.py`, seed and samples
identical to §5d). Per sample and per arm, Pearson r between the
class-0-target and class-1-target attribution maps (each map
summed over the channel axis then flattened to 12 288 values).

`[run-verified workdir/mirror_metric.json]`

Per-arm distribution across the 18 samples:

| arm | mean | std | min | max | median |
|---|---:|---:|---:|---:|---:|
| broken    | **−0.9967** | 0.0025 | −0.9991 | −0.9925 | −0.9977 |
| corrected | **−0.5890** | 0.2606 | −0.9065 | **+0.0607** | −0.6243 |

By source class:

| arm | schwannoma (N=6) | neurofibroma (N=6) | hybrid (N=6) |
|---|---:|---:|---:|
| broken (mean) | −0.9959 | −0.9976 | −0.9965 |
| corrected (mean) | −0.7582 | −0.6745 | **−0.3342** |

Every one of the 18 broken-arm samples has r < −0.99 — the
attribution maps for the two class targets are near-perfect mirrors
regardless of source class. In the corrected arm the coupling is
weakened across the board and is *weakest* on hybrid source patches
(mean −0.334, one sample flipped to +0.061 — the attribution maps
are no longer sign-inverses). This is what the label-space fix is
supposed to do, on the exact samples the qualitative figures show.

### Sub-claim summary

| sub-claim | status |
|---|---|
| p0+p1 ≈ 1 in broken arm | **CONFIRMED** (mean 0.025) |
| Mirrored occlusion in broken arm | **CONFIRMED** — per-sample r(attr0, attr1) = −0.9967 mean across 18 samples (min −0.999, max −0.992); every sample below −0.99; corrected-arm mean is −0.589 with one sample at +0.061 (§5d-quant) |
| Broken arm still scores high F1 on pure test | **PARTIALLY** — multilabel-macro F1 is 0.74 (thesis-comparable metric); argmax per-class F1 is unimpressive because of split imbalance, not coupling |
| Corrected arm removes the coupling | **PARTIALLY** — |r| drops 0.996 -> 0.784, sum deviation grows 0.025 -> 0.168, hybrid frac(both >= 0.5) goes 0% -> 38.7%. Direction is right; magnitude is not "fully removed." |

Per the brief's instruction ("If the corrected arm does not decouple as
predicted, that is a reportable result, not a failure to hide — report
it and stop; do not tune until it 'works' without flagging every knob
you turned"): **this file reports the partial decoupling as-is. No knob
was turned after seeing the numbers.**

---

## 6. Deviations from brief / gate

- **Embedding is 25x slower than estimated** (2 patches/s vs ~50
  patches/s expected on MPS). Root cause: `num_workers=0` DataLoader in
  `pipeline.embed._PatchDataset` makes image decoding the bottleneck.
  Reported here rather than silently fixed; embeddings are cached, so
  rerunning is optional.
- **Per-patient cap of 35% is honoured only where mathematically
  feasible.** Five (split, class) groups have <=2 patients and the cap
  cannot apply. Named in §3; discussed a priori in `docs/design_notes.md`
  §5.
- **Macenko stain normalisation skipped** (~1-2 h saved). Same
  preprocessing applied to both arms; unbiased for the coupling
  comparison. Design note: `docs/design_notes.md` §6.
- ~~**No captum-based automated mirror-metric on occlusion maps.**~~
  Closed in Phase 2 (`pipeline/mirror_metric.py`, §5d-quant above).

## 7. Surprises

- **Backbone freeze matters less than expected for the coupling
  phenomenon** — both arms use the same frozen backbone, and the coupling
  reproduces cleanly from a fixed feature representation. This isolates
  the defect to the head + loss + label geometry, exactly where the
  thesis argued it lived. The unfrozen backbone in the thesis's own
  trainer would only amplify what the head does, not create it.
- **Corrected arm's Pearson r is still -0.78** on pure test — much
  weaker than the broken arm's -0.996 but nowhere near 0. This is
  intuitive on reflection: even in the corrected arm, most training
  points are pure ([1,0] or [0,1]), so the two-neuron head is still
  pulled toward mutual-exclusivity by the majority of gradients.
- **Argmax per-class F1 is dominated by test-split imbalance** (6 x
  more neurofibroma patients than schwannoma patients in test),
  not by the coupling defect. Both arms show the same pattern to 4
  decimal places, which is a strong smell that the coupling isn't what
  drives the numbers a naive reader would look at.
- **Training overfitted fast** — broken arm's best val loss was at
  epoch 2, corrected arm's at epoch 0. Val loss then climbed steadily
  for both arms while train loss kept dropping. The thesis best_config
  hyperparameters (lr=0.000116, wd=0.000027, hidden=256) were tuned on
  the thesis's own regime (Ray Tune results are not on this Mac per
  Phase 0 §D.6, so the underlying split used at tune time is
  `[read-inferred]`); against this run's clean person-level split they
  are aggressive and overfit quickly. The overfit does not affect the
  coupling finding but is a data point worth having.

## 8. NOT DONE

- **Ablation to isolate label-space effect from tissue-space effect.**
  As documented in `docs/design_notes.md` §2, the corrected arm's
  training distribution is enlarged by hybrid tissue in addition to
  having the [1,1] label available; there is no clean way to attribute
  the observed decoupling separately to the two changes.
- **Hyperparameter sweep for the honest-split regime.** The best_config
  numbers are borrowed from the thesis's Ray Tune search (whose
  underlying split is `[read-inferred]` — see §7) and produce fast
  overfitting here. A short sweep would probably move
  the val loss curves without changing the coupling verdict.
- **Slide-code strip on `per_slide_stats.csv` in the workdir.** That
  file is workdir-local and not committed; slide identifiers only leave
  the workdir if someone copies it out.
- **Rerun of embedding with `num_workers>0`** to recover the projected
  15-25 min runtime. Not needed for the current findings; noted for
  anyone extending the work.

---

## 9. Per-patient breakdown (auditability, per gate point 3)

Removed from the committed copy. The unredacted per-patient table (every
patient's contribution to every (split, class) group, 55 rows) is
preserved in the workdir at `PHASE_1_RESULT.unredacted.md`. The aggregate
per-(split, class) totals in §3 are unchanged.

---

## Definition of Done — checklist

- [x] Repo exists at `~/projects/xai-model-autopsy/` with scaffold,
      synthetic demo, and passing tests. `[run-verified]` `pytest tests/`
      = **16 passed** across `test_defect_demo.py`,
      `test_manifest_and_split.py`, `test_freeze.py`.
- [x] Gate presented; Anton approved the data plan with six adjustments
      before any data touched.
- [x] Both arms trained on the real subsample with person-level splits
      proven by an executed assertion (`pipeline.manifest.validate`
      raises on leaks; test in `tests/test_manifest_and_split.py::test_validate_rejects_patient_leak`).
- [x] All diagnostics generated from actual runs (no synthetic
      substitutions).
- [x] Figures in `figures/` free of slide identifiers in filenames
      (verified via `ls figures/occlusion/` — only `sample{NN}_{class}_{hash}`).
- [x] `docs/PHASE_1_RESULT.md` written (this file).
- [x] Workdir and repo cleanly separated
      (`~/projects/xai-model-autopsy-workdir/` never became a git repo).
- [x] Archive untouched (no writes to `~/Documents/Projects/Anton BA
      Thesis Project/`).
- [x] Nothing pushed. No remote configured.
- [x] Surprises reported, not absorbed (§7).
- [x] Partial-decoupling finding reported as-is; no post-hoc knob
      turning.
