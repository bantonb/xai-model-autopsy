# Design notes — honest caveats

This file exists so future readers can distinguish "load-bearing choices"
from "arbitrary conventions" without re-deriving them from the code.
Anything that could later look like a bug because it was too briefly
justified belongs here.

## 1. Broken vs corrected arms — exact scope

| | Broken arm | Corrected arm |
|---|---|---|
| Pure-class patches ([1,0] / [0,1]) | yes | yes (same set) |
| Hybrid-slide patches labeled [1,1] | **no** — held out entirely | yes |
| Model architecture (768 -> 256 -> 2, ReLU, Dropout) | same | same |
| Loss (BCEWithLogitsLoss) | same | same |
| Optimizer (Adam lr=0.000116 wd=0.000027) | same | same |
| Batch size, epochs, seed | same | same |
| WeightedRandomSampler (inverse-frequency per label tuple) | same | same |
| Backbone (ConvNeXt-Small, frozen — .eval() AND requires_grad_(False)) | same | same |
| Embeddings cache (768-dim, precomputed once) | shared | shared |

The broken arm mirrors the thesis's original design: only pure-class
patches, only [1,0] / [0,1] targets. Hybrid slides are excluded entirely
so the broken arm never sees them at training time.

## 2. What the corrected arm buys — and what it costs

The corrected arm's extended label space (adding hybrid patches with the
[1,1] target) is not just a labelling change: it necessarily also
introduces **new tissue** — the hybrid-slide patches that the broken arm
never trains on. This is inherent to the fix. It cannot be factored out
by matching on tissue, because [1,1] is only a sensible target for tissue
that plausibly contains both classes.

Consequences to be honest about:

- The corrected arm's decoupling of p0 and p1 is not purely a label-space
  effect; it is a label-space effect on a modestly enlarged training
  distribution. Some fraction of the improvement is exposure to more
  varied tissue, not the [1,1] target per se.
- To isolate the label-space effect alone, one would need a "corrected
  arm on pure patches only" ablation — which is impossible by construction
  (a pure-class patch cannot be legitimately labelled [1,1]).
- Therefore the finding stated here is: *the coupling is removed under the
  extended labelling on hybrid-inclusive data*, not *removing the coupling
  is possible on the same data*.

The alternative — leaving the broken arm untouched and adding an entirely
synthetic [1,1] channel — was not attempted; it would require injecting
fabricated labels into pure-class tissue, which defeats the purpose.

## 3. Hybrid-slide patch label is a WEAK label

A patch extracted from a hybrid slide is not necessarily hybrid tissue
at the patch scale. A hybrid slide can contain regions of pure schwannoma
tissue, pure neurofibroma tissue, and interfacial hybrid tissue in
varying proportions. Assigning [1,1] to every patch from a hybrid slide
is a *slide-level weak label*. Some fraction of hybrid-labelled patches
in training are, in fact, pure-class tissue.

This is the same weakness the thesis's own labelling had (labels came
from slide-level pathology reports, not per-patch review). It is
documented here rather than fixed because per-patch review is out of
scope for a portfolio project.

## 4. Person-level split

- Person = the case identifier that appears before ' - ' / ' -' / '-' in
  a `.ndpi` filename. Examples: `S18 - 1_5_0.ndpi` -> `S18`;
  `NF 12.2 -1_1.ndpi` -> `NF 12.2`; `NfE3 - 1_0.ndpi` -> `NfE3`.
- **Dotted-suffix codes are distinct patients, not sub-slides of one
  patient.** The archive's per-class case lists
  (`Pipeline/case_lists/{neurofibrome,schwannone}_case_list.txt`) contain
  144 + 189 = 333 slide paths. Under the raw key (dots kept: `NF 12.1`,
  `NF 12.2`, `NF 12.3` counted separately), the distinct-code counts are
  **30 neurofibroma** and **28 schwannoma** — exactly the individual counts
  the thesis reports. Under the grouped key (dotted suffix stripped:
  `NF 12.*` collapsed to one), neurofibroma drops to 22, which does not
  match the thesis. Schwannoma has no dotted codes so both counts are 28.
  Verified 2026-08-21 by direct count of the case lists; the raw code is
  the correct patient key. Case-insensitive normalisation does not create
  spelling-variant collisions (`Nf7` vs `NF 17` are distinct once the
  numeric part is compared).
- Split fractions (target): train 0.70, val 0.15, test 0.15.
- Pinned rules (see `pipeline.split.PINNED_SPLITS`):
    - Hybrid class: `EH12, EH9 -> train; EH15 -> val; EH14, EH7 -> test`.
      Chosen to guarantee at least one hybrid patient in every split
      given only five hybrid patients exist.
    - Neurofibroma `NfE3` (27 slides — an outlier) -> train, so its
      slide-heavy signal never contaminates val/test.
- Non-pinned patients are bucketed by
  `sha256("{class}::{patient}::{seed}").hexdigest() % 10_000_000 / 10_000_000`
  compared against cumulative thresholds `[0.70, 0.85]`.
- `pipeline.manifest.validate` asserts no patient appears in more than one
  split. This assertion runs on every manifest load, not as a comment.

## 5. Per-patient cap (35% of split-class total)

For each (split, class), any patient whose patches exceed 35% of the
group's total is uniformly downsampled to `int(0.35 / 0.65 * others)`
where `others` is the total-minus-patient count. This solves
`x / (x + others) <= 0.35` for `x`.

**Where the cap is mathematically infeasible**: when a group has only one
patient, the cap is 100%; there is nothing to cap against. This applies
to the hybrid val split (only EH15, 100% by construction) and any pure
split that ends up with a single patient after person-level assignment.
The report table in `docs/PHASE_1_RESULT.md` names each such case.

## 6. Deviations from the thesis

- **Macenko stain normalisation: skipped.** The coupling defect is a
  labels/loss phenomenon and is independent of stain normalization. Both
  arms use identical preprocessing (ImageNet-standard resize + normalize),
  so the coupling comparison is unbiased. Adds ~1-2 h of wall clock to
  reinsert if desired.
- **Backbone is genuinely frozen** here. The thesis's trainer called
  `.eval()` but never `.requires_grad_(False)` and passed
  `self.parameters()` to Adam, so its backbone was fine-tuned end-to-end
  despite the thesis text describing a frozen backbone. This repo asserts
  the freeze in `tests/test_freeze.py`.
- **Metrics reported both ways** (argmax per class AND
  MultilabelF1(num_labels=2, macro)). The multilabel version is the one
  the thesis used and is kept for comparability; the argmax version is
  the one a naive reader would expect for a 2-class problem.

## 7. Torch / macOS pin (item 6 of gate approval)

`torch==2.4.x` is the last line to support MPS on macOS 12.3+.
`torch>=2.5` requires macOS 14.0+ for MPS — this box is on 13.2.1 and
would fall back to CPU on that version. `pyproject.toml` pins accordingly.

## 8. Data separation

- Repo (`~/projects/xai-model-autopsy/`) contains code, tests, docs,
  small figures. `.gitignore` blocks `*.npy`, `*.pt/.pth/.ckpt`, `*.ndpi`,
  `**/data/`, and images outside `figures/`.
- Workdir (`~/projects/xai-model-autopsy-workdir/`) holds all patches,
  embeddings, checkpoints, per-slide stats, and the manifest. Not a git
  repo. Regenerable from the archive + repo code.
- Archive path is read from the `XAI_AUTOPSY_ARCHIVE` env var and
  defaults to `~/Documents/Projects/Anton BA Thesis Project/Data 2`.
  On this machine that archive is read-only and untouched.
