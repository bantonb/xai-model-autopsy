# Thesis summary — one page

This is a compressed record of the bachelor's thesis whose training
code and design decisions this repository audits.

## Task

Automatic patch-level classification of `schwannoma` vs `neurofibroma`
tissue from H&E-stained whole-slide images (`.ndpi`), with a third
`hybrid` class for slides that pathology reports labelled as
containing both tumour types.

## Data

- Private clinical archive of `.ndpi` slides organised by class
  (`Schwannome/`, `Neurofibrome/`, `Hybrid/`).
- Slide-level labels only — every patch inherits its slide's label.
  This is a **weak** labelling: a hybrid slide contains regions of
  pure schwannoma, pure neurofibroma, and interfacial tissue in
  varying proportions, but the pipeline sees the whole slide as one
  class (see `docs/design_notes.md` §3).
- **Thesis corpus** (per the thesis text): 371 WSIs (~27 GB) — 144
  neurofibroma / 189 schwannoma / 38 hybrid, from 30 / 28 / 15
  individuals.
- **Local archive subset** (what this machine holds and what the
  autopsy pipeline used for the real-data reproduction): 58 slides
  from 55 patients across the three classes; only **five** hybrid
  patients are present in this local subset, so the hybrid sample
  sizes for the autopsy are correspondingly small.
- Training corpus (per `Pipeline/PORTFOLIO_RECON_REPORT.md` §E.1,
  thesis-side): ~3.13M patches at 224×224, rebalanced to ~252k per
  class → ~504k for the final training loop.
- Person identifier is the leading token of the `.ndpi` filename
  (see `docs/design_notes.md` §4). Dotted-suffix codes like
  `NF 12.1` / `NF 12.2` are distinct patients, not sub-slides of
  one patient — verified against the archive's case lists (raw
  distinct = 30 NF / 28 S, matching the thesis's individual counts).

## Architecture

- **Backbone:** `timm.create_model("convnext_small", pretrained=True,
  num_classes=0)` → 768-dim feature vector.
- **Head:** MLP `Linear(768, 256) → ReLU → Dropout(0.273) →
  Linear(256, 2)`.
- **Loss:** `torch.nn.BCEWithLogitsLoss()`, treating each output
  neuron as an independent Bernoulli.
- **Label encoder:** `np.array([float(x == 0), float(x == 1)],
  dtype=np.float32)` — only ever emits `[1, 0]` or `[0, 1]`. The
  `[1, 1]` corner is unreachable.
- **Optimizer:** Adam, lr=0.000116, weight_decay=0.000027, hidden=256,
  dropout=0.273294, batch=128 — best config from a Ray Tune ASHA
  search (per `PORTFOLIO_RECON_REPORT.md` §C.5).
- **Thesis-reported metric:** `MultilabelF1Score(num_labels=2,
  average="macro")` = ~0.9582 on the thesis's validation split.

## XAI method selection

Six post-hoc methods were run on the thesis's ConvNeXt-Small +
two-neuron head, chosen to triangulate across explanation families:

- **Gradient-based:** Integrated Gradients, Input×Gradient, DeepLIFT,
  saliency — all sensitive to sign structure of the output.
- **Occlusion-based:** Captum Occlusion, sliding window over the
  input, per class-target.
- **Perturbation-based on features:** KernelSHAP on the 768-dim
  ConvNeXt embedding, isolating which backbone features the head
  actually uses.

The rationale for six methods was defensive: any single one could be
dismissed as an XAI artefact, but agreement across families implicates
the model rather than the explainer.

## Findings

All six methods converged on the same signature:

- Occlusion attribution maps for target class 0 and target class 1
  on the same patch are near-mirror images.
- IG / I×G / DeepLIFT produced near-identical attribution maps
  *across the three methods* despite different mechanisms — evidence
  of a shallow, low-diversity fit on a frozen representation rather
  than three independent explanations converging on the same signal.
- KernelSHAP on embeddings concentrates on ~5 of 768 features.
- Saliency highlights intercellular space rather than nuclei.

The cause is a two-neuron BCE head trained on mutually exclusive
one-hot labels, forcing p0 + p1 ≈ 1 as a learned invariant. Post-hoc
explanations of an implicitly-softmaxed head are structurally forced
to be mirror images across the two output neurons.

## Outlook

Two orthogonal fixes that a follow-up should try:

1. **Structural:** a single-logit sigmoid head, or an explicit
   two-class softmax if mutual exclusivity is intended, or an
   expanded label space with real `[1, 1]` and `[0, 0]` data.
2. **Verification:** every training run should include a smoke test
   that asserts the model can, in principle, output `[1, 1]` on some
   synthetic input. The absence of that test is what let the coupling
   go unnoticed for a full thesis cycle.

The full PDF is available on request.
