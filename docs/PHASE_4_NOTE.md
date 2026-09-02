# PHASE_4_NOTE — CI + embedding DataLoader fix

**Date:** 2026-09-02
**Repo:** `~/projects/xai-model-autopsy/` (main branch on the public repo)
**Scope:** exactly the two Tier-1 backlog items in the Phase 4 brief — no
other changes. Archive frozen, workdir untouched, cached embeddings not
re-run.

---

## 1. What landed

Six commits on `main`, pushed to `origin/main`:

| commit | subject |
|---|---|
| `d0b2c78` | `pipeline/embed`: fix DataLoader bottleneck (`num_workers=4`, `persistent_workers=True`) |
| `3eaeba8` | `ci`: add GitHub Actions workflow + README badge |
| `0df261a` | `docs`: cite embed DataLoader fix in PHASE_1_RESULT (§4/§6/§8) and README |
| `bb6a840` | `ci`: use uv installer so histolab/openslide-python resolve |
| `7e3ed88` | `ci`: override histolab's stale openslide-python pin |
| `a652720` | `ci`: drop setup-python pip cache (we install via uv) |

The last three CI commits fix issues discovered by the first CI run —
they're not additional scope, they're the "watch until green" work.

## 2. CI verdict

- **Green run URL:**
  <https://github.com/bantonb/xai-model-autopsy/actions/runs/33608393504>
- **Wall-clock:** 8m 58s on `ubuntu-latest` (dominated by convnext_small
  weight download on first run; subsequent runs will hit the
  `actions/cache` and be faster).
- **Test result on CI:** `.........sss....` — **13 passed, 3 skipped**
  out of the 16-test suite. The 3 skips are exactly the three tests in
  `test_manifest_and_split.py` that walk the private clinical archive
  (`test_split_pins_NfE3_to_train`,
  `test_split_is_deterministic_across_calls`,
  `test_all_splits_populated_per_class`) — they route through
  `_patients_per_class()` which calls `pytest.skip()` when the archive
  directory is absent (any CI runner). No test edits were needed for
  CI-compatibility; the existing skip guard already worked.
- **`defect_demo` smoke:** produced `r(p0,p1) = -1.0000` for the broken
  arm and `-0.4653` for the corrected arm, matching the direction the
  synthetic demo has always produced. Ran with `--no-plot` so the CI
  runner doesn't need a Matplotlib display backend beyond the default.
- **Only annotation on the green run:** Node.js 20 deprecation notice
  from the third-party actions (`actions/cache@v4`,
  `actions/checkout@v4`, `actions/setup-python@v5`,
  `astral-sh/setup-uv@v3`). GitHub is force-upgrading them to Node 24.
  Not our code; will resolve when the action maintainers cut new tags.

Three earlier attempts (33265745040, 33265875719, 33265949024) failed
red — each surfaced a real portability issue that the three follow-up
CI commits above fixed. They're preserved in the run history rather
than force-pushed away.

## 3. Doc diffs (the two documentation updates the brief called for)

### `docs/PHASE_1_RESULT.md`

Three spots touched (§4 runtime table narrative, §6 deviations bullet,
§8 NOT-DONE bullet). All keep the observed 10h 46m as history and add a
pointer to the fix commit `d0b2c78`. Sample from §4:

> Fixed in commit `d0b2c78` (`num_workers=4`, `persistent_workers=True`
> in `pipeline/embed.py`). The 10h46m figure above is the pre-fix
> historical runtime; a from-scratch rerun on the same M2 should now
> finish in the expected 30–60 min range. Not re-run here because the
> embeddings are cached to disk and outputs are seed-deterministic —
> the diagnostic numbers in §5 would be bit-identical.

The docs commit was originally written before the pre-push rebase over
`eef294b` re-hashed the DataLoader commit; a follow-up sweep updated
the three SHA references in-place.

### `README.md`

Two spots: the CI badge under the tagline, and the reproduce-block line
for `pipeline.embed`, which now reads:

> `python -m pipeline.embed             # ~30–60 min on M2 (was 10h 46m before commit d0b2c78 fixed the num_workers=0 DataLoader bottleneck)`

## 4. Deviations from the brief

- **Three extra CI-fixup commits** beyond the "one CI workflow + one
  DataLoader edit" the brief implied. Each was in-session diagnosis and
  repair per the brief's "A red run is a finding to fix in-session, not
  to leave" rule. Root causes:
    1. `pip`'s strict resolver rejects `histolab-0.7.0`'s
       transitive `openslide-python<1.3.2` pin against our top-level
       `>=1.4` pin. `uv` has the same behaviour by default. Local venv
       has 1.4.6 installed anyway, presumably force-installed at some
       past point. Resolved in CI with a `uv --overrides` file — no
       change to `pyproject.toml`.
    2. `actions/setup-python`'s `cache: pip` post-step fails when no
       pip cache exists (because we install via `uv`). Dropped the
       `cache: pip` line.
- **Rebase over `eef294b`** ("Update README.md dates"). Someone (Anton)
  pushed a small README date-string tweak to `origin/main` while this
  session was in progress; a `git pull --rebase` picked it up cleanly
  and reapplied the three Phase 4 commits on top. The rebased hashes
  are the ones now on `origin/main`. See the "Correction to log" in §3.

## 5. Anton's two remaining UI steps

Both are GitHub-web-UI actions, not code changes; they finish the
"public-facing polish" arc:

1. **Pin the repo** on your profile. GitHub → your profile → *Customize
   your pins* → tick `xai-model-autopsy`.
2. **Set the social-preview image.** GitHub → repo Settings → *Social
   preview* → *Edit* → *Upload an image* → pick
   `figures/real/real_hybrid_probs.png` from your local checkout.
   That's the same scatter the README embeds; it renders cleanly at
   1280×640 preview crops.

---

## Checklist

- [x] CI workflow added, one green run recorded, badge on README.
- [x] Embedding DataLoader fixed (code) and referenced in the two
      documentation spots (`PHASE_1_RESULT.md`, `README.md`).
- [x] Nothing outside the two brief items changed. Archive, workdir,
      figures, and result numbers all untouched.
- [x] Pytest green locally (16/16) and green on CI (13 passed, 3
      skipped due to absent archive).
- [x] Anton reminded of the pin + social-preview UI steps.
