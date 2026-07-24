# CLAUDE.md — Gugak Stem Separation
<!-- STABLE project truth only. The living plan (roadmap, current stage, decisions, evolving
     strategy) lives in Notion — see "Plan & tracking" below; do NOT re-add a roadmap here.
     Personal working style, sym8/compute etiquette, uv rules, and general guardrails live in
     ~/.claude/CLAUDE.md. Commit format + tags live in the git-commit skill. -->

## Plan & tracking — Notion is the single source of truth
The **roadmap, current stage, decision log, and evolving strategy** (stem-class scheme,
augmentation recipe, experiment queue, survey findings) live **only in Notion** — never here.
This file is intentionally roadmap-free so it can't drift out of sync the way it did before.

- **Fetch at session start, and before any planning/status discussion:** Notion page
  **Gugak Stem Separation** — id `3a4cb377-8743-8136-8f9a-c2bc24ef8269`
  (https://app.notion.com/p/3a4cb377874381368f9ac2bc24ef8269).
  Child pages: **Experiments** `3a4cb377-8743-8012-921b-e6174eb0effb` ·
  **Survey** `3a4cb377-8743-8075-932e-ca518395c3e1`.
- **Keep it live:** update the Notion page in the SAME turn a plan changes, an item completes, or
  a decision is made. (The old CLAUDE.md-as-living-planner discipline now applies to Notion.)
- **On any conflict, Notion wins** — it is the newer, authoritative plan. This file may lag on
  anything plan-shaped; trust Notion for that, trust this file for stable facts/conventions.

## Project
Stem separation on a traditional Korean music (gugak) multitrack dataset
(AI Hub **국악합주곡 디지털 음원 데이터**, datasetkey 71955). First task in the MALer lab
archiving/cataloguing workstream; also a full experiment-cycle practice run:
data → baseline → fine-tune → compare → share.

## sym8 power-off — 2026-08-01 (routine, ~1 day)
Building maintenance during summer break: **sym8 loses power for ~1 day on 2026-08-01**, back up
same day or next. **Disks survive — this is NOT a decommission.** No backups or data evacuation
needed; `data/`, wandb offline runs, and the uv env all persist.
**Only ask:** leave things in a clean, resumable state — no jobs mid-flight, work committed + pushed.
Remove this block once the power-off has passed.

## Environment
- `uv` project, Python 3.11 (uv usage rules → global CLAUDE.md).
- `ffmpeg` = system dependency (provided by server), not a Python dep.
- PyTorch cu128 via the PyTorch index (uv), pinned in `pyproject.toml`.
- **GPU (confirmed 2026-07-19):** sym8 driver 590.48 (CUDA 13.1); PyTorch **cu128** build
  (Blackwell sm_120 — hardware details in global CLAUDE.md).

## Dataset  (full status → `docs/dataset_status.md`)
- **903 songs** with audio (published 1,004 minus the 101 withheld-test songs); 1 master + N stems
  each; **6,670 WAVs**. 48 kHz / 24-bit / stereo, **except 130 files at 96 kHz** (창작국악 → resample).
- **On disk:** `data/extracted/<dl_folder>/source/<song>/` = `<song>_master.wav` + `<song>_<instrument>.wav`.
  `data/` is a symlink → `~/storage/nia-gugak`.
- **Manifest = source of truth** (never walk directories): `manifests/songs.parquet` (903) +
  `stems.parquet` (5,767) [+ csv mirrors], built by `src/data/build_manifest.py`.
- **Our own split** (publisher's split CSVs/folders are inconsistent & incomplete — ignored):
  song-level, stratified by genreSub, **seed 42, 80/10/10** = train 721 / val 91 / test 91. Frozen in the manifest.

### Key facts & gotchas
- **No vocal/소리 stem** anywhere (incl. 판소리 — its masters also appear voiceless) → **no voice
  stem in any scheme.**
- **master ≠ Σ(stems) (verified, Phase 2).** Build training mixtures by **summing stems**
  (mix:=Σstems, targets:=stems — self-consistent). The real master is NOT the sum: mastered
  (per-stem faders + reverb/FX), genre-dependent (민요 ≈ clean sum, 창작국악 heavily processed).
  → master = **secondary real-world eval only**; do NOT invert master→stems.
- **Our mixture recipe:** mix := Σ(stems) → **trim all stems to the shortest per song** →
  **peak-normalize the MIXTURE to 0.99** (never per-stem — preserves the balance). Native SR
  (96 kHz resampled to model SR at inference). This is OUR mixture (our analog to a "master"),
  distinct from the dataset's real mastered master. Code: `src/data/audio.read_and_sum`
  + `scripts/build_mixtures.py`.
- **Stem QC (Phase 2 sample) = clean:** 0 dead stems, clipping negligible (drop the check),
  loudness ~−19 LUFS. **Sparse ≠ dead** — do NOT auto-drop low-activity stems (박 etc. play
  rarely by design). Normalize the mixture, never per-stem. No full scan needed.
- **판소리 length-align (verified, Phase 2):** stems are start-aligned; in 28 판소리 songs one stem
  (almost always **가야금**) runs *longer* than the master (usually <0.5 s, one 11 s tail) with real
  (non-silent) content the master lacks. Rule: **trim every stem to the shortest per song before
  summing** (don't pad — padding injects content the master never had). 판소리-only.
- **Multi-instrument stems** exist (피리1/피리2/피리3…) → strip trailing digit to base, **sum same-base
  into one source**. (향/세/당피리 subtype ambiguity — experts pending.)
- **Korean filenames need NFC** normalization for any name join. **Master naming translation:**
  on disk `<song>_master.wav` vs metadata records `<song>.wav`.

## Repo Conventions
- **Experiment configs:** one per experiment (`configs/*.yaml`) = one wandb run; change configs,
  not code, for hyperparameters — prefer editing configs over editing code for experiment variation.
  **Shared/cross-experiment configs** (e.g. `configs/listening_set.yaml`) also live in `configs/`.
  If it grows, split `configs/experiments/` vs `configs/shared/`.
- Run naming: `expNNN_<model>_<stemscheme>_<key-hparam>` (e.g. `exp003_bsroformer_4stem_lr1e-5`).
- **Experiment folders** (`experiments/`) carry a `YYMMDD` date stamp: named exps = `<name>_<date>`
  (e.g. `zeroshot_baseline_260719`); numbered exps = `exp<NNN>_<date>_<rest>` (date right after the
  number, e.g. `exp001_260722_htdemucs_4stem`). Track parquet/csv/md; figures are gitignored
  (regenerable from tracked metrics + script). Full convention → `experiments/README.md`.
- Log git commit hash + full config to wandb for every run.
- Manifests, not directory walking. Split frozen in the manifest; commit it. Never let chunks of one
  song cross splits (audio leakage).
- Notebooks (`notebooks/`) = EDA only.

## Commits
- Format + tag taxonomy → **`git-commit` skill.** No repo-specific tag overrides currently.
- **NEVER commit or push without explicit confirmation** (global rule; restated because it matters).

## Repo Structure
```
configs/     one YAML per experiment
src/data/    manifest building, dataset class, mixing
src/models/  MSST wrappers (later: custom models)
scripts/     data acquisition / extraction one-offs
notebooks/   EDA only
experiments/ per-experiment LIGHT artifacts (metrics parquet+csv + notes) — committed;
             figures gitignored (regenerable). Convention → experiments/README.md
manifests/   frozen manifest (songs/stems parquet+csv) — committed
docs/        status page & write-ups
data/        -> symlink to ~/storage/nia-gugak (dataset + heavy eval audio; not tracked)
external/    ZFTurbo MSST repo (submodule/vendored)
```

## Modeling tooling
- **MSST-first:** wrap ZFTurbo/Music-Source-Separation-Training. Config-driven HTDemucs,
  BS-RoFormer, Mel-Band RoFormer, SCNet; **fine-tune from pretrained** (not from scratch —
  ~300 h multitrack is fine-tuning territory). Don't reimplement what MSST provides.
- **Custom later:** `src/models/` for gugak-specific ideas + ablations.
- Experiment order, stem-class scheme, and model roadmap are live decisions → **Notion**.

## Evaluation
- Training metric: SI-SDR (fast). Reported metric: museval-style chunked SDR (literature-comparable).
- Always break results down **per-stem AND per-genre**, never global mean only.
- Expect lower SDR than Western benchmarks (heterophony → high source overlap). Quantify, don't panic.

## Logging (wandb)
- Curves: train/val loss, LR, grad norm. Per-stem SDR/SI-SDR on a fixed val subset every N steps.
- **Audio:** `wandb.Audio` triplets (mixture, ground-truth, predicted) for 3–5 fixed val songs each eval.
- Lab share-outs go in wandb Reports.
