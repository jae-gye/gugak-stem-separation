# CLAUDE.md — Gugak Stem Separation
<!-- STABLE project truth only. The living plan (roadmap, current stage, decisions, evolving
     strategy) lives in Notion — see "Plan & tracking" below; do NOT re-add a roadmap here.
     Personal working style, sym8/compute etiquette, uv rules, and general guardrails live in
     ~/.claude/CLAUDE.md. Commit format + tags live in the git-commit skill. -->

## Plan & tracking — Notion is the single source of truth
The **roadmap, current stage, decision log, and evolving strategy** live **in Notion**.
This file is intentionally roadmap-free so it can't drift out of sync.

- **Fetch at session start, and before any planning/status discussion:** Notion page
  **Gugak Stem Separation** — id `3a4cb377-8743-8136-8f9a-c2bc24ef8269`
  (https://app.notion.com/p/3a4cb377874381368f9ac2bc24ef8269).
  Child pages: **Experiments** `3a4cb377-8743-8012-921b-e6174eb0effb` ·
  **Survey** `3a4cb377-8743-8075-932e-ca518395c3e1`.
- **Notion gets the git treatment — it's the project brain, keep me on top of edits to it:** when a
  plan changes, an item completes, or a decision is made, **flag in the same turn that Notion needs
  updating and propose the edit — but do NOT write to Notion until I confirm.** Confirming a Notion
  edit is its own go-ahead, separate from any git confirmation.
- **On any conflict, Notion wins** — it is the newer, authoritative plan. This file may lag on
  anything plan-shaped; trust Notion for that, trust this file for stable facts/conventions.
- **Single-copy rule:** every fact has exactly one home. Anything plan-shaped, or already stated in
  Notion, gets a pointer here — not a copy.

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
- **Audio I/O:** use `soundfile` (handles float wavs; stdlib `wave` chokes on them).
  **Avoid `librosa`** — lab convention.

## Datasets  (full status → `docs/dataset_status.md`; instrument taxonomy →
`docs/stem_taxonomy.md` + canonical mapping `configs/stem_taxonomy.yaml`; counts/tables → Notion)
- refer to each dataset as either ensemble dataset or solo dataset rather than their idx.
- **71955 (ensemble, ours-for-everything):** **903 songs** with audio (published 1,004 minus the 101
  withheld-test songs); 1 master + N stems each; **6,670 WAVs**. 48 kHz / 24-bit / stereo,
  **except 130 files at 96 kHz** (창작국악 → resampled).
- **71470 (solo phrases, train-only pool):** ~9,945 usable single-instrument clips + per-clip MIDI
  (악보) and annotations. **Format-heterogeneous** — 15 (sr, bit, ch) combos incl. float wavs,
  clips ~1–78 s → resampled + unified channel/bit on ingest.
- **On disk:** `data/` is a real directory holding one symlink per set:
  - `data/gugak_ensemble_71955/` → `~/storage/nia-gugak` — `source/<song>/` =
    `<song>_master.wav` + `<song>_<instrument>.wav`; tagging JSONs in `labels/`.
  - `data/gugak_solo_71470/` → `~/storage/ngc-gugak` — flat `audio/` · `midi/` · `labels/`.
- **Manifests = source of truth** (never walk directories). All live in `manifests/`:
  - `eval_manifest` — frozen 71955 song-level split (`src/data/data_splitter.py`).
  - `ingest_manifest` — one row per ingested file: provenance + ops applied (`src/data/ingest.py`).
  - `audio_qc*` — QC scans: `audio_qc` = raw sources, `audio_qc_ingest_<set>` = processed store
    (`scripts/audio_qc.py`). `audio_qc.parquet` is the channel-decision table ingest consumes.
  - **`source_manifest` = the one dataloaders read** — ingest ⋈ QC ⋈ taxonomy, one row per
    ingested source file, keyed by stable `file_id` (`src/data/build_source_manifest.py`).
    The pitch-shift pool will be a SEPARATE table at (source × semitone) grain, foreign-keying
    here — it must not re-copy split/instrument/content columns.
- **Split principle** (publisher's split CSVs/folders are inconsistent & incomplete — ignored):
  own song-level split, stratified by genreSub, seed-pinned, no song crossing splits. Test + val
  are frozen 71955 song lists; the training side is **generated augmented mixes, not a song-count
  split**. Concrete ratios/counts and the augmentation design → Notion.

### Key facts & gotchas
- **Vocals:** 71955 has **no vocal/소리 stem** anywhere (incl. 판소리 — its masters also appear
  voiceless). 71470 **does** supply clean solo vocal clips → whether the scheme gains a voice class
  is a live decision → Notion. Do not assume "no voice stem" as settled.
- **master ≠ Σ(stems) (verified, Phase 2).** The publisher master is mastered (per-stem faders +
  reverb/FX) and genre-dependent (민요 ≈ clean sum, 창작국악 heavily processed).
  → **never invert master → stems**; the stem targets do not sum to it.
  **Training mixtures** — they are generated augmented mixes → Notion
  (Training Data Strategy).
- **Two variants per eval song:** publisher master + Σstem mix. **Early stopping and monitoring run
  on the Σstem variant only** — master-val carries an irreducible error floor (the mastering
  residual), which muddies curves and early-stop signals. Master-val is a real-world reference,
  never model selection. Rationale → Notion.
- **Audio QC — raw sources scanned 2026-07-25, processed store verified 2026-07-27**
  (`scripts/audio_qc.py` → `manifests/audio_qc*.parquet`, both sets; findings → Notion).
  Ingest verification: 16,615/16,615 files at 44.1k/PCM_24, 0 peaks >1.0, 0 dead, 0 anti-phase or
  dual-mono survivors, duration conserved 465.38 h. Standing rules from the raw scan:
  - **Sparse ≠ dead** — do NOT auto-drop low-activity stems (박 etc. play rarely by design).
  - Normalize the mixture, never per-stem (loudness ~−19 LUFS in 71955).
  - 71470 ingest: clamp/normalize peaks >1.0, DC-remove, and do **not** naive-average L/R to mono
    (anti-phase clips cancel) — take one channel or a phase-aware downmix.
- **판소리 length-align (verified, Phase 2):** stems are start-aligned; in 28 판소리 songs one stem
  (almost always **가야금**) runs *longer* than the master (usually <0.5 s, one 11 s tail) with real
  (non-silent) content the master lacks. Rule: **trim every stem to the shortest per song before
  summing** (don't pad — padding injects content the master never had). 판소리-only.
- **Multi-instrument stems** exist (피리1/피리2/피리3…) → strip trailing digit to base, **sum same-base
  into one source**.
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
- Manifests, not directory walking. Split frozen in the manifest; commit the **parquet** (source of
  truth). Its **CSV twin is not auto-committed — surface the file size and let me decide** per case
  (small diff-able → commit both; large → parquet-only + gitignore the csv). Never let chunks of one
  song cross splits (audio leakage).
- Notebooks (`notebooks/`) = EDA only.

## Commits
- Format + tag taxonomy → **`git-commit` skill.** No repo-specific tag overrides currently.
- **NEVER commit or push without explicit confirmation** (global rule; restated because it matters).

## Modeling tooling
- **MSST-first:** wrap ZFTurbo/Music-Source-Separation-Training. Config-driven HTDemucs,
  BS-RoFormer, Mel-Band RoFormer, SCNet; **fine-tune from pretrained** (not from scratch —
  ~370 h ensemble stems + ~40 h solo clips is fine-tuning territory). Don't reimplement what MSST
  provides.
- **Custom later:** `src/models/` for gugak-specific ideas + ablations.
- Experiment order, stem-class scheme, and model roadmap are live decisions → **Notion**.

## Evaluation & logging
Full protocol + wandb design → **Notion (Evaluation & Logging)**. Always-on rules:
- Training metric: SI-SDR (fast). Reported metric: museval-style chunked SDR
  (literature-comparable).
- Always break results down **per-stem AND per-genre**, never global mean only.
- Val monitoring/early stopping on the **Σstem variant** (see gotchas above).
- Expect lower SDR than Western benchmarks (heterophony → high source overlap). Quantify, don't panic.