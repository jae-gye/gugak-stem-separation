# CLAUDE.md — Gugak Stem Separation

## Project
Stem separation on a traditional Korean music (gugak) multitrack dataset
(AI Hub **국악합주곡 디지털 음원 데이터**, datasetkey 71955). First task in the MALer lab
archiving/cataloguing workstream; also a full experiment-cycle practice run:
data → baseline → fine-tune → compare → share.

## Status (2026-07-19)
- **Data:** downloaded → extracted → **manifested**. Working set = **903 songs / 6,670 tracks**.
  Full status: `docs/dataset_status.md`.
- **Env:** `uv` project, Python 3.11. PyTorch not installed yet (added at modeling step 0.1).
- **Now:** EDA **done & frozen** (Phases 0–3; Phase 4 deliverable assembly remains). Meeting figures
  ready: `notebooks/fig_master_vs_sum.png`, `fig_dataset_overview.png`. **Modeling: 0.1–0.2 done
  (torch 2.11.0+cu128; MSST submodule; inference deps in). Next: 0.3 pick pinned listening set.** No experiments run yet.
- **GPU (confirmed 2026-07-19):** 3× RTX PRO 6000 Blackwell Server Edition, 96 GB each, driver
  590.48 (CUDA 13.1). Blackwell = sm_120 → PyTorch **cu128** build (pinned in `pyproject.toml`).

## Roadmap (tick off)
**EDA**
- [x] Phase 0 — Repo scaffold + uv env
- [x] Phase 1 — Manifest: own stratified split + header-level audio props
- [x] Phase 2 — Audio-content EDA: master ≠ Σ(stems) verified (details in gotchas); stem QC clean
      (sample only); 96 kHz confirmed (130 files); 판소리 length-align (trim-to-shortest);
      per-genre/instrument stats + overview dashboard
- [x] Phase 3 — **Frozen 2026-07-19.** 4-class scheme ratified by user (revisitable — see Stem
      Schemes); 양금→발현; 판소리 included normally (trim-to-shortest). → EDA done
- [~] Phase 4 — Deliverables: `fig_master_vs_sum.png` + `fig_dataset_overview.png` produced
      (local, gitignored); assemble/polish for the instructor meeting

**Modeling**
- [ ] (0) Zero-shot pretrained baselines → wandb. **Models: single `htdemucs` + viperx
      `bs_roformer`, both via MSST** (NOT htdemucs_ft — its Western fine-tune edge doesn't
      transfer to out-of-domain gugak, 4× slower, awkward bag-of-4 in MSST; single also aligns
      with exp001 which fine-tunes the single model).
      Framing: **pipeline validation + qualitative floor** — pretrained heads are Western
      (vocals/drums/bass/other ≠ our stems; only 타악↔drums loosely maps). Granular plan:
      - [x] 0.1 torch 2.11.0+cu128 + torchaudio installed; 3× Blackwell sm_120 verified (matmul on GPU OK)
      - [x] 0.2 MSST added as submodule `external/msst` (fork jae-gye/…, upstream=ZFTurbo, pinned 83d495d)
      - [x] 0.2b MSST inference deps added (librosa/omegaconf/ml_collections/einops/openunmix/demucs 4.1.0
            + beartype/rotary_embedding_torch/hyper_connections pinned); both models import, torch/numpy intact
      - [x] 0.3 **Pinned listening set** (`scripts/make_listening_set.py` → `data/listening_set/`,
            logged as `wandb.Audio` each eval across ALL experiments): 판소리 0631 · 창작국악 0854 ·
            풍류음악 0006 · 민요 0721 · 산조 0196. Doubles as smoke-test input in 0.5.
      - [ ] 0.4 Build mixtures = Σstems, trim-to-shortest (reuse `src/data/audio.read_and_sum`)
      - [ ] 0.5 single htdemucs (MSST) zero-shot inference on the smoke set (plumbing check)
      - [ ] 0.6 Metric pass on **FULL val (91 songs)**: SI-SDR/SDR, per-stem AND per-genre
      - [ ] 0.7 wandb run: git hash + config + `wandb.Audio` triplets for the fixed songs
      - [ ] 0.8 Repeat for viperx BS-RoFormer (MSST) → stage (0) done
- [ ] (1) 4-stem grouping + stem-summing dataloader (+ build `data/splits/{train,val,test}/`
      symlink view from the manifest for MSST's dir-based training — do NOT physically move
      `extracted/`; our split is logical, symlink-view only)
- [ ] (2) exp001: HTDemucs fine-tune 4-stem (validates training pipeline)
- [ ] (3) exp002: BS-RoFormer fine-tune → compare → first wandb Report
- [ ] (4) Produce deliverables (graphs, charts, tables, key insights + plans) briefing progress for the instructor meeting.

## Environment
- **Package manager: uv** (not conda). `uv add`, `uv run`, `uv sync`. Deps in `pyproject.toml`;
  commit `uv.lock`. Never `pip install`; never hand-edit deps or the lockfile.
- `ffmpeg` = system dependency (provided by server), not a Python dep.
- PyTorch: install the CUDA build via the PyTorch index (uv) when we reach modeling.

## Compute
- Server sym8. 3× RTX Pro 6000 Blackwell (96 GB each). Default **1 GPU** unless scaling up.
- Long jobs in `tmux`, never a bare SSH session. Data on server-local NVMe (`~/storage`), not NFS home.

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
- **No vocal/소리 stem** anywhere (incl. 판소리 — its masters also appear voiceless) → **4-stem scheme,
  no 5th voice stem.**
- **master ≠ Σ(stems) (verified, Phase 2).** Build training mixtures by **summing stems**
  (mix:=Σstems, targets:=stems — self-consistent). The real master is NOT the sum: mastered
  (per-stem faders + reverb/FX), genre-dependent (민요 ≈ clean sum, 창작국악 heavily processed).
  → master = **secondary real-world eval only**; do NOT invert master→stems.
- **Stem QC (Phase 2 sample) = clean:** 0 dead stems, clipping negligible (drop the check),
  loudness ~−19 LUFS. **Sparse ≠ dead** — do NOT auto-drop low-activity stems (박 etc. play
  rarely by design). Normalize the mixture, never per-stem. No full scan needed.
- **판소리 length-align (verified, Phase 2):** stems are start-aligned; in 28 판소리 songs one stem
  (almost always **가야금**) runs *longer* than the master (usually <0.5 s, one 11 s tail) with real
  (non-silent) content the master lacks. Rule: **trim every stem to the shortest per song before
  summing** (don't pad — padding injects content the master never had). 판소리-only.
- **Multi-instrument stems** exist (피리1/피리2/피리3…) → strip trailing digit to base, **sum same-base
  into one source**. (향/세/당피리 subtype ambiguity — experts pending — affects exp2 only, not 4-stem.)
- **Korean filenames need NFC** normalization for any name join. **Master naming translation:**
  on disk `<song>_master.wav` vs metadata records `<song>.wav`.

## Repo Conventions
- **Experiment configs:** one per experiment (`configs/*.yaml`) = one wandb run; change configs,
  not code, for hyperparameters. **Shared/cross-experiment configs** (e.g. `configs/listening_set.yaml`)
  also live in `configs/`. If it grows, split `configs/experiments/` vs `configs/shared/`.
- Run naming: `expNNN_<model>_<stemscheme>_<key-hparam>` (e.g. `exp003_bsroformer_4stem_lr1e-5`).
- Log git commit hash + full config to wandb for every run.
- Manifests, not directory walking. Split frozen in the manifest; commit it. Never let chunks of one
  song cross splits (audio leakage).
- Notebooks (`notebooks/`) = EDA only. Seeds set (torch/numpy/random). Paths in config/.env, never hardcoded.

## Commit Guidelines
- **Format:** `[tag] imperative summary`. Optional body = terse bullets, one line per file/area
  (no prose sentences). **Header-only** for small/single-purpose commits; add bullets only when
  there's genuinely more than one notable change. **No `Co-Authored-By` trailer.** Tag by primary intent.
- **NEVER commit or push without explicit confirmation** (see Guardrails).
- **Tags:** `init` (scaffolding) · `data` (dataset/manifests/preprocessing/dataloaders) ·
  `eda` (analysis/notebooks/figures) · `model` (model code & MSST wrappers we write) ·
  `exp` (per-experiment run configs + training runs) ·
  `config` (shared/infra configs & config plumbing, e.g. listening_set — NOT per-experiment run configs) ·
  `eval` (metrics/results/wandb reports) ·
  `deps` (dependencies, lockfile, submodules, toolchain/env — pyproject, uv.lock, `external/*`) ·
  `fix` (bug fixes) · `docs` (README, `docs/`, CLAUDE.md, status) ·
  `chore` (pure housekeeping: gitignore, file moves, formatting, CI) ·
  `refactor` (restructure, no behavior change)

## Repo Structure
```
configs/     one YAML per experiment
src/data/    manifest building, dataset class, mixing
src/models/  MSST wrappers (later: custom models)
scripts/     data acquisition / extraction one-offs
notebooks/   EDA only
manifests/   frozen manifest (songs/stems parquet+csv) — committed
docs/        status page & write-ups
data/        -> symlink to ~/storage/nia-gugak (dataset; not tracked)
external/    ZFTurbo MSST repo (submodule/vendored)
```

## Stem Schemes
- **4-stem (exp1) — ratified 2026-07-19, deliberately revisitable:** 타악 / 관악 / 찰현 / 발현
  (voice absent → no 5th stem). Mapping frozen in the manifest (`stem_group_4`); 양금→발현
  (struck string, but pitched-decay timbre — user signed off by ear).
  ⚠️ **This grouping is OUR design, not the dataset's** (the dataset ships 24 sub-classes / 9 majors,
  no 4-family scheme). Chosen to match 4-headed pretrained models + balanced hours (114/92/91/72).
  It is a **crucial, revisitable decision**: we may later adopt the 9 majors, or re-group entirely
  (e.g. for granular per-instrument work). Revisit deliberately, not by accident.
- **2-stem target-vs-rest (exp2):** per big instrument; hard pairs 대금↔피리, 해금↔아쟁, 가야금↔거문고.

## Modeling Plan
- **MSST-first (now):** wrap ZFTurbo/Music-Source-Separation-Training. Config-driven HTDemucs,
  BS-RoFormer, Mel-Band RoFormer, SCNet; fine-tune from pretrained. Don't reimplement what MSST provides.
- **Custom later:** `src/models/` for gugak-specific ideas + ablations.
- Order: (0) zero-shot baselines → (1) HTDemucs fine-tune 4-stem → (2) BS-RoFormer 4-stem → compare.

## Evaluation
- Training metric: SI-SDR (fast). Reported metric: museval-style chunked SDR (literature-comparable).
- Always break results down **per-stem AND per-genre**, never global mean only.
- Expect lower SDR than Western benchmarks (heterophony → high source overlap). Quantify, don't panic.

## Logging (wandb)
- Curves: train/val loss, LR, grad norm. Per-stem SDR/SI-SDR on a fixed val subset every N steps.
- **Audio:** `wandb.Audio` triplets (mixture, ground-truth, predicted) for 3–5 fixed val songs each eval.
- Lab share-outs go in wandb Reports.

## Guardrails for Claude Code
- **NEVER, EVER commit without discussion first. ALWAYS get an EXPLICIT confirmation before commit or push - VERY IMPORTANT!**
- CLAUDE.md changes need no permission, but **ALWAYS report them explicitly in the reply**
  ("added X to CLAUDE.md", "ticked Y in Roadmap") so the user can track the evolving plan/context.
- **CLAUDE.md is the living planner (user preference — IMPORTANT).** The Roadmap here must never
  diverge from what we're actually doing. In the SAME turn that (a) a plan is agreed in conversation,
  (b) an item completes, or (c) the plan changes — update the Roadmap/Status here (and report it,
  per the bullet above). Keep granular sub-plans only for the current milestone; prune once done.
- **Work granularly (personal preference — IMPORTANT).** One task/step per turn, then STOP and
  check in before the next. Prefer many small decisions the user makes over batching work
  autonomously. This project is **as much about the user's education as productivity** — explain
  the *why*, surface choices, ask questions, and let the user steer. Do NOT chain multiple tasks
  in a single turn, even if they seem related.
- Confirm the current experiment's config before running training.
- Don't hardcode paths, bypass the manifest, or edit `uv.lock` by hand.
- Prefer editing configs over editing code for experiment variation.
- Ask before large downloads or long-running training launches.
