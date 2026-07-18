# CLAUDE.md — Gugak Stem Separation

## Project
Stem separation on a traditional Korean music (gugak) multitrack dataset
(AI Hub **국악합주곡 디지털 음원 데이터**, datasetkey 71955). First task in the MALer lab
archiving/cataloguing workstream; also a full experiment-cycle practice run:
data → baseline → fine-tune → compare → share.

## Status (2026-07-18)
- **Data:** downloaded → extracted → **manifested**. Working set = **903 songs / 6,670 tracks**.
  Full status: `docs/dataset_status.md`.
- **Env:** `uv` project, Python 3.11. PyTorch not installed yet (added at modeling).
- **Now:** EDA Phase 2 (master ≈ Σ(stems) residual test). No experiments run yet.

## Roadmap (tick off)
**EDA**
- [x] Phase 0 — Repo scaffold + uv env
- [x] Phase 1 — Manifest: own stratified split + header-level audio props
- [ ] Phase 2 — Audio-content EDA: master ≈ Σ(stems) residual (all genres incl. 판소리);
      silence/clipping/loudness; per-genre & per-instrument stats; confirm 96 kHz resample
      + 판소리 length-align plan
- [ ] Phase 3 — Finalize 24→4 stem mapping; decide 판소리 handling; freeze → EDA done
- [ ] Phase 4 — Produce deliverables (graphs, charts, tables with numbers) for the instructor meeting.

**Modeling**
- [ ] Zero-shot pretrained baselines (HTDemucs + BS-RoFormer) on a val subset → wandb
- [ ] 4-stem grouping + stem-summing dataloader
- [ ] exp001: HTDemucs fine-tune (validates pipeline)
- [ ] exp002: BS-RoFormer fine-tune → compare → first wandb Report
- [ ] Produce deliverables (graphs, charts, tables, key insights + plans) briefing progress for the instructor meeting.

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
- **master ≈ Σ(stems)?** Build training mixtures by **summing stems** — but this linear-mixing
  assumption is **not yet verified**; run the residual test (all genres) first. Master = real-world test reference only.
- **판소리:** 28 songs have master slightly longer than stems (~0.05–0.5 s) → align/trim before summing.
- **Multi-instrument stems** exist (피리1/피리2/피리3…) → strip trailing digit to get the base instrument.
- **Korean filenames need NFC** normalization for any name join. **Master naming translation:**
  on disk `<song>_master.wav` vs metadata records `<song>.wav`.

## Repo Conventions
- One experiment = one config (`configs/*.yaml`) = one wandb run. Change configs, not code, for hyperparameters.
- Run naming: `expNNN_<model>_<stemscheme>_<key-hparam>` (e.g. `exp003_bsroformer_4stem_lr1e-5`).
- Log git commit hash + full config to wandb for every run.
- Manifests, not directory walking. Split frozen in the manifest; commit it. Never let chunks of one
  song cross splits (audio leakage).
- Notebooks (`notebooks/`) = EDA only. Seeds set (torch/numpy/random). Paths in config/.env, never hardcoded.

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
- **4-stem (exp1):** 타악 / 관악 / 찰현 / 발현 (voice absent → no 5th stem). Provisional 24→4 mapping is
  in the manifest (`stem_group_4`); finalize in EDA Phase 3.
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
- Confirm the current experiment's config before running training.
- Don't hardcode paths, bypass the manifest, or edit `uv.lock` by hand.
- Prefer editing configs over editing code for experiment variation.
- Ask before large downloads or long-running training launches.
