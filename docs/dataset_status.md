# Dataset status — gugak stem separation

**Last updated:** 2026-07-27 · Counts/tables → Notion · Instrument taxonomy → [`stem_taxonomy.md`](stem_taxonomy.md)

## Pipeline state

| Stage | 71955 (ensemble) | 71470 (solo) |
|---|---|---|
| Downloaded + extracted | ✅ | ✅ |
| Full audio QC (raw) | ✅ 2026-07-25 | ✅ 2026-07-25 |
| Split frozen | ✅ 2026-07-25 | n/a — train-only pool |
| Offline ingest → 44.1 kHz / PCM_24 | ✅ 2026-07-25 | ✅ 2026-07-25 |
| Ingest store verified | ✅ 2026-07-27 | ✅ 2026-07-27 |
| Instrument taxonomy assigned | ✅ 2026-07-27 | ⚠️ 222 of 9,945 clips held aside |
| Pitch-shift pool | ⬜ not started | ⬜ not started |
| Augmentation / mixing pipeline | ⬜ not started | ⬜ not started |

## The two datasets

| | **71955** ensemble | **71470** solo |
|---|---|---|
| AI Hub name | 국악합주곡 디지털 음원 데이터 | 국악 악보 및 음원 데이터 |
| Role | everything — train pool + the only eval source | train-only solo-source pool |
| Units | **903 songs** → 903 masters + 5,767 stems = **6,670 WAVs** | **9,945 clips** (+ per-clip MIDI 악보 + annotations) |
| Audio | ~57 h masters · ~369 h stems | ~39.8 h, clips 1.2–77.5 s (median 12) |
| Raw format | 48 kHz / 24-bit / stereo (130 files at 96 kHz) | heterogeneous — 15 (sr, bit, ch) combos incl. 719 float |
| On disk | `data/gugak_ensemble_71955/` → `~/storage/nia-gugak` | `data/gugak_solo_71470/` → `~/storage/ngc-gugak` |

**Availability:** the published set lists 1,004 songs; the publisher withholds audio for its
101-song test split (metadata only). Available audio = **903 songs**, all complete (master +
every stem + metadata). Our dataset is branched from exactly these.

## Frozen split (71955, seed 42, genre-stratified, song-level)

**train 677 · val 91 · test 135** — no song crosses splits. Per-genre breakdown → Notion.

- Test + val exist in **two variants** per song: publisher master · our Σstem mix.
- Early stopping and monitoring run on the **Σstem variant only** (master carries an
  irreducible mastering residual). Master-val is a real-world reference, never selection.
- The train column is a *song pool* feeding generated augmented mixes, not a song-count split.
- 71470 clips join **train only**.

## Ingest store

Deterministic per-file preprocessing, run once offline: resample (soxr VHQ) → DC removal →
peak clamp → channel rule → PCM_24. Recipe + rationale → Notion.

- Canonical format **44.1 kHz / PCM_24** (matches HTDemucs / BS-RoFormer pretrain checkpoints)
- **397 GB** at `<dataset>/ingest/` on NVMe · **16,615 rows, 0 errors**
- Verified 2026-07-27: 16,615/16,615 files at 44.1k/PCM_24 · 0 peaks >1.0 · 0 dead · 0 clipping
  · |DC| ≤ 0.00002 · 0 anti-phase or dual-mono survivors · duration conserved 465.38 h ·
  0 truncated files

## Manifests (`manifests/` — source of truth, never walk directories)

| File | Grain | Built by |
|---|---|---|
| `eval_manifest` | one row per 71955 song | `src/data/data_splitter.py` |
| `ingest_manifest` | one row per ingested file — provenance + ops applied | `src/data/ingest.py` |
| `audio_qc` · `audio_qc_ingest_<set>` | one row per file — QC metrics (raw / processed) | `scripts/audio_qc.py` |
| **`source_manifest`** | **one row per ingested source file — the one dataloaders read** | `src/data/build_source_manifest.py` |

`source_manifest` = ingest ⋈ QC ⋈ taxonomy, keyed by stable `file_id`. The pitch-shift pool
will be a **separate** table at (source × semitone) grain, foreign-keying into it.

Parquet is committed; CSV twins are disk-only (gitignored, not diff-able at 16k rows).

## Key gotchas

- **master ≠ Σ(stems)** — masters are mastered (faders + reverb/FX), genre-dependent.
  Never invert master → stems.
- **No vocal stem in 71955**, anywhere, including 판소리. All voice material comes from 71470.
- **Sparse ≠ dead** — do not auto-drop low-activity stems (박 plays ~twice per song by design).
- **판소리 length-align** — 28 songs have one stem (usually 가야금) running longer than the
  master with real content. Trim every stem to the shortest per song before summing; never pad.
- **Multi-instrument stems** (피리1/2/3) → trailing digit stripped, same-base summed into one source.
- **Korean filenames need NFC** for any name join. On disk `<song>_master.wav`; metadata says `<song>.wav`.
- Normalize the mixture, never per-stem (71955 loudness ~−19 LUFS).

## Publisher split files — resolved 2026-07-21

The publisher ships three overlapping, mutually-inconsistent indexes (`genre_split.csv`,
`instrument_split.csv`, and `Training`/`Validation` folders) that disagree on membership and
each omit songs. **They were never a designed split** — the two CSVs are artifacts generated
at training time by the publisher's own validation scripts (`train.py`, `train_instrument.py`),
each doing an independent 9:1 random hold-out for a different task.

→ No canonical split ever existed to inherit, which retroactively confirms branching our own.

## No prior separation baseline

The dataset's official validation is **tagging models only** — two ResNet-18 classifiers
(genre tagging on masters, instrument tagging on stems), ACC-70% target, sparse documentation.

→ **No source-separation model or baseline exists for this dataset.** This project establishes
the first separation results and eval protocol on it. Prior-art context → Notion · Survey · Strand 0.
