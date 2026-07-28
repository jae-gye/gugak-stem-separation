"""build_activity_manifest.py — STAGE 2 of the stem activity scan: envelopes -> tables.

Reads the per-block loudness curves stage 1 saved (src/data/activity_scan.py) and turns
them into the three decision tables the augmentation pipeline reads. Every judgment call
the scanner deliberately avoided lives HERE — what counts as playing, what counts as a
rest, what a training excerpt actually hears. Touches no audio, so a full run takes
seconds and re-running after a threshold change is free.

Outputs (parquet source-of-truth under manifests/parquet/, csv twins under manifests/csv/):
  activity_segments   one row per continuous playing stretch of every file
  activity_summary    per stem_group x genre x split: hours, active fraction,
                      stretch/rest length percentiles (the pitch-shift-skip evidence)
  chunk_activities    per 71955 song x window length x window: what fraction of the
                      window each of the 11 stem classes is audible (the density-knob
                      calibration table; ~140k rows)

Segment extraction — four cleanup rules, in this order (all values from the yaml):
  1. thermostat thresholds  open at threshold_open_db, close below threshold_close_db —
                            one threshold flickers on decaying notes, two don't
  2. bridge short gaps      a rest < min_gap_s between stretches is a breath, not a rest
  3. drop blips             a stretch < min_segment_s is a bump/click, not playing
  4. extend tails           pad each stretch end by release_margin_s so the decay a level
                            threshold always clips stays inside the segment

Built-in check: per file, playing time from segments can only exceed the raw
above-threshold time (bridging + tails add, dropping blips is the sole subtraction) —
a file where segments UNDERSHOOT raw beyond the dropped-blip allowance is a logic bug
and aborts the run. Overshoot is reported as a distribution, not policed.

Run:
    uv run python src/data/build_activity_manifest.py --config configs/activity_scan.yaml
      --dry-run        build + report, write nothing
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# --- repo-root discovery (cwd-agnostic, mirrors activity_scan.py) ---
def find_root(start: Path | None = None) -> Path:
    """Locate the repo root (holds pyproject.toml)."""
    p = Path.cwd() if start is None else start
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists():
            return cand
    raise FileNotFoundError("repo root (pyproject.toml) not found above cwd")


def table_paths(base: Path) -> tuple[Path, Path]:
    """Map a manifests/<name> basename to its (parquet, csv) twin paths.

    Layout rule (repo-wide): parquet = source of truth in manifests/parquet/,
    csv = eyeball copy in manifests/csv/. Both dirs are created on demand.

    Args:
        base: table basename, e.g. Path(".../manifests/activity_segments").
    """
    parquet = base.parent / "parquet" / f"{base.name}.parquet"
    csv = base.parent / "csv" / f"{base.name}.csv"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    return parquet, csv


# --- config -----------------------------------------------------------------
@dataclass
class ActivityManifestConfig:
    """Stage-2 view of configs/activity_scan.yaml (paths absolute against root)."""
    threshold_open_db: float
    threshold_close_db: float
    min_gap_s: float
    min_segment_s: float
    release_margin_s: float
    chunk_lengths_s: list
    coverage_thresholds: list
    source_manifest: Path
    envelope_dir: Path
    out_index: Path
    out_segments: Path
    out_summary: Path
    out_chunks: Path

    @classmethod
    def load(cls, path: Path, root: Path) -> "ActivityManifestConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            threshold_open_db=float(raw["threshold_open_db"]),
            threshold_close_db=float(raw["threshold_close_db"]),
            min_gap_s=float(raw["min_gap_s"]),
            min_segment_s=float(raw["min_segment_s"]),
            release_margin_s=float(raw["release_margin_s"]),
            chunk_lengths_s=[float(x) for x in raw["chunk_lengths_s"]],
            coverage_thresholds=[float(x) for x in raw["coverage_thresholds"]],
            source_manifest=root / raw["source_manifest"],
            envelope_dir=root / raw["envelope_dir"],
            out_index=root / raw["out_index"],
            out_segments=root / raw["out_segments"],
            out_summary=root / raw["out_summary"],
            out_chunks=root / raw["out_chunks"],
        )


# --- segment extraction (the heart) ------------------------------------------

def runs_of_true(mask: np.ndarray) -> np.ndarray:
    """Maximal runs of True in a boolean array, as an (n, 2) array of [start, end).

    Args:
        mask: 1-D boolean array.
    """
    edges = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False]))))
    return edges.reshape(-1, 2)


def segments_from_envelope(env_db: np.ndarray, block_s: float,
                           cfg: ActivityManifestConfig) -> np.ndarray:
    """Loudness curve in -> playing stretches out, as (n, 2) block intervals [start, end).

    Applies the four cleanup rules documented in the module docstring. Hysteresis is
    computed without a sequential state machine: an active region is exactly a maximal
    run of above-close blocks that contains at least one above-open block (opens on the
    open threshold, then survives while above close — same result, vectorised).

    Args:
        env_db: per-block dBFS envelope (float32).
        block_s: block duration in seconds (from the activity index).
        cfg: stage-2 thresholds and shaping values.
    """
    if env_db.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    # rule 1 — thermostat thresholds (hysteresis)
    above_close = env_db > cfg.threshold_close_db
    above_open = env_db >= cfg.threshold_open_db
    runs = runs_of_true(above_close)
    opened = [r for r in runs if above_open[r[0]:r[1]].any()]
    if not opened:
        return np.empty((0, 2), dtype=np.int64)
    segs = np.array(opened, dtype=np.int64)

    # rule 2 — bridge gaps shorter than min_gap_s
    max_gap_blocks = int(round(cfg.min_gap_s / block_s))
    merged = [segs[0].copy()]
    for start, end in segs[1:]:
        if start - merged[-1][1] < max_gap_blocks:
            merged[-1][1] = end
        else:
            merged.append(np.array([start, end]))
    segs = np.array(merged, dtype=np.int64)

    # rule 3 — drop stretches shorter than min_segment_s
    min_seg_blocks = max(1, int(round(cfg.min_segment_s / block_s)))
    segs = segs[(segs[:, 1] - segs[:, 0]) >= min_seg_blocks]
    if len(segs) == 0:
        return segs

    # rule 4 — extend each end by release_margin_s, then re-merge any overlaps created
    release_blocks = int(round(cfg.release_margin_s / block_s))
    segs[:, 1] = np.minimum(segs[:, 1] + release_blocks, len(env_db))
    remerged = [segs[0].copy()]
    for start, end in segs[1:]:
        if start <= remerged[-1][1]:
            remerged[-1][1] = max(remerged[-1][1], end)
        else:
            remerged.append(np.array([start, end]))
    return np.array(remerged, dtype=np.int64)


# --- table builders -----------------------------------------------------------

def build_segments(index: pd.DataFrame, blob: np.ndarray,
                   cfg: ActivityManifestConfig) -> tuple[pd.DataFrame, dict]:
    """Table 1: run segment extraction over every file.

    Returns (segments dataframe, {file_id: (n, 2) block intervals}). The interval dict
    is kept in block units for the chunk pass, which rasterises them back to blocks.
    """
    rows: list[dict] = []
    intervals: dict[str, np.ndarray] = {}
    for r in index.itertuples(index=False):
        env = np.asarray(blob[r.offset:r.offset + r.n_blocks])
        segs = segments_from_envelope(env, r.block_s, cfg)
        intervals[r.file_id] = segs
        for k, (b0, b1) in enumerate(segs):
            chunk = env[b0:b1]
            rows.append(dict(
                file_id=r.file_id, seg_idx=k,
                start_s=round(b0 * r.block_s, 3), end_s=round(b1 * r.block_s, 3),
                dur_s=round((b1 - b0) * r.block_s, 3),
                mean_dbfs=round(float(chunk.mean()), 2),
                peak_dbfs=round(float(chunk.max()), 2)))
    return pd.DataFrame(rows), intervals


def check_segments(index: pd.DataFrame, blob: np.ndarray, intervals: dict,
                   cfg: ActivityManifestConfig) -> None:
    """Abort if any file's segments UNDERSHOOT its raw above-threshold time.

    Segments are built from raw active blocks by adding (bridged gaps, tails) and
    subtracting only dropped blips — so per file:
        segment_blocks >= raw_blocks - dropped_blip_allowance
    must hold. Violation = logic bug, not a tuning issue. Overshoot (bridging + tails)
    is legitimate and only reported.
    """
    overshoot: list[float] = []
    for r in index.itertuples(index=False):
        env = np.asarray(blob[r.offset:r.offset + r.n_blocks])
        if r.n_blocks == 0:
            continue
        raw = int((env >= cfg.threshold_open_db).sum())
        seg = int((intervals[r.file_id][:, 1] - intervals[r.file_id][:, 0]).sum())
        min_seg_blocks = max(1, int(round(cfg.min_segment_s / r.block_s)))
        if seg < raw - _dropped_blip_allowance(env, cfg, min_seg_blocks, r.block_s):
            raise AssertionError(
                f"{r.file_id}: segments cover {seg} blocks < raw active {raw} "
                f"beyond the dropped-blip allowance — segment extraction is buggy")
        overshoot.append((seg - raw) / r.n_blocks)
    o = np.array(overshoot)
    print(f"[check] segment undershoot: none (all {len(o):,} files consistent)")
    print(f"[check] overshoot vs raw (bridged gaps + tails, fraction of file): "
          f"mean {o.mean():.3f} · p95 {np.percentile(o, 95):.3f} · max {o.max():.3f}")


def _dropped_blip_allowance(env: np.ndarray, cfg: ActivityManifestConfig,
                            min_seg_blocks: int, block_s: float) -> int:
    """Upper bound on raw-active blocks that rule 3 may legitimately remove."""
    above_close = env > cfg.threshold_close_db
    runs = runs_of_true(above_close)
    short = runs[(runs[:, 1] - runs[:, 0]) < min_seg_blocks]
    return int((short[:, 1] - short[:, 0]).sum())


def build_summary(segments: pd.DataFrame, intervals: dict, index: pd.DataFrame,
                  manifest: pd.DataFrame) -> pd.DataFrame:
    """Table 2: per stem_group x genre x split — hours, active fraction, stretch/rest stats.

    Rests are the INTERNAL gaps between a file's stretches (edge silence excluded —
    leading/trailing silence is a different phenomenon and the mixer never draws there
    anyway). Percentiles are computed over all stretches/rests in the cell.
    """
    meta = manifest.set_index("file_id")
    dur = segments.groupby("file_id").dur_s.sum()

    file_rows = []
    for r in index.itertuples(index=False):
        m = meta.loc[r.file_id]
        if pd.isna(m.stem_group):
            continue
        segs = intervals[r.file_id]
        gaps = ((segs[1:, 0] - segs[:-1, 1]) * r.block_s) if len(segs) > 1 else np.empty(0)
        file_rows.append(dict(
            file_id=r.file_id, dataset=m.dataset, stem_group=m.stem_group,
            genre_sub=m.genre_sub, split=m.split, total_s=m.out_duration,
            active_s=float(dur.get(r.file_id, 0.0)),
            seg_lens=(segs[:, 1] - segs[:, 0]) * r.block_s, gap_lens=gaps))
    per_file = pd.DataFrame(file_rows)

    def agg(cell: pd.DataFrame) -> pd.Series:
        seg_lens = np.concatenate(list(cell.seg_lens)) if len(cell) else np.empty(0)
        gap_lens = np.concatenate(list(cell.gap_lens)) if len(cell) else np.empty(0)
        pct = lambda a, q: round(float(np.percentile(a, q)), 2) if a.size else np.nan
        return pd.Series(dict(
            files=len(cell),
            total_h=round(cell.total_s.sum() / 3600, 2),
            active_h=round(cell.active_s.sum() / 3600, 2),
            active_frac=round(cell.active_s.sum() / cell.total_s.sum(), 4),
            n_segments=int(sum(len(s) for s in cell.seg_lens)),
            seg_p25_s=pct(seg_lens, 25), seg_p50_s=pct(seg_lens, 50),
            seg_p75_s=pct(seg_lens, 75),
            gap_p25_s=pct(gap_lens, 25), gap_p50_s=pct(gap_lens, 50),
            gap_p75_s=pct(gap_lens, 75), gap_p95_s=pct(gap_lens, 95)))

    summary = (per_file.groupby(["dataset", "stem_group", "genre_sub", "split"])
               .apply(agg, include_groups=False).reset_index())
    return summary


def build_chunks(intervals: dict, index: pd.DataFrame, manifest: pd.DataFrame,
                 cfg: ActivityManifestConfig) -> pd.DataFrame:
    """Table 3: per 71955 song x window length x window — audible fraction per class.

    Windows are cut on the song's SHORTEST stem (same trim-to-shortest rule as Σstem
    mixing), so no window claims time some stem doesn't have. Same-class multi-tracks
    (피리1/2/3) are OR-ed into one boolean track before coverage is measured.
    """
    idx = index.set_index("file_id")
    stems = manifest[(manifest.dataset == "71955") & (manifest.role == "stem")
                     & manifest.stem_group.notna()]
    all_groups = sorted(manifest.stem_group.dropna().unique())

    rows: list[dict] = []
    for song_id, song_stems in stems.groupby("song_id"):
        first = song_stems.iloc[0]
        n_blocks = int(idx.loc[song_stems.file_id, "n_blocks"].min())  # trim-to-shortest
        block_s = float(idx.loc[song_stems.iloc[0].file_id, "block_s"])

        # rasterise each class: OR of its member stems' segment intervals
        class_active: dict[str, np.ndarray] = {}
        for group, members in song_stems.groupby("stem_group"):
            mask = np.zeros(n_blocks, dtype=bool)
            for fid in members.file_id:
                for b0, b1 in intervals[fid]:
                    mask[b0:min(b1, n_blocks)] = True
            class_active[group] = mask
        n_classes_song = len(class_active)

        for chunk_len_s in cfg.chunk_lengths_s:
            win = int(round(chunk_len_s / block_s))
            n_win = n_blocks // win
            if n_win == 0:
                continue
            # (classes, windows) coverage matrix in one reshape per class
            cov = {g: m[:n_win * win].reshape(n_win, win).mean(axis=1)
                   for g, m in class_active.items()}
            for w in range(n_win):
                row = dict(song_id=song_id, genre_sub=first.genre_sub, split=first.split,
                           chunk_len_s=chunk_len_s, window_idx=w,
                           start_s=round(w * win * block_s, 2),
                           n_classes_song=n_classes_song)
                for g in all_groups:
                    row[f"cov_{g}"] = round(float(cov[g][w]), 4) if g in cov else 0.0
                for t in cfg.coverage_thresholds:
                    n_active = sum(1 for g in cov if cov[g][w] > t)
                    row[f"n_active_gt{int(t * 100):02d}"] = n_active
                rows.append(row)
    return pd.DataFrame(rows)


# --- reporting / io -----------------------------------------------------------

def report(segments: pd.DataFrame, summary: pd.DataFrame, chunks: pd.DataFrame) -> None:
    """One-screen digest: headline numbers for each table."""
    print(f"\nactivity_segments: {len(segments):,} rows "
          f"({segments.file_id.nunique():,} files, "
          f"{segments.dur_s.sum() / 3600:.1f} active hours)")

    print("\nactivity_summary — collapsed to stem_group (train split, both datasets):")
    train = summary[summary.split == "train"]
    g = (train.groupby("stem_group")
         .apply(lambda c: pd.Series(dict(
             total_h=c.total_h.sum(), active_h=c.active_h.sum(),
             active_frac=round(c.active_h.sum() / c.total_h.sum(), 3),
             gap_p50_s=round(np.nanmedian(c.gap_p50_s), 2))), include_groups=False)
         .sort_values("total_h", ascending=False))
    print(g.round(2).to_string())

    print(f"\nchunk_activities: {len(chunks):,} rows ({chunks.song_id.nunique()} songs)")
    for L in sorted(chunks.chunk_len_s.unique()):
        sub = chunks[chunks.chunk_len_s == L]
        print(f"  {L:>4.0f} s windows: designed 편성 mean {sub.n_classes_song.mean():.2f} "
              f"→ audible mean {sub.n_active_gt25.mean():.2f} (coverage >25%)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: envelopes -> activity tables.")
    ap.add_argument("--config", default="configs/activity_scan.yaml")
    ap.add_argument("--dry-run", action="store_true", help="build + report, write nothing")
    args = ap.parse_args()

    root = find_root()
    cfg = ActivityManifestConfig.load(root / args.config, root)
    index = pd.read_parquet(table_paths(cfg.out_index)[0])
    blob = np.load(cfg.envelope_dir / "envelopes.npy", mmap_mode="r")
    manifest = pd.read_parquet(cfg.source_manifest)
    print(f"config={args.config}  files={len(index):,}  "
          f"open/close={cfg.threshold_open_db}/{cfg.threshold_close_db} dB  "
          f"gap<{cfg.min_gap_s}s bridged · seg<{cfg.min_segment_s}s dropped · "
          f"tail+{cfg.release_margin_s}s")

    segments, intervals = build_segments(index, blob, cfg)
    check_segments(index, blob, intervals, cfg)
    summary = build_summary(segments, intervals, index, manifest)
    chunks = build_chunks(intervals, index, manifest, cfg)
    report(segments, summary, chunks)

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return
    for frame, out in [(segments, cfg.out_segments), (summary, cfg.out_summary),
                       (chunks, cfg.out_chunks)]:
        out_parquet, out_csv = table_paths(out)
        frame.to_parquet(out_parquet, index=False)
        frame.to_csv(out_csv, index=False)
        print(f"wrote {out_parquet} + {out_csv}")


if __name__ == "__main__":
    main()
