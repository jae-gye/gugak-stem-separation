"""activity_scan.py — STAGE 1 of the stem activity scan: audio -> dBFS envelopes.

Reads every file in the ingest store once and records a block-RMS level for each
~46 ms of audio. That is all it does: no thresholds, no segments, no decisions. The
output is a raw measurement the cheap stage-2 pass (build_activity_manifest.py) can
re-interpret as often as we like without ever touching 397 GB of audio again.

Why the two-stage split (see Notion · Data pipeline → "Stem activity scan"):
  - stage 1 is I/O-bound over the whole store — minutes to tens of minutes, run rarely
  - stage 2 holds every tunable knob (thresholds, gap/segment shaping, chunk length,
    what "audible" means) — and those WILL be re-tuned
  Coupling them would make every re-tune a full re-read. So they are separate.

Method notes:
  - Blocks are NON-OVERLAPPING, so each block RMS is an exact mean over its samples,
    computed by reshape. A prefix-sum/sliding-window formulation would be faster but
    differencing a large running energy sum loses all precision for a quiet block that
    follows loud content — exactly the measurement this scan exists to make.
  - Level is taken as the MAX across channels, not the mono downmix: a downmix can
    cancel, and reporting an instrument as silent because its channels opposed would
    be the one failure mode that silently corrupts everything downstream.
  - Files are streamed in bounded reads (the longest is 19.2 min) so worker memory
    stays flat regardless of file length.

Output:
  data/activity/envelopes.npy    flat float32 dBFS, all files concatenated (~146 MB)
  manifests/parquet/activity_index.parquet   file_id -> offset, n_blocks, block_s, sr

Run:
    uv run python src/data/activity_scan.py --config configs/activity_scan.yaml
      --limit N        scan only the first N files (smoke test)
      --workers N      parallel processes (default: os.cpu_count()-8, shared box)
      --dry-run        plan + report, write nothing
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml


# --- repo-root discovery (cwd-agnostic, mirrors ingest.py) ---
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
        base: table basename, e.g. Path(".../manifests/activity_index").
    """
    parquet = base.parent / "parquet" / f"{base.name}.parquet"
    csv = base.parent / "csv" / f"{base.name}.csv"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    return parquet, csv


# --- config -----------------------------------------------------------------
@dataclass
class ActivityScanConfig:
    """Stage-1 view of configs/activity_scan.yaml (paths absolute against root)."""
    block_s: float
    frames_per_read: int
    envelope_floor_db: float
    source_manifest: Path
    envelope_dir: Path
    out_index: Path

    @classmethod
    def load(cls, path: Path, root: Path) -> "ActivityScanConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            block_s=float(raw["block_s"]),
            frames_per_read=int(raw["frames_per_read"]),
            envelope_floor_db=float(raw["envelope_floor_db"]),
            source_manifest=root / raw["source_manifest"],
            envelope_dir=root / raw["envelope_dir"],
            out_index=root / raw["out_index"],
        )


# --- envelope extraction ----------------------------------------------------

_CFG: ActivityScanConfig | None = None


def _init_worker(cfg: ActivityScanConfig) -> None:
    """Seed each worker process with the config (avoids re-pickling it per task)."""
    global _CFG
    _CFG = cfg


# scripts/audio_qc.py measured activity as the fraction of individual SAMPLES whose
# max-channel |x| exceeds this amplitude. That is a different quantity from block RMS —
# an oscillating signal dips below it at every zero crossing, so it undercounts playing,
# increasingly so for quiet material. We recompute it here purely as a regression check
# that this scan decodes the audio identically to the earlier one; it is NOT the
# activity measure anything downstream uses.
QC_SILENCE_AMP = 10 ** (-60 / 20)


def block_dbfs_envelope(path: Path, block_s: float, frames_per_read: int,
                        floor_db: float) -> tuple[np.ndarray, int, float]:
    """Per-block dBFS level for one audio file, max across channels.

    Streams the file in reads that are an exact multiple of the block size, so no block
    ever straddles a read boundary. A trailing partial block is dropped.

    Args:
        path: audio file to scan.
        block_s: block duration in seconds; blocks do not overlap.
        frames_per_read: how many blocks to pull per read (bounds memory).
        floor_db: dBFS written for a digitally-silent block.

    Returns:
        (envelope as float32 dBFS, sample rate, QC-compatible sample-level active fraction).
    """
    with sf.SoundFile(str(path)) as handle:
        sr = handle.samplerate
        block_samples = max(1, int(round(block_s * sr)))
        read_samples = block_samples * frames_per_read

        levels: list[np.ndarray] = []
        qc_active_samples, qc_total_samples = 0, 0
        while True:
            audio = handle.read(read_samples, dtype="float32", always_2d=True)
            if len(audio) == 0:
                break
            # QC parity accumulator: counts EVERY sample, including any trailing partial
            # block, because the original scan worked on the whole decoded array.
            qc_active_samples += int((np.abs(audio).max(axis=1) > QC_SILENCE_AMP).sum())
            qc_total_samples += len(audio)

            n_blocks = len(audio) // block_samples
            if n_blocks:
                # (blocks, block_samples, channels) -> per-channel RMS -> loudest channel
                usable = audio[: n_blocks * block_samples]
                shaped = usable.reshape(n_blocks, block_samples, usable.shape[1])
                mean_square = np.square(shaped, dtype=np.float64).mean(axis=1)
                levels.append(np.sqrt(mean_square).max(axis=1))
            if len(audio) < read_samples:
                break

    qc_active_frac = (qc_active_samples / qc_total_samples) if qc_total_samples else 0.0
    if not levels:
        return np.empty(0, dtype=np.float32), sr, qc_active_frac
    rms = np.concatenate(levels)
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(rms)
    return np.maximum(db, floor_db).astype(np.float32), sr, qc_active_frac


def scan_file(task: tuple[str, str]) -> dict:
    """Scan one file into an envelope. Returns a row dict; never raises."""
    file_id, out_path = task
    cfg = _CFG
    assert cfg is not None, "worker not initialised"
    try:
        envelope, sr, qc_active_frac = block_dbfs_envelope(
            Path(out_path), cfg.block_s, cfg.frames_per_read, cfg.envelope_floor_db)
        return dict(file_id=file_id, envelope=envelope, sr=sr, n_blocks=len(envelope),
                    qc_active_frac=qc_active_frac, error="")
    except Exception as exc:  # a single unreadable file must not abort a 397 GB scan
        return dict(file_id=file_id, envelope=np.empty(0, dtype=np.float32), sr=0,
                    n_blocks=0, qc_active_frac=float("nan"),
                    error=f"{type(exc).__name__}: {exc}")


# --- orchestration ----------------------------------------------------------

def build_tasks(cfg: ActivityScanConfig, limit: int | None) -> list[tuple[str, str]]:
    """Read the source manifest and return (file_id, out_path) for every file."""
    manifest = pd.read_parquet(cfg.source_manifest, columns=["file_id", "out_path"])
    root = find_root()
    tasks = [(str(r.file_id), str(root / r.out_path)) for r in manifest.itertuples(index=False)]
    return tasks[:limit] if limit else tasks


def write_outputs(rows: list[dict], cfg: ActivityScanConfig) -> pd.DataFrame:
    """Concatenate envelopes into one blob and write the offset index beside it.

    Rows are re-sorted into manifest order first so the blob layout is deterministic
    and a re-run with the same inputs is byte-identical.
    """
    by_id = {r["file_id"]: r for r in rows}
    ordered = [by_id[fid] for fid in
               pd.read_parquet(cfg.source_manifest, columns=["file_id"]).file_id
               if fid in by_id]

    index_rows, offset = [], 0
    for r in ordered:
        index_rows.append(dict(file_id=r["file_id"], offset=offset, n_blocks=r["n_blocks"],
                               sr=r["sr"], block_s=cfg.block_s,
                               qc_active_frac=r["qc_active_frac"], error=r["error"]))
        offset += r["n_blocks"]

    blob = (np.concatenate([r["envelope"] for r in ordered if r["n_blocks"]])
            if offset else np.empty(0, dtype=np.float32))
    cfg.envelope_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.envelope_dir / "envelopes.npy", blob)

    index = pd.DataFrame(index_rows)
    index_parquet, index_csv = table_paths(cfg.out_index)
    index.to_parquet(index_parquet, index=False)
    index.to_csv(index_csv, index=False)
    return index


def report(index: pd.DataFrame, cfg: ActivityScanConfig, elapsed: float) -> None:
    """Print a scan summary, plus the two checks that decide whether stage 1 is trustworthy.

    Check 1 (decode parity, must be ~exact): re-derive scripts/audio_qc.py's sample-level
    active_frac and compare against the stored value. Same definition, independent run —
    any real disagreement means this scan is not reading the audio the earlier one read,
    and every table built on top would inherit the fault.

    Check 2 (duration conservation): blocks x block_s must reproduce the manifest duration
    up to the dropped trailing partial block.

    The block-RMS activity fraction is also printed, but as a MEASUREMENT, not a test —
    it legitimately differs from the QC number (see QC_SILENCE_AMP).
    """
    blob_mb = (cfg.envelope_dir / "envelopes.npy").stat().st_size / 1e6
    failed = index[index.error != ""]
    print(f"\nscanned {len(index):,} files in {elapsed/60:.1f} min  ·  "
          f"{index.n_blocks.sum():,} blocks  ·  envelopes {blob_mb:.0f} MB")
    if len(failed):
        print(f"⚠️  {len(failed)} failures (first 5):")
        for r in failed.head(5).itertuples(index=False):
            print(f"   {r.file_id}: {r.error}")

    manifest = pd.read_parquet(cfg.source_manifest,
                               columns=["file_id", "active_frac", "out_duration"])
    merged = index.merge(manifest, on="file_id")
    merged = merged[(merged.n_blocks > 0) & merged.active_frac.notna()]

    # check 1 — decode parity against the previous scan (same metric, so expect ~0)
    parity = np.abs(merged.qc_active_frac.to_numpy() - merged.active_frac.to_numpy())
    verdict = "PASS" if parity.max() < 1e-6 else "⚠️  MISMATCH"
    print(f"\n[check 1] decode parity vs audio_qc active_frac ({len(merged):,} files): {verdict}")
    print(f"   mean |Δ| {parity.mean():.2e}  ·  max |Δ| {parity.max():.2e}")
    if parity.max() >= 1e-6:
        worst = merged.assign(delta=parity).nlargest(3, "delta")
        for r in worst.itertuples(index=False):
            print(f"   {r.file_id}: qc={r.active_frac:.6f} rescanned={r.qc_active_frac:.6f}")

    # check 2 — duration conservation
    expected, got = merged.out_duration.sum(), merged.n_blocks.sum() * cfg.block_s
    drift = 100 * (got - expected) / expected
    print(f"\n[check 2] duration {got/3600:.2f} h vs manifest {expected/3600:.2f} h  "
          f"({drift:+.2f}%, partial trailing blocks dropped): "
          f"{'PASS' if -0.5 < drift <= 0 else '⚠️  CHECK'}")

    # measurement — block-RMS activity, the quantity stage 2 actually thresholds
    blob = np.load(cfg.envelope_dir / "envelopes.npy", mmap_mode="r")
    block_active = np.array([(np.asarray(blob[r.offset:r.offset + r.n_blocks]) > -60.0).mean()
                             for r in merged.itertuples(index=False)])
    weights = merged.out_duration.to_numpy()
    print(f"\n[measure] block-RMS active fraction @ -60 dB, duration-weighted: "
          f"{np.average(block_active, weights=weights):.4f}")
    print(f"   (audio_qc's sample-level metric gives "
          f"{np.average(merged.active_frac, weights=weights):.4f} — it dips below threshold "
          f"at every zero crossing, so it reads low; the block figure is the real one)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: block-dBFS envelopes for the ingest store.")
    ap.add_argument("--config", default="configs/activity_scan.yaml")
    ap.add_argument("--limit", type=int, default=None, help="scan only first N files (smoke test)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 8))
    ap.add_argument("--dry-run", action="store_true", help="plan + report, write nothing")
    args = ap.parse_args()

    root = find_root()
    cfg = ActivityScanConfig.load(root / args.config, root)
    tasks = build_tasks(cfg, args.limit)
    print(f"config={args.config}  files={len(tasks):,}  workers={args.workers}  "
          f"block={cfg.block_s*1000:.0f} ms  dry_run={args.dry_run}")
    if args.dry_run:
        print("[dry-run] nothing written")
        return

    # scan (parallel across files — every task is independent and read-only)
    start = time.time()
    rows: list[dict] = []
    if args.workers <= 1:
        _init_worker(cfg)
        for i, t in enumerate(tasks, 1):
            rows.append(scan_file(t))
            if i % 500 == 0:
                print(f"  {i:,}/{len(tasks):,}  ({time.time()-start:.0f}s)")
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker, initargs=(cfg,)) as ex:
            futures = [ex.submit(scan_file, t) for t in tasks]
            for i, f in enumerate(as_completed(futures), 1):
                rows.append(f.result())
                if i % 500 == 0:
                    print(f"  {i:,}/{len(tasks):,}  ({time.time()-start:.0f}s)")

    index = write_outputs(rows, cfg)
    report(index, cfg, time.time() - start)
    index_parquet, index_csv = table_paths(cfg.out_index)
    print(f"\nwrote {cfg.envelope_dir / 'envelopes.npy'} + {index_parquet} + {index_csv}")


if __name__ == "__main__":
    main()
