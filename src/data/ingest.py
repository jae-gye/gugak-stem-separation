"""ingest.py — offline deterministic preprocessing of BOTH gugak datasets.

One-shot, per-file, idempotent. Turns the format-heterogeneous raw audio (71955
ensemble masters+stems, 71470 solo clips) into a training-ready store at a single
canonical rate/format, and emits a processed manifest the dataloaders walk.

This is strictly a PER-FILE transform — no stem-summing, no Σstem mixing, no
pitch-shifting (all separate, later stages). Op chain per file (see Notion ·
Training Data Strategy → "Offline ingest"):

    read → channel rule → DC removal → resample(44.1k) → peak clamp → write PCM_24 → manifest row

Design:
  - The full QC scan (manifests/audio_qc.parquet) is the per-file DECISION TABLE:
    it already holds sr / subtype / channels / L-R corr for every file, so channel
    decisions reuse the documented QC thresholds instead of re-scanning. We never
    walk directories.
  - Idempotent: existing outputs are skipped (still recorded, ops="cached") → cheap,
    resumable re-runs. `--overwrite` forces a rewrite.
  - Config-driven: every knob (target sr, peak target, corr thresholds, paths) lives
    in configs/ingest.yaml.
  - Outputs land in ingest/ INSIDE each dataset dir (→ server NVMe via the symlinks).

Run:
    uv run python src/data/ingest.py --config configs/ingest.yaml [options]
      --dataset {71955,71470,both}   which set(s) to ingest (default: both)
      --limit N                      process only the first N files (smoke test)
      --dry-run                      plan + report, write nothing
      --overwrite                    re-process even if the output exists
      --workers N                    parallel processes (default: os.cpu_count()-2)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
import yaml

# --- repo-root discovery (cwd-agnostic, mirrors data_splitter.py) ---
def find_root(start: Path | None = None) -> Path:
    """Locate the repo root (holds pyproject.toml)."""
    p = Path.cwd() if start is None else start
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists():
            return cand
    raise FileNotFoundError("repo root (pyproject.toml) not found above cwd")


def nfc(s: str) -> str:
    """NFC-normalize a Korean string (disk names vs CSV must match on join)."""
    return unicodedata.normalize("NFC", str(s))


# --- config -----------------------------------------------------------------
@dataclass
class IngestConfig:
    """Flat view of configs/ingest.yaml (paths resolved to absolute against root)."""
    target_sr: int
    output_subtype: str
    resampler_quality: str
    peak_ceiling: float
    peak_target: float
    dualmono_corr: float
    antiphase_corr: float
    keep_channel: int
    qc_parquet: Path
    eval_manifest: Path
    out_manifest: Path
    datasets: dict  # {"71955": {"ingest_dir": Path}, "71470": {"ingest_dir", "labels_dir"}}

    @classmethod
    def load(cls, path: Path, root: Path) -> "IngestConfig":
        raw = yaml.safe_load(Path(path).read_text())
        ds = {}
        for key, d in raw["datasets"].items():
            ds[str(key)] = {k: (root / v) for k, v in d.items()}
        return cls(
            target_sr=int(raw["target_sr"]),
            output_subtype=str(raw["output_subtype"]),
            resampler_quality=str(raw["resampler_quality"]),
            peak_ceiling=float(raw["peak_ceiling"]),
            peak_target=float(raw["peak_target"]),
            dualmono_corr=float(raw["dualmono_corr"]),
            antiphase_corr=float(raw["antiphase_corr"]),
            keep_channel=int(raw["keep_channel"]),
            qc_parquet=root / raw["qc_parquet"],
            eval_manifest=root / raw["eval_manifest"],
            out_manifest=root / raw["out_manifest"],
            datasets=ds,
        )


# --- per-file ops -----------------------------------------------------------
# Each op is small and independent so the chain is auditable (ops_applied string).

def decide_channel_action(channels: int, lr_corr: float, lr_identical: bool,
                          cfg: IngestConfig) -> str:
    """'keep_mono' | 'keep_stereo' | 'pick_channel' from QC L/R correlation.

    dual-mono (corr ~ +1, or byte-identical) and anti-phase (corr ~ -1) collapse to
    one channel — absolute polarity is inaudible and a single channel avoids
    comb-filter artifacts near ±1. Genuinely decorrelated stereo is kept.
    """
    if channels == 1:
        return "keep_mono"
    corr = float(lr_corr) if lr_corr is not None and not pd.isna(lr_corr) else 0.0
    if bool(lr_identical) or corr >= cfg.dualmono_corr:
        return "pick_channel"          # dual-mono
    if corr <= cfg.antiphase_corr:
        return "pick_channel"          # anti-phase
    return "keep_stereo"


def apply_channel(audio: np.ndarray, action: str, keep_channel: int) -> np.ndarray:
    """Collapse to one channel when the action says so; else leave shape untouched."""
    if action == "pick_channel":
        return audio[:, [keep_channel]]   # (frames, 1), stays 2-D
    return audio


def remove_dc(audio: np.ndarray) -> np.ndarray:
    """Subtract per-channel mean (DC offset up to 0.165 seen in 71470 float clips)."""
    return audio - audio.mean(axis=0, keepdims=True)


def resample_to(audio: np.ndarray, src_sr: int, target_sr: int, quality: str) -> np.ndarray:
    """soxr resample (VHQ). No-op when already at target. Keeps (frames, ch) shape."""
    if src_sr == target_sr:
        return audio
    return soxr.resample(audio, src_sr, target_sr, quality=quality)


def peak_normalize(audio: np.ndarray, ceiling: float, target: float):
    """Scale down to `target` only if peak exceeds `ceiling`. Returns (audio, before, after)."""
    before = float(np.abs(audio).max()) if audio.size else 0.0
    if before > ceiling:
        audio = audio * (target / before)
        return audio, before, float(np.abs(audio).max())
    return audio, before, before


# --- identity extraction (fast, no audio I/O) -------------------------------
_TRAILING_DIGITS = re.compile(r"\d+$")


def identity_71955(src_path: Path, split_map: dict) -> dict:
    """song_id / instrument(_raw) / genre / split for a 71955 master or stem file.

    Filename is `<song_id>_<instrument>.wav` (instrument == 'master' for the mix).
    Multi-instrument stems (피리1/2/3) keep the raw token AND a digit-stripped base
    (피리) — the base strip is deterministic; the contested stem-class clumping is a
    live decision and stays OUT of ingest.
    """
    song_id = src_path.parent.name
    stem = src_path.stem                       # <song_id>_<instrument>
    prefix = song_id + "_"
    token = stem[len(prefix):] if stem.startswith(prefix) else stem.split("_")[-1]
    is_master = token == "master"
    instrument_raw = None if is_master else token
    instrument = None if is_master else _TRAILING_DIGITS.sub("", token)
    meta = split_map.get(nfc(song_id), {})
    return {
        "song_id": song_id, "clip_id": None,
        "instrument_raw": instrument_raw, "instrument": instrument,
        "genre_sub": meta.get("genre_sub"), "split": meta.get("split"),
    }


def identity_71470(src_path: Path, labels_dir: Path) -> dict:
    """clip_id / instrument_cd / genre_cd for a 71470 solo clip (train-only pool).

    instrument name mapping (SR02 → 해금) is the phrase-mined taxonomy — deferred to a
    later join; here we store the raw instrument_cd + genre_cd from the label JSON.
    """
    clip_id = src_path.stem
    instrument_cd = genre_cd = None
    label = labels_dir / f"{clip_id}.json"
    if label.exists():
        info = json.loads(label.read_text()).get("music_type_info", {})
        instrument_cd = info.get("instrument_cd")
        genre_cd = info.get("music_genre_cd")
    return {
        "song_id": None, "clip_id": clip_id,
        "instrument_raw": instrument_cd, "instrument": instrument_cd,
        "genre_sub": genre_cd, "split": "train",
    }


def output_path(dataset: str, src_path: Path, ingest_dir: Path) -> Path:
    """Map a source file to its ingest/ counterpart.

    71955: ingest/<song>/<file>.wav (mirror source/ layout) · 71470: ingest/<clip>.wav (flat).
    """
    if dataset == "71955":
        return ingest_dir / src_path.parent.name / src_path.name
    return ingest_dir / src_path.name


# --- the full per-file transform (runs in a worker process) -----------------
def process_task(task: dict) -> dict:
    """Ingest one file end-to-end; return its manifest row (never raises → error field)."""
    row = dict(task)                     # carries identity + src_* fields already
    src = Path(task["src_path_abs"])
    out = Path(task["out_path_abs"])
    row["error"] = ""

    # skip existing (idempotent / resumable) — still emit a row from the output header
    if out.exists() and not task["overwrite"]:
        try:
            info = sf.info(str(out))
            row.update(out_sr=info.samplerate, out_channels=info.channels,
                       out_frames=info.frames, out_duration=round(info.frames / info.samplerate, 3),
                       peak_before=None, peak_after=None, peak_normalized=None,
                       resampled=None, channel_action=None, ops_applied="cached", cached=True)
        except Exception as exc:                       # noqa: BLE001 — record, don't crash the run
            row["error"] = f"stat-existing: {exc}"
        return row

    if task["dry_run"]:
        row.update(out_sr=task["target_sr"], out_channels=None, out_frames=None,
                   out_duration=None, peak_before=None, peak_after=None, peak_normalized=None,
                   resampled=(task["src_sr"] != task["target_sr"]),
                   channel_action=None, ops_applied="dry-run", cached=False)
        return row

    try:
        # 1. read (float) → 2. channel rule → 3. DC → 4. resample → 5. peak clamp → 6. write
        audio, src_sr = sf.read(str(src), dtype="float32", always_2d=True)
        ops = []

        action = decide_channel_action(task["src_channels"], task["lr_corr"],
                                        task["lr_identical"], _CFG)
        audio = apply_channel(audio, action, _CFG.keep_channel)
        if action == "pick_channel":
            ops.append(f"pick_ch{_CFG.keep_channel}")

        audio = remove_dc(audio); ops.append("dc")

        resampled = src_sr != _CFG.target_sr
        if resampled:
            audio = resample_to(audio, src_sr, _CFG.target_sr, _CFG.resampler_quality)
            ops.append(f"resample({src_sr}->{_CFG.target_sr})")

        audio, pk_before, pk_after = peak_normalize(audio, _CFG.peak_ceiling, _CFG.peak_target)
        normalized = pk_after != pk_before
        if normalized:
            ops.append(f"peak_norm({pk_before:.3f}->{pk_after:.3f})")

        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, _CFG.target_sr, subtype=_CFG.output_subtype)

        row.update(out_sr=_CFG.target_sr, out_channels=int(audio.shape[1]),
                   out_frames=int(audio.shape[0]),
                   out_duration=round(audio.shape[0] / _CFG.target_sr, 3),
                   peak_before=round(pk_before, 6), peak_after=round(pk_after, 6),
                   peak_normalized=bool(normalized), resampled=bool(resampled),
                   channel_action=action, ops_applied=";".join(ops), cached=False)
    except Exception as exc:                           # noqa: BLE001
        row["error"] = f"process: {exc}"
    return row


# --- worker config bootstrap (avoid re-pickling the config per task) --------
_CFG: IngestConfig | None = None


def _init_worker(cfg: IngestConfig) -> None:
    global _CFG
    _CFG = cfg


# --- worklist assembly ------------------------------------------------------
def build_tasks(cfg: IngestConfig, which: str, overwrite: bool, dry_run: bool,
                limit: int | None) -> list[dict]:
    """Turn the QC scan into a list of per-file tasks with identity pre-resolved.

    The QC parquet is the file list + the src-format decision table; the eval manifest
    supplies genre/split for 71955. Nothing walks the filesystem.
    """
    root = find_root()
    qc = pd.read_parquet(cfg.qc_parquet)
    qc["dataset"] = qc["dataset"].astype(str)

    wanted = {"71955", "71470"} if which == "both" else {which}
    qc = qc[qc["dataset"].isin(wanted)].reset_index(drop=True)

    # 71955 split/genre lookup, keyed by NFC song_id
    ev = pd.read_parquet(cfg.eval_manifest)
    split_map = {nfc(r.song_id): {"genre_sub": r.genre_sub, "split": r.split}
                 for r in ev.itertuples(index=False)}

    tasks: list[dict] = []
    for r in qc.itertuples(index=False):
        ds = r.dataset
        src = root / r.path
        ingest_dir = cfg.datasets[ds]["ingest_dir"]
        out = output_path(ds, src, ingest_dir)

        if ds == "71955":
            ident = identity_71955(src, split_map)
        else:
            ident = identity_71470(src, cfg.datasets[ds]["labels_dir"])

        tasks.append({
            "dataset": ds, "role": r.role,
            **ident,
            "src_path": r.path, "out_path": str(out.relative_to(root)),
            "src_path_abs": str(src), "out_path_abs": str(out),
            "src_sr": int(r.sr), "src_subtype": r.subtype, "src_channels": int(r.channels),
            "src_frames": int(r.frames),
            "lr_corr": (None if pd.isna(r.lr_corr) else float(r.lr_corr)),
            "lr_identical": bool(r.lr_identical),
            "out_subtype": cfg.output_subtype, "target_sr": cfg.target_sr,
            "overwrite": overwrite, "dry_run": dry_run,
        })
        if limit and len(tasks) >= limit:
            break
    return tasks


# --- manifest column order (stable, readable) -------------------------------
_COLS = [
    "dataset", "role", "song_id", "clip_id", "instrument_raw", "instrument",
    "genre_sub", "split",
    "src_path", "out_path",
    "src_sr", "out_sr", "src_subtype", "out_subtype", "src_channels", "out_channels",
    "src_frames", "out_frames", "out_duration",
    "channel_action", "resampled", "peak_before", "peak_after", "peak_normalized",
    "ops_applied", "cached", "error",
]


def write_manifest(rows: list[dict], cfg: IngestConfig) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in _COLS:                       # ensure every column exists even if all-None
        if c not in df.columns:
            df[c] = None
    df = df[_COLS].sort_values(["dataset", "role", "src_path"]).reset_index(drop=True)
    cfg.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.out_manifest.with_suffix(".parquet"), index=False)
    df.to_csv(cfg.out_manifest.with_suffix(".csv"), index=False)
    return df


def report(df: pd.DataFrame) -> None:
    """Concise run summary: counts, errors, and what actually got transformed."""
    n = len(df)
    errs = df[df["error"].fillna("") != ""]
    done = df[(df["error"].fillna("") == "") & (df["cached"] != True)]  # noqa: E712
    cached = df[df["cached"] == True]                                   # noqa: E712
    print(f"\n=== ingest summary — {n} files ===")
    print(df.groupby(["dataset", "role"]).size().to_string())
    print(f"\nprocessed: {len(done)} · cached(skipped): {len(cached)} · errors: {len(errs)}")
    proc = df[df["cached"] != True]                                     # noqa: E712
    if len(proc):
        print(f"resampled: {int(proc['resampled'].fillna(False).sum())} · "
              f"peak-normalized: {int(proc['peak_normalized'].fillna(False).sum())}")
        ca = proc["channel_action"].value_counts(dropna=True)
        if len(ca):
            print("channel actions: " + " · ".join(f"{k} {v}" for k, v in ca.items()))
    if len(errs):
        print("\n⚠️  errors (first 10):")
        for r in errs.head(10).itertuples(index=False):
            print(f"  {r.src_path}: {r.error}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline deterministic ingest of both gugak datasets.")
    ap.add_argument("--config", default="configs/ingest.yaml")
    ap.add_argument("--dataset", choices=["71955", "71470", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="process only first N files (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="plan + report, write no audio")
    ap.add_argument("--overwrite", action="store_true", help="re-process even if output exists")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    root = find_root()
    cfg = IngestConfig.load(root / args.config, root)
    tasks = build_tasks(cfg, args.dataset, args.overwrite, args.dry_run, args.limit)
    print(f"config={args.config}  dataset={args.dataset}  files={len(tasks)}  "
          f"workers={args.workers}  dry_run={args.dry_run}  overwrite={args.overwrite}")

    # process (parallel across files — each task is independent + idempotent)
    rows: list[dict] = []
    if args.workers <= 1:
        _init_worker(cfg)
        for i, t in enumerate(tasks, 1):
            rows.append(process_task(t))
            if i % 500 == 0:
                print(f"  {i}/{len(tasks)}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker, initargs=(cfg,)) as ex:
            futs = [ex.submit(process_task, t) for t in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if i % 500 == 0:
                    print(f"  {i}/{len(tasks)}")

    df = write_manifest(rows, cfg)
    report(df)
    print(f"\nwrote {cfg.out_manifest}.parquet + .csv")


if __name__ == "__main__":
    main()
