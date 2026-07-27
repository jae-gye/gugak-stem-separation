#!/usr/bin/env python
"""audio_qc.py — one-time audio quality-control scan across the gugak datasets.

Per-file checks
---------------
Header (cheap, no decode):
  sample rate · bit depth / encoding (libsndfile subtype) · channel count · duration
Content (needs decode):
  peak · RMS · DC offset · clipping fraction · active/silence fraction + dead flag ·
  and — for stereo files — L/R inter-channel correlation, which separates TRUE stereo
  from dual-mono (fake stereo). The dual-mono question is the Notion "Stereo QC" task:
  L/R-swap augmentation only creates new material when channels genuinely differ.

The script is dataset-agnostic: datasets are entries in DATASETS (root + glob), and
every path/threshold is overridable on the CLI. Constants live at the top, not inline.

Examples
--------
  uv run python scripts/audio_qc.py 71955 71470                 # both known datasets, full scan
  uv run python scripts/audio_qc.py 71955 --checks header       # headers only (fast)
  uv run python scripts/audio_qc.py 71470 --max-seconds 60      # cap decode to 60 s/file
  uv run python scripts/audio_qc.py --name foo --root some/dir --glob '**/*.wav'
"""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

# --- QC thresholds (tune here, never inline) ---
CLIP_LEVEL = 0.9995                     # |x| >= this => a clipped sample
SILENCE_AMP = 10 ** (-60 / 20)          # ~1e-3; per-frame amplitude below this => "silent"
DEAD_PEAK = 1e-4                        # whole-file peak below this => dead / empty
DUAL_MONO_CORR = 0.9999                 # L/R Pearson corr above this => effectively dual-mono

# --- dataset registry: repo-relative root + glob; `role` tags non-master files ---
DATASETS: dict[str, dict] = {
    "71955": {"root": "data/gugak_ensemble_71955", "glob": "source/*/*.wav", "role": "stem",
              "label": "71955 ensemble (stems + masters)"},
    "71470": {"root": "data/gugak_solo_71470", "glob": "audio/*.wav", "role": "clip",
              "label": "71470 solo (clips)"},
}


# ---------- individual checks (operate on plain data, so they're unit-testable) ----------
def header_metrics(path: Path) -> dict:
    """Cheap header read — no audio decode."""
    info = sf.info(str(path))
    dur = info.frames / info.samplerate if info.samplerate else 0.0
    return {"sr": info.samplerate, "subtype": info.subtype,
            "channels": info.channels, "frames": info.frames, "duration": dur}


def content_metrics(x: np.ndarray) -> dict:
    """Signal-level checks on a decoded array of shape (frames, channels), float."""
    if x.size == 0:
        return {"peak": 0.0, "rms": 0.0, "dc_offset": 0.0, "clip_frac": 0.0,
                "active_frac": 0.0, "dead": True,
                "lr_corr": math.nan, "lr_identical": math.nan, "lr_max_diff": math.nan}
    peak = float(np.abs(x).max())
    m = {
        "peak": peak,
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "dc_offset": float(np.mean(x)),
        "clip_frac": float(np.mean(np.abs(x) >= CLIP_LEVEL)),
        "active_frac": float(np.mean(np.abs(x).max(axis=1) > SILENCE_AMP)),
        "dead": bool(peak < DEAD_PEAK),
    }
    if x.shape[1] == 2:                                   # L/R only meaningful for stereo
        left, right = x[:, 0], x[:, 1]
        if left.std() < 1e-9 or right.std() < 1e-9:      # constant channel => corr undefined
            corr = 1.0 if np.allclose(left, right) else 0.0
        else:
            corr = float(np.corrcoef(left, right)[0, 1])
        m.update(lr_corr=corr,
                 lr_identical=bool(np.array_equal(left, right)),
                 lr_max_diff=float(np.abs(left - right).max()))
    else:
        m.update(lr_corr=math.nan, lr_identical=math.nan, lr_max_diff=math.nan)
    return m


# ---------- per-file worker (top-level so it's picklable for ProcessPoolExecutor) ----------
def qc_file(task: tuple) -> dict:
    path, dataset, role, do_content, max_seconds = task
    rec = {"dataset": dataset, "role": role, "path": str(path)}
    try:
        head = header_metrics(path)
        rec.update(head)
        if do_content:
            n = int(max_seconds * head["sr"]) if (max_seconds and max_seconds > 0) else -1
            x, _ = sf.read(str(path), frames=n, dtype="float32", always_2d=True)
            rec.update(content_metrics(x))
        rec["error"] = ""
    except Exception as exc:  # noqa: BLE001 - QC must never abort on one bad file
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


# ---------- enumeration ----------
def enumerate_tasks(names, root_override, glob_override, do_content, max_seconds, limit):
    """Yield (path, dataset, role, do_content, max_seconds) for every file to scan."""
    for name in names:
        spec = DATASETS.get(name, {})
        root = Path(root_override or spec.get("root", "."))
        glob = glob_override or spec.get("glob", "**/*.wav")
        default_role = spec.get("role", "file")
        paths = sorted(root.glob(glob))
        if limit:
            paths = paths[:limit]
        if not paths:
            print(f"  ! {name}: no files matched {root}/{glob}")
        for p in paths:
            role = "master" if p.stem.endswith("_master") else default_role
            yield (p, name, role, do_content, max_seconds)


# ---------- reporting ----------
def summarize(df: pd.DataFrame) -> None:
    for name, d in df.groupby("dataset"):
        print(f"\n{'=' * 70}\n{name}  —  {len(d):,} files"
              f"  ({d['error'].astype(bool).sum()} errors)\n{'=' * 70}")
        ok = d[d["error"] == ""]
        # roles
        if ok["role"].nunique() > 1:
            print("roles:", ok["role"].value_counts().to_dict())
        # format catalogue
        print("\n-- sample rate --");  print(ok["sr"].value_counts().to_string())
        print("\n-- subtype (bit/encoding) --"); print(ok["subtype"].value_counts().to_string())
        print("\n-- channels --"); print(ok["channels"].value_counts().to_string())
        combo = ok.groupby(["sr", "subtype", "channels"]).size().sort_values(ascending=False)
        print(f"\n-- {len(combo)} distinct (sr, subtype, channels) combos --")
        print(combo.to_string())
        print(f"\ntotal audio: {ok['duration'].sum() / 3600:.1f} h"
              f"  ·  duration s: min {ok['duration'].min():.1f} / med {ok['duration'].median():.1f}"
              f" / max {ok['duration'].max():.1f}")
        # content checks (only if present)
        if "peak" in ok.columns and ok["peak"].notna().any():
            dead = int(ok["dead"].sum())
            clipped = ok[ok["clip_frac"] > 1e-4]
            over1 = int((ok["peak"] > 1.0).sum())
            print(f"\n-- content --\n  dead/silent files: {dead}"
                  f"\n  files with clipping (clip_frac>1e-4): {len(clipped)}"
                  f"  (max clip_frac {ok['clip_frac'].max():.4f})"
                  f"\n  files peaking >1.0 (float over-range): {over1}"
                  f"\n  |DC offset| max: {ok['dc_offset'].abs().max():.5f}"
                  f"\n  active_frac: min {ok['active_frac'].min():.3f}"
                  f" / med {ok['active_frac'].median():.3f}")
            # stereo / dual-mono verdict
            st = ok[ok["channels"] == 2].copy()
            if len(st):
                dual = st[(st["lr_corr"] >= DUAL_MONO_CORR) | (st["lr_identical"] == True)]  # noqa: E712
                true_st = len(st) - len(dual)
                print(f"\n-- stereo L/R ({len(st):,} stereo files) --"
                      f"\n  dual-mono (corr>={DUAL_MONO_CORR} or identical): {len(dual):,}"
                      f"  ({len(dual) / len(st) * 100:.1f}%)"
                      f"\n  true stereo: {true_st:,}  ({true_st / len(st) * 100:.1f}%)"
                      f"\n  L/R corr: min {st['lr_corr'].min():.4f}"
                      f" / med {st['lr_corr'].median():.4f}")
        errs = d[d["error"] != ""]
        if len(errs):
            print(f"\n-- errors ({len(errs)}) --")
            print(errs["error"].value_counts().head(5).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Audio QC scan across gugak datasets.")
    ap.add_argument("names", nargs="*", default=list(DATASETS),
                    help=f"dataset keys to scan (default: all = {list(DATASETS)})")
    ap.add_argument("--root", help="override dataset root (use with a single --name)")
    ap.add_argument("--glob", help="override file glob")
    ap.add_argument("--checks", choices=["header", "content", "both"], default="both")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="decode at most N s/file for content checks (0 = full file)")
    ap.add_argument("--workers", type=int, default=max(1, (Path("/proc/cpuinfo").exists()
                    and __import__("os").cpu_count() or 4) - 2))
    ap.add_argument("--out", default="manifests/audio_qc",
                    help="output basename (writes .parquet + .csv)")
    ap.add_argument("--limit", type=int, default=0, help="cap files per dataset (for testing)")
    args = ap.parse_args()

    do_content = args.checks in ("content", "both")
    tasks = list(enumerate_tasks(args.names, args.root, args.glob,
                                 do_content, args.max_seconds, args.limit))
    print(f"scanning {len(tasks):,} files · checks={args.checks}"
          f" · max_seconds={args.max_seconds or 'full'} · workers={args.workers}")

    records, done = [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(qc_file, t) for t in tasks]
        for fut in as_completed(futures):
            records.append(fut.result())
            done += 1
            if done % 1000 == 0 or done == len(tasks):
                print(f"  {done:,}/{len(tasks):,}", flush=True)

    df = pd.DataFrame(records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out.with_suffix(".parquet"), index=False)
    df.to_csv(out.with_suffix(".csv"), index=False)
    print(f"\nwrote {out}.parquet + .csv  ({len(df):,} rows)")
    summarize(df)


if __name__ == "__main__":
    main()
