#!/usr/bin/env python3
"""Per-stem quality control: silence / dead-stem / clipping / loudness over the manifest.

For each stem: duration, peak, RMS, active fraction (non-silent blocks), leading/trailing
silence, clipping fraction + longest clipped run, and integrated loudness (LUFS, ITU-R
BS.1770). Feeds dataloader decisions: which stems to drop (dead), whether to loudness-
normalize, and whether clipping needs handling.

Usage:
  uv run python scripts/stem_qc.py --per-genre 5        # stratified sample
  uv run python scripts/stem_qc.py --all                # full 5,767 stems (tmux)

Writes <out>.parquet + <out>.csv (default data/eda/stem_qc[_sample]).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyloudnorm as pyln

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.audio import (active_fraction, clipping_fraction, edge_silence,  # noqa: E402
                            peak, read_audio, rms, to_mono)

SEED = 42


def select_songs(songs: pd.DataFrame, args) -> pd.DataFrame:
    if args.all:
        return songs
    if args.songs:
        return songs[songs.song_id.isin(args.songs)]
    parts = [d.sample(min(len(d), args.per_genre), random_state=SEED)
             for _, d in songs.groupby("genre_sub")]
    return pd.concat(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true")
    g.add_argument("--per-genre", type=int, default=5)
    ap.add_argument("--songs", nargs="+")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    songs = pd.read_parquet(ROOT / "manifests" / "songs.parquet")
    stems = pd.read_parquet(ROOT / "manifests" / "stems.parquet")
    sel_ids = set(select_songs(songs, args).song_id)
    sub = stems[stems.song_id.isin(sel_ids)].reset_index(drop=True)

    default = "stem_qc" if (args.all or args.songs) else "stem_qc_sample"
    out = Path(args.out) if args.out else ROOT / "data" / "eda" / default
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"stem QC on {len(sub)} stems from {len(sel_ids)} songs -> {out}.parquet")

    meters: dict[int, pyln.Meter] = {}
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pyloudnorm complains on near-silent blocks
        for i, r in sub.iterrows():
            rec = dict(song_id=r.song_id, genre_sub=r.genre_sub, instrument=r.instrument,
                       instrument_base=r.instrument_base, stem_group_4=r.stem_group_4, split=r.split)
            try:
                audio, sr = read_audio(ROOT / r.stem_path)
                mono = to_mono(audio)
                pk = peak(audio)
                clip_frac, clip_run = clipping_fraction(audio)
                lead, trail = edge_silence(mono, sr)
                if sr not in meters:
                    meters[sr] = pyln.Meter(sr)
                lufs = meters[sr].integrated_loudness(audio)
                rec.update(dur_s=round(len(audio) / sr, 2), sr=sr, peak=round(pk, 5),
                           rms=round(rms(audio), 6), active_frac=round(active_fraction(mono, sr), 4),
                           lead_sil_s=round(lead, 2), trail_sil_s=round(trail, 2),
                           clip_frac=clip_frac, clip_run=clip_run,
                           lufs=(round(float(lufs), 2) if np.isfinite(lufs) else float("-inf")),
                           dead=bool(pk < 1e-4))
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"
            rows.append(rec)
            if (i + 1) % 50 == 0 or i + 1 == len(sub):
                print(f"  {i + 1}/{len(sub)}")

    df = pd.DataFrame(rows)
    df.to_parquet(f"{out}.parquet", index=False)
    df.to_csv(f"{out}.csv", index=False)

    ok = df[df.get("active_frac").notna()] if "active_frac" in df else df.iloc[0:0]
    print(f"\ndone. {len(ok)}/{len(df)} stems measured")
    if not len(ok):
        return
    print(f"\nDEAD stems (peak<1e-4): {int(ok.dead.sum())}")
    print(f"near-silent (active<2%): {(ok.active_frac < 0.02).sum()}")
    print(f"clipping (clip_frac>0):  {(ok.clip_frac > 0).sum()}  "
          f"(>0.01%: {(ok.clip_frac > 1e-4).sum()})")
    print("\nactive_fraction distribution:")
    print(ok.active_frac.describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())
    fin = ok[np.isfinite(ok.lufs)]
    print(f"\nintegrated loudness LUFS (n={len(fin)}, {int((~np.isfinite(ok.lufs)).sum())} silent):")
    print(fin.lufs.describe(percentiles=[.05, .5, .95]).round(2).to_string())
    spread = fin.groupby("song_id").lufs.agg(lambda s: s.max() - s.min())
    print(f"\nwithin-song LUFS spread (loudest-quietest stem): median {spread.median():.1f} dB, "
          f"max {spread.max():.1f} dB")


if __name__ == "__main__":
    main()
