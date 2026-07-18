#!/usr/bin/env python3
"""master ≈ Σ(stems) residual test over the manifest.

For each song: sum its stems, align the master to the sum, fit an optimal scalar gain,
and record the residual RMS relative to the master (dB) — raw and gain-compensated.
Low residual dB => linear mixing holds => building training mixtures by summing stems is
safe. This is the go/no-go for the summing dataloader.

Usage:
  uv run python scripts/residual_test.py --per-genre 3          # stratified sample
  uv run python scripts/residual_test.py --all                  # full 903 (do in tmux)
  uv run python scripts/residual_test.py --songs 0979_창작국악_창작국악 0001_정악_풍류음악

Writes <out>.parquet + <out>.csv (default data/eda/residuals[_sample]).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.audio import residual_report  # noqa: E402

SEED = 42


def select_songs(songs: pd.DataFrame, args) -> pd.DataFrame:
    if args.all:
        return songs
    if args.songs:
        return songs[songs.song_id.isin(args.songs)]
    # stratified: N per genre_sub (reproducible). Explicit loop — pandas 3.0 groupby.apply
    # drops the grouping column, so we concat per-genre samples instead.
    parts = [d.sample(min(len(d), args.per_genre), random_state=SEED)
             for _, d in songs.groupby("genre_sub")]
    return pd.concat(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="run all 903 songs")
    g.add_argument("--per-genre", type=int, default=3, help="stratified: songs per genre_sub")
    ap.add_argument("--songs", nargs="+", help="explicit song_id list")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    songs = pd.read_parquet(ROOT / "manifests" / "songs.parquet")
    stems = pd.read_parquet(ROOT / "manifests" / "stems.parquet")
    stem_paths = stems.groupby("song_id")["stem_path"].apply(list).to_dict()

    sel = select_songs(songs, args).reset_index(drop=True)
    default = "residuals" if (args.all or args.songs) else "residuals_sample"
    out = Path(args.out) if args.out else ROOT / "data" / "eda" / default
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"residual test on {len(sel)} songs -> {out}.parquet")

    rows = []
    for i, r in sel.iterrows():
        rec = dict(song_id=r.song_id, genre_sub=r.genre_sub, is_pansori=bool(r.is_pansori),
                   split=r.split, master_sr=int(r.master_sr))
        try:
            rec.update(residual_report([ROOT / p for p in stem_paths[r.song_id]],
                                       ROOT / r.master_path))
        except Exception as e:  # noqa: BLE001 — record, don't abort the batch
            rec["error"] = f"{type(e).__name__}: {e}"
        rows.append(rec)
        if (i + 1) % 10 == 0 or i + 1 == len(sel):
            print(f"  {i + 1}/{len(sel)}")

    df = pd.DataFrame(rows)
    df.to_parquet(f"{out}.parquet", index=False)
    df.to_csv(f"{out}.csv", index=False)

    ok = df[df.get("resid_db_gain").notna()] if "resid_db_gain" in df else df.iloc[0:0]
    print(f"\ndone. {len(ok)}/{len(df)} succeeded"
          + (f", {df['error'].notna().sum()} errored" if "error" in df else ""))
    if len(ok):
        print("\nresidual (gain-compensated, dB rel. master) by genre — lower = more linear:")
        print(ok.groupby("genre_sub")["resid_db_gain"]
                .agg(["count", "median", "min", "max"]).round(2).to_string())


if __name__ == "__main__":
    main()
