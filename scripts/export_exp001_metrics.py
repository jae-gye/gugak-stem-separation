"""export_exp001_metrics.py — freeze exp001's metric history into tracked tables.

Reads the metric history MSST persists inside a checkpoint (`all_metrics`: per epoch,
per stem class, one SI-SDR value per scored val song) and writes deck-ready tables:

  eval_trajectory   one row per eval cycle: avg SI-SDR + per-class means (headline chart)
  per_song_best     one row per (song, class) at the best checkpoint, with genre
  per_genre_best    per (genre, class) means at the best checkpoint (the hardness story)

Song identity recovery: MSST stores bare score lists whose order follows
`valid.py`'s `Path(valid_root).rglob("mixture.flac")` — filesystem order, NOT sorted.
This script reproduces that traversal and, per class, keeps the songs whose folder
actually holds `<class>.flac` (absent classes are skipped during eval). The mapping is
asserted against every class's score count, so a mismatch fails loudly instead of
silently mislabelling songs.

Run:
    uv run python scripts/export_exp001_metrics.py
      --checkpoint experiments/.../checkpoints/model_htdemucs_ep_33_si_sdr_1.0409.ckpt
      --valid-root data/gugak_ensemble_71955/sumstem_9stem/val
      --out-dir experiments/.../metrics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def find_root(start: Path | None = None) -> Path:
    """Locate the repo root (holds pyproject.toml)."""
    p = Path.cwd() if start is None else start
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists():
            return cand
    raise FileNotFoundError("repo root (pyproject.toml) not found above cwd")


def eval_song_order(valid_root: Path) -> list[str]:
    """Song ids in the exact order valid.py scored them (rglob traversal, unsorted)."""
    return [p.parent.name for p in valid_root.rglob("mixture.flac")]


def build_trajectory(all_metrics: dict, steps_per_epoch: int) -> pd.DataFrame:
    """One row per eval cycle: per-class means + the average that drove selection.

    Args:
        all_metrics: MSST's {epoch_key: {metric: {class: [per-song values]}}}.
        steps_per_epoch: optimizer steps between evals (for an x-axis in real units).
    """
    rows = []
    for key in sorted(all_metrics, key=lambda k: int(k.split("_")[1])):
        epoch = int(key.split("_")[1])
        per_class = all_metrics[key]["si_sdr"]
        means = {c: float(np.mean(v)) for c, v in per_class.items()}
        rows.append({"epoch": epoch,
                     "optimizer_steps": (epoch + 1) * steps_per_epoch,
                     "avg_si_sdr": float(np.mean(list(means.values()))),
                     **{f"si_sdr_{c}": m for c, m in means.items()}})
    return pd.DataFrame(rows)


def build_per_song(all_metrics: dict, epoch_key: str, song_order: list[str],
                   valid_root: Path, extension: str = "flac") -> pd.DataFrame:
    """Per-(song, class) scores at one epoch, with song identity recovered.

    Args:
        all_metrics: MSST metric history.
        epoch_key: which epoch's scores to unpack (e.g. "epoch_33").
        song_order: song ids in eval traversal order.
        valid_root: Σstem val tree (used to test which songs hold each class).
    """
    rows = []
    for stem_class, values in all_metrics[epoch_key]["si_sdr"].items():
        scored = [s for s in song_order
                  if (valid_root / s / f"{stem_class}.{extension}").exists()]
        if len(scored) != len(values):
            raise ValueError(
                f"{stem_class}: {len(values)} scores vs {len(scored)} songs holding the "
                "class — traversal order or val tree changed; mapping unsafe")
        rows.extend({"song_id": song, "stem_class": stem_class, "si_sdr": float(v)}
                    for song, v in zip(scored, values))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze exp001 metrics into tables.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--valid-root", default="data/gugak_ensemble_71955/sumstem_9stem/val")
    ap.add_argument("--out-dir", default="experiments/exp001_260728_htdemucs_9stem/metrics")
    ap.add_argument("--steps-per-epoch", type=int, default=2500,
                    help="optimizer steps per eval cycle (10,000 iters / accum 4)")
    ap.add_argument("--source-manifest",
                    default="manifests/parquet/source_manifest.parquet")
    args = ap.parse_args()

    root = find_root()
    valid_root = root / args.valid_root
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(root / args.checkpoint, map_location="cpu", weights_only=False)
    all_metrics = checkpoint["all_metrics"]
    best_key = max(all_metrics, key=lambda k: int(k.split("_")[1]))

    # --- trajectory ---
    trajectory = build_trajectory(all_metrics, args.steps_per_epoch)
    trajectory.to_parquet(out_dir / "eval_trajectory.parquet", index=False)
    trajectory.to_csv(out_dir / "eval_trajectory.csv", index=False)

    # --- per-song at the best epoch, joined to genre ---
    per_song = build_per_song(all_metrics, best_key, eval_song_order(valid_root), valid_root)
    manifest = pd.read_parquet(root / args.source_manifest)
    genres = (manifest[manifest.dataset == "71955"][["song_id", "genre_sub"]]
              .drop_duplicates().set_index("song_id")["genre_sub"])
    per_song["genre_sub"] = per_song.song_id.map(genres)
    if per_song.genre_sub.isna().any():
        raise ValueError("some songs did not join to a genre — check song_id keys")
    per_song.to_parquet(out_dir / "per_song_best.parquet", index=False)
    per_song.to_csv(out_dir / "per_song_best.csv", index=False)

    # --- per (genre, class) at the best epoch ---
    per_genre = (per_song.groupby(["genre_sub", "stem_class"])
                 .agg(si_sdr_mean=("si_sdr", "mean"), si_sdr_median=("si_sdr", "median"),
                      songs=("song_id", "nunique")).reset_index())
    per_genre.to_parquet(out_dir / "per_genre_best.parquet", index=False)
    per_genre.to_csv(out_dir / "per_genre_best.csv", index=False)

    print(f"best epoch: {best_key} · avg SI-SDR "
          f"{trajectory.avg_si_sdr.iloc[-1]:+.4f}")
    print(f"eval cycles: {len(trajectory)} · per-song rows: {len(per_song)} · "
          f"genres × classes: {len(per_genre)}")
    print(f"wrote -> {out_dir}")


if __name__ == "__main__":
    main()
