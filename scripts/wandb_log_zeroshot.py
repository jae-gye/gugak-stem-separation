#!/usr/bin/env python3
"""Log the zero-shot baseline to wandb (run attended — needs `wandb login`).

Creates project `gugak-stem-separation`, run `zeroshot-baseline`, logging:
  - summary: overall 타악↔drums SI-SDR median per model
  - per-song metrics table (from data/_val/metrics.parquet)
  - per-genre 타악↔drums SI-SDR bar charts
  - per-genre energy-distribution tables
  - 25s mono audio excerpts (mix + both models' 4 stems) for the 5 pinned songs
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import wandb
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.audio import loudest_window, prefix_energy, read_audio, to_mono  # noqa: E402

STEMS = ["drums", "bass", "other", "vocals"]
MODELS = ["htdemucs", "bsroformer"]
PROJECT = "gugak-stem-separation"


def excerpt_mono(path: Path, secs: float = 25.0):
    a, sr = read_audio(path)
    m = to_mono(a).astype("float32")
    win = min(int(secs * sr), len(m))
    s = loudest_window(prefix_energy(m), win, max(1, int(0.1 * sr)))
    return m[s:s + win], sr


def main() -> None:
    metrics = pd.read_parquet(ROOT / "data" / "_val" / "metrics.parquet")
    pinned = yaml.safe_load((ROOT / "configs" / "listening_set.yaml").read_text("utf-8"))["listening_set"]
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())

    run = wandb.init(
        project=PROJECT, name="zeroshot-baseline", job_type="eval",
        config=dict(git=git, code_dirty=dirty, n_val=int(len(metrics)),
                    htdemucs="htdemucs (native demucs, 4-stem)",
                    bsroformer="bs_roformer 4-stem MUSDB ep17_sdr9.66",
                    framing="zero-shot OOD floor — Western heads (drums/bass/other/vocals) != gugak stems"))
    try:
        for m in MODELS:
            run.summary[f"{m}/tak_drums_sisdr_median"] = float(metrics[f"{m}_tak_drums_sisdr"].median())

        run.log({"metrics/per_song": wandb.Table(dataframe=metrics.round(2))})

        si = metrics.groupby("genre")[[f"{m}_tak_drums_sisdr" for m in MODELS]].median().reset_index()
        for m in MODELS:
            t = wandb.Table(data=si[["genre", f"{m}_tak_drums_sisdr"]].round(2).values.tolist(),
                            columns=["genre", "sisdr_dB"])
            run.log({f"sisdr_by_genre/{m}": wandb.plot.bar(t, "genre", "sisdr_dB",
                     title=f"{m} · 타악↔drums SI-SDR (dB) by genre")})

        for m in MODELS:
            eg = metrics.groupby("genre")[[f"{m}_{s}_pct" for s in STEMS]].mean().round(1).reset_index()
            eg.columns = ["genre", *STEMS]
            run.log({f"energy/{m}_by_genre_pct": wandb.Table(dataframe=eg)})

        cols = ["song", "mixture"] + [f"{m}:{s}" for m in MODELS for s in STEMS]
        at = wandb.Table(columns=cols)
        for sid in pinned:
            mx, sr = excerpt_mono(ROOT / "data" / "_val" / "mixes" / f"{sid}_mix.wav")
            cells = [sid, wandb.Audio(mx, sample_rate=sr, caption="mixture")]
            for m in MODELS:
                for s in STEMS:
                    ex, esr = excerpt_mono(ROOT / "data" / "_val" / f"pred_{m}" / sid / f"{s}.wav")
                    cells.append(wandb.Audio(ex, sample_rate=esr, caption=f"{m}:{s}"))
            at.add_data(*cells)
            print(f"  audio excerpts done: {sid}", flush=True)
        run.log({"audio/pinned_listening_set": at})
    finally:
        run.finish()
    print("\nwandb run URL:", run.url)


if __name__ == "__main__":
    main()
