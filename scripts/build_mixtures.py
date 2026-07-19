#!/usr/bin/env python3
"""Build Σ(stems) mixtures for the pinned listening set (trim-to-shortest per song).

The mixture is our canonical model input: mix := Σ stems (targets := stems). Reuses
`src/data/audio.read_and_sum`, which trims all stems to the shortest per song (handles
the 판소리 length case). Writes native-SR float32:
  data/listening_set/<song>/<song>_mix.wav
Σstems peaks well above 1.0 (parts stack), so we peak-normalize the MIXTURE to 0.99
(never per-stem — that would wreck the balance). float32, native SR.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.audio import peak, read_and_sum  # noqa: E402

CONFIG = ROOT / "configs" / "listening_set.yaml"
OUT = ROOT / "data" / "listening_set"
TARGET_PEAK = 0.99


def main() -> None:
    song_ids = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["listening_set"]
    stems = pd.read_parquet(ROOT / "manifests" / "stems.parquet")
    sp = stems.groupby("song_id")["stem_path"].apply(list).to_dict()

    for sid in song_ids:
        mix, sr, lens = read_and_sum([ROOT / p for p in sp[sid]])
        raw_peak = peak(mix)
        if raw_peak > 0:
            mix = mix * (TARGET_PEAK / raw_peak)          # peak-normalize the MIXTURE, never per-stem
        out = OUT / sid / f"{sid}_mix.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), mix, sr, subtype="FLOAT")
        print(f"{sid}: {len(sp[sid]):2d} stems -> {len(mix) / sr:5.1f}s @ {sr} "
              f"| raw peak {raw_peak:.3f} -> {peak(mix):.3f} | trimmed {max(lens) - min(lens)} frames")


if __name__ == "__main__":
    main()
