#!/usr/bin/env python3
"""Zero-shot baseline metric on the full val set (91) for htdemucs + 4-stem bs_roformer.

Two honest measures given the Western↔gugak head mismatch:
  - per-stem ENERGY distribution (% of predicted energy in each Western head) — shows where
    each model dumps gugak content;
  - 타악↔drums SI-SDR — the one loosely-comparable number (predicted 'drums' vs GT 타악 = Σ타악 stems).
Full per-stem SDR isn't meaningful cross-domain; the real per-stem SDR comes after fine-tuning.
Saves data/_val/metrics.parquet + prints a per-genre summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.audio import read_and_sum, read_audio, rms, to_mono  # noqa: E402

MODELS = ["htdemucs", "bsroformer"]
STEMS = ["drums", "bass", "other", "vocals"]
PRED_SR = 44100


def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-9) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = float((est * ref).sum() / ((ref * ref).sum() + eps))
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10(((target ** 2).sum() + eps) / ((noise ** 2).sum() + eps)))


def main() -> None:
    songs = pd.read_parquet(ROOT / "manifests" / "songs.parquet")
    stems = pd.read_parquet(ROOT / "manifests" / "stems.parquet")
    val = songs[songs.split == "val"]
    tak = (stems[stems.stem_group_4 == "타악"].groupby("song_id")["stem_path"]
           .apply(list).to_dict())

    rows = []
    for _, r in val.iterrows():
        sid = r.song_id
        rec = {"song_id": sid, "genre": r.genre_sub}
        # ground-truth 타악 (Σ타악 stems) resampled to the prediction rate, once per song
        gt_tak = None
        if sid in tak:
            g, sr, _ = read_and_sum([ROOT / p for p in tak[sid]])
            gt_tak = librosa.resample(to_mono(g), orig_sr=sr, target_sr=PRED_SR)
        for model in MODELS:
            pdir = ROOT / "data" / "_val" / f"pred_{model}" / sid
            energy = {s: rms(read_audio(pdir / f"{s}.wav")[0]) ** 2
                      for s in STEMS if (pdir / f"{s}.wav").exists()}
            tot = sum(energy.values()) or 1.0
            for s in STEMS:
                rec[f"{model}_{s}_pct"] = 100 * energy.get(s, 0.0) / tot
            if gt_tak is not None and (pdir / "drums.wav").exists():
                dr = to_mono(read_audio(pdir / "drums.wav")[0])
                n = min(len(dr), len(gt_tak))
                rec[f"{model}_tak_drums_sisdr"] = si_sdr(dr[:n], gt_tak[:n])
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_parquet(ROOT / "data" / "_val" / "metrics.parquet", index=False)

    pd.set_option("display.width", 200)
    print(f"\n=== ENERGY distribution — mean % per genre (n={len(df)} val songs) ===")
    for model in MODELS:
        cols = [f"{model}_{s}_pct" for s in STEMS]
        g = df.groupby("genre")[cols].mean().round(0)
        g.columns = STEMS
        g["n"] = df.groupby("genre").size()
        print(f"\n[{model}]"); print(g.to_string())
    print("\n=== 타악↔drums SI-SDR (dB) — higher = better; per genre median ===")
    sicols = [f"{m}_tak_drums_sisdr" for m in MODELS]
    have = df[df[sicols].notna().any(axis=1)]
    s = have.groupby("genre")[sicols].median().round(1)
    s.columns = MODELS
    s["n"] = have.groupby("genre").size()
    print(s.to_string())
    print("\noverall median 타악↔drums SI-SDR:",
          {m: round(float(df[f"{m}_tak_drums_sisdr"].median()), 1) for m in MODELS})


if __name__ == "__main__":
    main()
