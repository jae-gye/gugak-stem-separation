#!/usr/bin/env python3
"""Full val-set (91) zero-shot inference — the deliverable run.

  htdemucs (single)   -> native `demucs`   -> drums/bass/other/vocals
  4-stem bs_roformer  -> MSST inference     -> drums/bass/other/vocals

Builds Σstems mixtures (trim-to-shortest, peak-norm 0.99), runs both models, saves all
predictions under data/_val/. Metric + wandb are a SEPARATE attended step afterwards.
Long job -> run in tmux:  CUDA_VISIBLE_DEVICES=0 uv run python scripts/infer_val.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.audio import peak, read_and_sum  # noqa: E402

MSST = ROOT / "external" / "msst"
CKPT = ROOT / "checkpoints" / "model_bs_roformer_4stem_ep17_sdr9.66.ckpt"
CFG = ROOT / "checkpoints" / "config_bs_roformer_4stem.yaml"
VAL = ROOT / "data" / "_val"
MIX = VAL / "mixes"
TMP_D = VAL / "_tmp_demucs"
TMP_B = VAL / "_tmp_bsrof"
TARGET_PEAK = 0.99


def sh(cmd: list) -> None:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> None:
    t0 = time.time()
    songs = pd.read_parquet(ROOT / "manifests" / "songs.parquet")
    stems = pd.read_parquet(ROOT / "manifests" / "stems.parquet")
    val = songs[songs.split == "val"].song_id.tolist()
    sp = stems.groupby("song_id")["stem_path"].apply(list).to_dict()
    for c in (CKPT, CFG):
        assert c.exists(), f"missing: {c}"

    # 1. mixtures (Σstems -> trim-to-shortest -> peak-norm 0.99)
    print(f"[1/3] building {len(val)} val mixtures -> {MIX.relative_to(ROOT)}", flush=True)
    MIX.mkdir(parents=True, exist_ok=True)
    for i, sid in enumerate(val, 1):
        out = MIX / f"{sid}_mix.wav"
        if not out.exists():
            mix, sr, _ = read_and_sum([ROOT / p for p in sp[sid]])
            pk = peak(mix)
            if pk > 0:
                mix = mix * (TARGET_PEAK / pk)
            sf.write(str(out), mix, sr, subtype="FLOAT")
        if i % 10 == 0 or i == len(val):
            print(f"  mixtures {i}/{len(val)}  ({time.time() - t0:.0f}s)", flush=True)
    mix_files = [str(MIX / f"{sid}_mix.wav") for sid in val]

    # 2. htdemucs (native demucs)
    print(f"\n[2/3] htdemucs on {len(val)} mixes  ({time.time() - t0:.0f}s)", flush=True)
    if TMP_D.exists():
        shutil.rmtree(TMP_D)
    sh([sys.executable, "-m", "demucs", "-n", "htdemucs", "-d", "cuda", "--float32",
        "-o", TMP_D, *mix_files])
    for sid in val:
        dst = VAL / "pred_htdemucs" / sid
        dst.mkdir(parents=True, exist_ok=True)
        for f in (TMP_D / "htdemucs" / f"{sid}_mix").glob("*.wav"):
            shutil.move(str(f), dst / f.name)

    # 3. 4-stem bs_roformer (MSST)
    print(f"\n[3/3] 4-stem bs_roformer on {len(val)} mixes  ({time.time() - t0:.0f}s)", flush=True)
    if TMP_B.exists():
        shutil.rmtree(TMP_B)
    TMP_B.mkdir(parents=True)
    sh([sys.executable, str(MSST / "inference.py"),
        "--model_type", "bs_roformer", "--config_path", str(CFG),
        "--start_check_point", str(CKPT), "--input_folder", str(MIX),
        "--store_dir", str(TMP_B), "--device_ids", "0"])
    for sid in val:
        dst = VAL / "pred_bsroformer" / sid
        dst.mkdir(parents=True, exist_ok=True)
        for f in (TMP_B / f"{sid}_mix").glob("*.wav"):
            shutil.move(str(f), dst / f.name)

    shutil.rmtree(TMP_D, ignore_errors=True)
    shutil.rmtree(TMP_B, ignore_errors=True)
    print(f"\nDONE in {(time.time() - t0) / 60:.1f} min. "
          f"preds -> data/_val/pred_htdemucs|pred_bsroformer/<song>/", flush=True)


if __name__ == "__main__":
    main()
