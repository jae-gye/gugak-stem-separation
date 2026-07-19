#!/usr/bin/env python3
"""Zero-shot smoke inference on the pinned listening set (local only, no wandb).

  htdemucs (single)      -> native `demucs`  -> drums/bass/other/vocals
  viperx bs_roformer     -> MSST inference   -> vocals/other

Both are Western pretrained models run out-of-domain on gugak mixtures — a pipeline
validation + qualitative floor, NOT a comparable score. Outputs land next to the stems:
  data/listening_set/<song>/pred_htdemucs/*.wav
  data/listening_set/<song>/pred_bsroformer/*.wav
Run 1-GPU: CUDA_VISIBLE_DEVICES=0 uv run python scripts/zeroshot_infer.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd  # noqa: F401  (kept for parity / future metric hooks)
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "listening_set.yaml"
LS = ROOT / "data" / "listening_set"
MSST = ROOT / "external" / "msst"
CKPT = ROOT / "checkpoints" / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
BSROF_CFG = MSST / "configs" / "viperx" / "model_bs_roformer_ep_317_sdr_12.9755.yaml"
MIX_IN = ROOT / "data" / "_smoke_mixes"
TMP_DEMUCS = ROOT / "data" / "_tmp_demucs"
TMP_BSROF = ROOT / "data" / "_tmp_bsrof"


def sh(cmd: list) -> None:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> None:
    song_ids = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["listening_set"]
    mixes = {sid: LS / sid / f"{sid}_mix.wav" for sid in song_ids}
    for sid, m in mixes.items():
        assert m.exists(), f"missing mix: {m} (run scripts/build_mixtures.py)"
    assert CKPT.exists(), f"missing bs_roformer ckpt: {CKPT}"

    # flat input folder of the 5 mixes for MSST's --input_folder
    if MIX_IN.exists():
        shutil.rmtree(MIX_IN)
    MIX_IN.mkdir(parents=True)
    for sid, m in mixes.items():
        (MIX_IN / f"{sid}_mix.wav").symlink_to(m.resolve())

    # --- htdemucs (native demucs) ---
    print("\n=== htdemucs (native demucs) ===", flush=True)
    if TMP_DEMUCS.exists():
        shutil.rmtree(TMP_DEMUCS)
    sh([sys.executable, "-m", "demucs", "-n", "htdemucs", "-d", "cuda", "--float32",
        "-o", TMP_DEMUCS, *[str(m) for m in mixes.values()]])
    for sid in song_ids:
        dst = LS / sid / "pred_htdemucs"
        dst.mkdir(parents=True, exist_ok=True)
        for f in (TMP_DEMUCS / "htdemucs" / f"{sid}_mix").glob("*.wav"):
            shutil.copy(f, dst / f.name)

    # --- viperx bs_roformer (MSST) ---
    print("\n=== viperx bs_roformer (MSST) ===", flush=True)
    if TMP_BSROF.exists():
        shutil.rmtree(TMP_BSROF)
    TMP_BSROF.mkdir(parents=True)
    # run inference.py from repo root: its own dir auto-joins sys.path so `utils`/`models` resolve
    sh([sys.executable, str(MSST / "inference.py"),
        "--model_type", "bs_roformer",
        "--config_path", str(BSROF_CFG),
        "--start_check_point", str(CKPT),
        "--input_folder", str(MIX_IN),
        "--store_dir", str(TMP_BSROF),
        "--extract_instrumental",       # also emit the complement (mix - vocals)
        "--device_ids", "0"])
    for sid in song_ids:                # MSST writes per-track subfolders (like demucs)
        dst = LS / sid / "pred_bsroformer"
        dst.mkdir(parents=True, exist_ok=True)
        for f in (TMP_BSROF / f"{sid}_mix").glob("*.wav"):
            shutil.copy(f, dst / f.name)

    print("\nDONE -> data/listening_set/<song>/pred_htdemucs/ + pred_bsroformer/", flush=True)


if __name__ == "__main__":
    main()
