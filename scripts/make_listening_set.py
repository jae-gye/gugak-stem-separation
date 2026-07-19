#!/usr/bin/env python3
"""Symlink the pinned listening set — the fixed val songs we follow by ear across ALL
experiments (logged as wandb.Audio each eval). Zero-copy symlinks, regenerable.

The set is defined in configs/listening_set.yaml (single source of truth); IDs resolve
against the manifest, never by walking folders. Creates:
  data/listening_set/<song_id>/<file> -> real WAV (master + stems).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "listening_set.yaml"
OUT = ROOT / "data" / "listening_set"


def main() -> None:
    song_ids = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["listening_set"]
    songs = pd.read_parquet(ROOT / "manifests" / "songs.parquet").set_index("song_id")
    stems = pd.read_parquet(ROOT / "manifests" / "stems.parquet")
    sp = stems.groupby("song_id")["stem_path"].apply(list).to_dict()

    missing = [s for s in song_ids if s not in songs.index]
    if missing:
        raise SystemExit(f"listening_set IDs not in manifest: {missing}")

    n_links = 0
    for sid in song_ids:
        for rel in [songs.loc[sid, "master_path"], *sp[sid]]:
            real = (ROOT / rel).resolve()
            link = OUT / sid / Path(rel).name
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(real)
            n_links += 1
    print(f"listening_set: {len(song_ids)} songs, {n_links} symlinks -> {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
