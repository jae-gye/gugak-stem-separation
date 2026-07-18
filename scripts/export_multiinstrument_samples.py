#!/usr/bin/env python3
"""One-off: export the multi-instrument stem subset (same base appearing >=2x per song).

Scope = every stem belonging to a (song, instrument_base) group of size >= 2
(the base plus its digit copies, e.g. 피리 + 피리2 + 피리3). All 창작국악.

Produces two trees under ``data/`` (gitignored, server-local NVMe):

  multi-instrument-samples/<song>/<base>/<instrument>.wav   -> SYMLINK to the real stem
      Zero-copy organized view of the full ~43 GB / ~37 h subset. Browse on the server.

  multi-instrument-excerpts/<song>/<base>/<instrument>.flac -> group-aligned FLAC excerpt
      Small, portable, lossless bundle for sharing with gugak experts. For each group we
      pick ONE window where ALL copies are simultaneously active (max over starts of the
      MIN per-member RMS) and cut the SAME passage from every copy -- so 피리/피리2/피리3 are
      heard over identical bars: the ideal A/B for "same instrument, or different?".
  + multi-instrument-excerpts/index.csv  (song, base, instrument, start_s, dur_s, peak, rms)

Run: uv run python scripts/export_multiinstrument_samples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.audio import peak, prefix_energy, read_audio, rms, to_mono  # noqa: E402

EXCERPT_SEC = 20.0
HOP_SEC = 0.5
SYM_ROOT = ROOT / "data" / "multi-instrument-samples"
EXC_ROOT = ROOT / "data" / "multi-instrument-excerpts"


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    stems = pd.read_parquet(ROOT / "manifests" / "stems.parquet")
    grp_size = stems.groupby(["song_id", "instrument_base"])["instrument"].transform("size")
    multi = stems[grp_size >= 2].copy()
    groups = list(multi.groupby(["song_id", "instrument_base"], sort=True))
    print(f"multi-instrument scope: {len(multi)} stems, {len(groups)} groups, "
          f"{multi.song_id.nunique()} songs, {multi.dur_s.sum() / 60:.1f} min")

    # --- 1. symlink farm (instant, zero-copy) ---
    n_links = 0
    for _, r in multi.iterrows():
        real = (ROOT / r.stem_path).resolve()
        link = SYM_ROOT / r.song_id / r.instrument_base / f"{r.instrument}.wav"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(real)
        n_links += 1
    print(f"[symlinks] wrote {n_links} links under {SYM_ROOT.relative_to(ROOT)}/")

    # --- 2. group-aligned FLAC excerpts ---
    index_rows = []
    for gi, ((song, base), grp) in enumerate(groups, 1):
        members = grp.sort_values("instrument")
        audios, prefixes, srs = [], [], []
        for _, r in members.iterrows():
            audio, sr = read_audio(ROOT / r.stem_path)
            audios.append(audio)
            srs.append(sr)
            prefixes.append(prefix_energy(to_mono(audio)))
        sr = srs[0]
        length = min(len(a) for a in audios)
        win = min(int(EXCERPT_SEC * sr), length)
        hop = max(1, int(HOP_SEC * sr))
        if length > win:
            starts = np.arange(0, length - win + 1, hop)
            # per-member window RMS, then the window maximizing the weakest member (all-active)
            member_rms = np.stack([np.sqrt((p[starts + win] - p[starts]) / win) for p in prefixes])
            best = int(starts[int(member_rms.min(axis=0).argmax())])
        else:
            best = 0
        for (_, r), audio in zip(members.iterrows(), audios):
            exc = audio[best:best + win]
            out = EXC_ROOT / song / base / f"{r.instrument}.flac"
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out), exc, sr, subtype="PCM_24", format="FLAC")
            index_rows.append(dict(song_id=song, base=base, instrument=r.instrument,
                                   start_s=round(best / sr, 2), dur_s=round(win / sr, 2),
                                   sr=sr, peak=round(peak(exc), 4), rms=round(rms(exc), 5)))
        if gi % 25 == 0 or gi == len(groups):
            print(f"[excerpts] {gi}/{len(groups)} groups")

    idx = pd.DataFrame(index_rows)
    idx.to_csv(EXC_ROOT / "index.csv", index=False)
    size_gb = dir_size_bytes(EXC_ROOT) / 1e9
    print(f"[excerpts] wrote {len(idx)} FLAC excerpts ({EXCERPT_SEC:.0f}s each), "
          f"{size_gb:.2f} GB under {EXC_ROOT.relative_to(ROOT)}/")
    n_quiet = (idx.peak < 0.02).sum()
    if n_quiet:
        print(f"[excerpts] note: {n_quiet} excerpt(s) are quiet (peak<0.02) — that member "
              f"is barely active even in the group's most-active window")


if __name__ == "__main__":
    main()
