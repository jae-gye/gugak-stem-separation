"""stem_duplicate_scan.py — find stems that are duplicates of each other or of the mix.

Detects two publisher-side data defects that the manifest cannot see, both confirmed in
71955 on 2026-07-30 (→ docs/solo_pool_duration_analysis.md §6):

  1. `0714_민속악_민요` ships the MIXDOWN three times in place of its 대금/아쟁/피리 stems.
  2. `0885_창작국악_창작국악` is `0886_창작국악_창작국악` repeated exactly 7 times.

Why this needs audio at all: neither defect survives a manifest check. Mastering changes the
crest factor, so a mix-copy's `peak`/`rms` do not match its master's, and every
gain-invariant descriptor (`rms/peak`) breaks too. File-level md5 also fails — the 0714
copies differ in a wav metadata chunk while the audio payload is byte-identical.

Two tests per song:
  - EXACT IDENTITY: md5 of the DECODED excerpt samples (ignores header/metadata chunks).
  - MIX-COPY: least-squares gain of the master onto each stem, plus the fraction of master
    energy that explains. Gain-invariant, so a fader difference cannot hide a copy.

⚠️ TWO TRAPS, both hit during development — do not remove these guards:
  - **Silent excerpts hash alike.** Sparse stems (박 etc. play rarely BY DESIGN) produce
    constant-valued excerpts: exactly zero, or a ±1-LSB residue (1.192e-07 = 2^-23) left by
    ingest's DC removal. Different residues land in different hash buckets, so silence
    masqueraded as two large duplicate "groups" of 35 and 36 unrelated files. Every hash
    group is therefore checked for `n_unique > 1` before it is believed.
  - **Correlation alone does not identify a mix-copy.** A 2-stem 산조 song is 해금 plus 장구,
    so its 해금 legitimately explains 83-96% of the master. Energy-explained RANKS
    candidates; only exact identity CONFIRMS one.

Known limitation: a song that is an N-times loop of ITSELF, with no shorter twin to hash
against, passes both tests. 0885 was only caught because 0886 existed.

Reads the ingest store (uniformly 44.1 kHz, so nothing needs resampling mid-comparison) and
never walks directories — the manifest is the file list.

Run:
    uv run python scripts/stem_duplicate_scan.py [options]
      --excerpt-seconds S    comparison window per file (default: 30)
      --dataset {71955,71470,both}
      --out BASE             output basename (default: manifests/stem_duplicate_scan)
      --top N                how many ranked mix-copy candidates to print (default: 25)
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

DEFAULT_SOURCE_MANIFEST = "manifests/parquet/source_manifest.parquet"
DEFAULT_OUT = "manifests/stem_duplicate_scan"
DEFAULT_EXCERPT_SECONDS = 30.0
DEFAULT_DECIMATE = 4
DEFAULT_TOP = 25


# --- repo-root discovery + table paths (mirrors src/data/ingest.py) ---
def find_root(start: Path | None = None) -> Path:
    """Locate the repo root (holds pyproject.toml)."""
    p = Path.cwd() if start is None else start
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists():
            return cand
    raise FileNotFoundError("repo root (pyproject.toml) not found above cwd")


def table_paths(base: Path) -> tuple[Path, Path]:
    """Map a manifests/<name> basename to its (parquet, csv) twin paths."""
    parquet = base.parent / "parquet" / f"{base.name}.parquet"
    csv = base.parent / "csv" / f"{base.name}.csv"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    return parquet, csv


# --- excerpt reading + fingerprinting ---
def read_excerpt(path: Path, start_frame: int, frames: int, decimate: int) -> np.ndarray:
    """Channel 0 of one excerpt, decimated, as float64 for stable dot products.

    Args:
        path: audio file to read.
        start_frame: first frame of the window.
        frames: window length in frames.
        decimate: keep every Nth sample (correlation survives decimation).
    """
    audio, _ = sf.read(str(path), start=start_frame, frames=frames,
                       dtype="float32", always_2d=True)
    return audio[::decimate, 0].astype(np.float64)


def excerpt_fingerprint(signal: np.ndarray) -> str:
    """md5 of the decoded samples — exact audio identity, blind to wav metadata chunks."""
    return hashlib.md5(np.ascontiguousarray(signal).tobytes()).hexdigest()


def carries_signal(signal: np.ndarray) -> bool:
    """True when an excerpt is not constant-valued.

    THE SILENCE GUARD. A constant excerpt is digital silence (or a ±1-LSB DC-removal
    residue) and hashes identically across completely unrelated files. Without this check
    a scan reports dozens of phantom duplicates.
    """
    return len(np.unique(signal)) > 1


def explained_fraction(target: np.ndarray,
                       predictor: np.ndarray) -> tuple[float, float, float]:
    """(correlation, least-squares gain, fraction of target energy explained).

    A pure gain cancels out, so a stem copied from the master at a different fader level
    still scores near 1.0.

    Args:
        target: the master excerpt.
        predictor: the candidate stem excerpt.
    """
    predictor_energy = float(np.dot(predictor, predictor))
    target_energy = float(np.dot(target, target))
    if predictor_energy <= 0.0 or target_energy <= 0.0:
        return 0.0, 0.0, 0.0
    gain = float(np.dot(target, predictor)) / predictor_energy
    residual = target - gain * predictor
    fraction = 1.0 - float(np.dot(residual, residual)) / target_energy
    correlation = float(np.dot(target, predictor)) / float(
        np.linalg.norm(target) * np.linalg.norm(predictor))
    return correlation, gain, fraction


# --- the per-song scan ---
def scan_song(song_id: str, group: pd.DataFrame, excerpt_seconds: float,
              decimate: int) -> tuple[list[dict], list[dict]]:
    """Fingerprint every file of one song and score its stems against its master.

    Args:
        song_id: the song being scanned.
        group: manifest rows for that song (master + stems).
        excerpt_seconds: comparison window length.
        decimate: excerpt decimation factor.

    Returns:
        (mix_copy_rows, fingerprint_rows) — empty lists when the song has no master.
    """
    master_rows = group[group["role"] == "master"]
    stem_rows = group[group["role"] == "stem"]
    if master_rows.empty or stem_rows.empty:
        return [], []

    # a common window: stems can outrun the master (판소리) -> clamp to the shortest file
    shortest = int(group["out_frames"].min())
    sample_rate = int(group["out_sr"].iloc[0])
    frames = min(int(excerpt_seconds * sample_rate), shortest)
    start = max(0, shortest // 2 - frames // 2)

    master = read_excerpt(Path(master_rows["out_path"].iloc[0]), start, frames, decimate)
    fingerprints = [{"song_id": song_id, "file_id": master_rows["file_id"].iloc[0],
                     "role": "master", "instrument": None,
                     "excerpt_md5": excerpt_fingerprint(master),
                     "carries_signal": carries_signal(master)}]

    mix_copies = []
    for stem in stem_rows.itertuples():
        signal = read_excerpt(Path(stem.out_path), start, frames, decimate)
        correlation, gain, fraction = explained_fraction(master, signal)
        mix_copies.append({
            "song_id": song_id, "file_id": stem.file_id,
            "instrument": stem.instrument_canonical, "stem_class": stem.stem_group,
            "genre_sub": stem.genre_sub, "split": stem.split,
            "num_stems_in_song": int(len(stem_rows)),
            "corr_with_master": correlation, "gain_master_over_stem": gain,
            "frac_master_energy_explained": fraction,
        })
        fingerprints.append({"song_id": song_id, "file_id": stem.file_id, "role": "stem",
                             "instrument": stem.instrument_canonical,
                             "excerpt_md5": excerpt_fingerprint(signal),
                             "carries_signal": carries_signal(signal)})
    return mix_copies, fingerprints


# --- reporting ---
def report_duplicate_groups(fingerprints: pd.DataFrame) -> pd.DataFrame:
    """Print and return duplicate groups, separating real audio from silent excerpts."""
    duplicated = fingerprints[fingerprints.duplicated("excerpt_md5", keep=False)]
    if duplicated.empty:
        print("  no duplicate excerpts at all")
        return duplicated

    silent_files = int((~duplicated["carries_signal"]).sum())
    real = duplicated[duplicated["carries_signal"]]
    print(f"  {len(duplicated)} files share a hash with another; {silent_files} of those are "
          f"SILENT excerpts (discarded — sparse stems, not duplicates)")
    print(f"  {len(real)} files in {real['excerpt_md5'].nunique()} groups carry real signal:")
    for _, group in real.groupby("excerpt_md5"):
        members = ", ".join(f"{r.song_id}/{r.instrument or 'MASTER'}"
                            for r in group.itertuples())
        print(f"    [{len(group)}] {members}")
    return real


def report_mix_copies(scan: pd.DataFrame, top: int) -> None:
    """Print the ranked mix-copy candidates, loudest signal first."""
    print(f"\n--- top {top} stems by master energy explained ---")
    print("(ranking only — a 2-stem 산조 해금 legitimately explains 83-96%; "
          "confirm with exact identity)")
    columns = ["song_id", "instrument", "num_stems_in_song", "split",
               "corr_with_master", "gain_master_over_stem", "frac_master_energy_explained"]
    print(scan.nlargest(top, "frac_master_energy_explained")[columns].round(4)
          .to_string(index=False))
    print("\n--- distribution of frac_master_energy_explained ---")
    print(scan["frac_master_energy_explained"]
          .quantile([0.5, 0.9, 0.99, 0.999, 1.0]).round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find stems that duplicate each other or the mixdown.")
    ap.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST)
    ap.add_argument("--dataset", choices=["71955", "71470", "both"], default="71955",
                    help="only 71955 has masters, so the mix-copy test needs it")
    ap.add_argument("--excerpt-seconds", type=float, default=DEFAULT_EXCERPT_SECONDS)
    ap.add_argument("--decimate", type=int, default=DEFAULT_DECIMATE)
    ap.add_argument("--out", default=DEFAULT_OUT, help="output basename under manifests/")
    ap.add_argument("--top", type=int, default=DEFAULT_TOP)
    args = ap.parse_args()

    # load the worklist (manifest is the file list — never walk directories)
    root = find_root()
    manifest = pd.read_parquet(root / args.source_manifest)
    manifest["dataset"] = manifest["dataset"].astype(str)
    wanted = {"71955", "71470"} if args.dataset == "both" else {args.dataset}
    manifest = manifest[manifest["dataset"].isin(wanted)]

    songs = list(manifest.groupby("song_id"))
    print(f"scanning {len(songs)} songs ({len(manifest)} files), "
          f"{args.excerpt_seconds:g} s excerpt each")

    # scan
    mix_copy_rows: list[dict] = []
    fingerprint_rows: list[dict] = []
    for index, (song_id, group) in enumerate(songs, 1):
        copies, fingerprints = scan_song(song_id, group, args.excerpt_seconds, args.decimate)
        mix_copy_rows.extend(copies)
        fingerprint_rows.extend(fingerprints)
        if index % 150 == 0:
            print(f"  {index}/{len(songs)}")

    scan = pd.DataFrame(mix_copy_rows)
    fingerprints = pd.DataFrame(fingerprint_rows)

    # persist both tables (parquet canonical + csv twin)
    scan_parquet, scan_csv = table_paths(root / args.out)
    scan.to_parquet(scan_parquet, index=False)
    scan.to_csv(scan_csv, index=False)
    hash_parquet, hash_csv = table_paths(root / f"{args.out}_fingerprints")
    fingerprints.to_parquet(hash_parquet, index=False)
    fingerprints.to_csv(hash_csv, index=False)

    # report
    print(f"\n=== scanned {len(scan)} stems across {scan['song_id'].nunique()} songs ===")
    print("\n--- exact-identity duplicate groups ---")
    report_duplicate_groups(fingerprints)
    report_mix_copies(scan, args.top)
    print(f"\nwrote {scan_parquet} and {hash_parquet}")


if __name__ == "__main__":
    main()
