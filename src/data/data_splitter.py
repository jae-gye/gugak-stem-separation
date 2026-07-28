"""data_splitter.py — freeze the 71955 (ensemble) eval split: song-level, genre-stratified.

Only the 903 71955 songs are split. 71470 (solo) is a train-only augmentation pool and
never enters val/test. Training data is generated augmented mixes, so "train" here just
means "the 71955 pool remainder" — the point of this manifest is to carve out the frozen
val + test song lists.

Principles (see Notion · Dataset 1 / Training Data Strategy):
  - split at SONG level  -> a song's master + all stems share one split (no audio leakage)
  - stratify by genreSub -> every genre appears in every split in proportion (tail genres
    like 대풍류 (21 songs) would otherwise risk vanishing from val/test)
  - seed-pinned + frozen -> reproducible; committed to manifests/ as the source of truth

Genre comes from metadata.csv (source of truth), joined to the on-disk song dirs by
NFC-normalized name. Output: manifests/parquet/eval_manifest.parquet +
manifests/csv/eval_manifest.csv.

Run:  uv run python src/data/data_splitter.py [--val-frac 0.10 --test-frac 0.15 --seed 42]
"""
from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# --- split parameters (override on the CLI) ---
SEED = 42
VAL_FRAC = 0.10          # ~10% — runs every N steps in training, kept lean
TEST_FRAC = 0.15         # ~15% — headline/paper number, computed rarely, robust tail
STRAT_COL = "genre_sub"

ENSEMBLE_SUBDIR = "data/gugak_ensemble_71955"
METADATA_CSV = "metadata.csv"
SOURCE_DIR = "source"
OUT_BASENAME = "manifests/eval_manifest"


def nfc(s: str) -> str:
    """NFC-normalize a Korean string (disk names are NFD-ish; CSV is NFC — must match)."""
    return unicodedata.normalize("NFC", str(s))


def find_root(start: Path | None = None) -> Path:
    """Locate the repo root (holds pyproject.toml), so the script is cwd-agnostic."""
    p = Path.cwd() if start is None else start
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists():
            return cand
    raise FileNotFoundError("repo root (pyproject.toml) not found above cwd")


def table_paths(base: Path) -> tuple[Path, Path]:
    """Map a manifests/<name> basename to its (parquet, csv) twin paths.

    Layout rule (repo-wide): parquet = source of truth in manifests/parquet/,
    csv = eyeball copy in manifests/csv/. Both dirs are created on demand.

    Args:
        base: table basename, e.g. Path(".../manifests/eval_manifest").
    """
    parquet = base.parent / "parquet" / f"{base.name}.parquet"
    csv = base.parent / "csv" / f"{base.name}.csv"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    return parquet, csv


def load_songs(root: Path) -> pd.DataFrame:
    """Return the 903 on-disk 71955 songs joined to their genre from metadata.csv."""
    ens = root / ENSEMBLE_SUBDIR
    meta = pd.read_csv(ens / METADATA_CSV)
    meta["key"] = meta["project_name"].map(nfc)
    genre = meta.set_index("key")[["genreMajor", "genreSub"]]

    song_dirs = sorted(p.name for p in (ens / SOURCE_DIR).iterdir() if p.is_dir())
    rows = []
    missing = []
    for name in song_dirs:
        key = nfc(name)
        if key not in genre.index:
            missing.append(name)
            continue
        g = genre.loc[key]
        rows.append({
            "song_id": name,
            "num_id": name.split("_", 1)[0],
            "genre_major": g["genreMajor"],
            "genre_sub": g["genreSub"],
        })
    if missing:
        raise ValueError(f"{len(missing)} song dirs not found in metadata.csv: {missing[:5]}")
    return pd.DataFrame(rows)


def stratified_split(df: pd.DataFrame, val_frac: float, test_frac: float,
                     seed: int, strat_col: str = STRAT_COL) -> pd.Series:
    """Assign each song 'train'|'val'|'test', stratified within each `strat_col` group.

    A single seeded RNG shuffles each group (groups visited in sorted order → fully
    deterministic). Each group reserves >=1 song for val and test when it has enough songs,
    so tail genres are never absent from an eval split.
    """
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index, dtype=object)
    for group in sorted(df[strat_col].unique()):
        idx = df.index[df[strat_col] == group].to_numpy()
        idx = idx[rng.permutation(len(idx))]          # seeded per-group shuffle
        n = len(idx)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        n_test = min(n_test, n - 2)                   # guarantee train + val remain
        n_val = min(n_val, n - n_test - 1)
        split.loc[idx[:n_test]] = "test"
        split.loc[idx[n_test:n_test + n_val]] = "val"
    return split


def build(root: Path, val_frac: float, test_frac: float, seed: int) -> pd.DataFrame:
    songs = load_songs(root)
    songs["split"] = stratified_split(songs, val_frac, test_frac, seed)
    return songs.sort_values(["genre_sub", "split", "song_id"]).reset_index(drop=True)


def report(df: pd.DataFrame) -> None:
    """Print the per-genre × split crosstab and totals for a sanity check."""
    ct = pd.crosstab(df["genre_sub"], df["split"], margins=True, margins_name="total")
    ct = ct[[c for c in ["train", "val", "test", "total"] if c in ct.columns]]
    print(ct.to_string())
    tot = df["split"].value_counts()
    print(f"\ntotals: train {tot.get('train', 0)} · val {tot.get('val', 0)} "
          f"· test {tot.get('test', 0)}  (of {len(df)})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze the 71955 genre-stratified eval split.")
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    ap.add_argument("--test-frac", type=float, default=TEST_FRAC)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None, help=f"output basename (default {OUT_BASENAME})")
    args = ap.parse_args()

    root = find_root()
    df = build(root, args.val_frac, args.test_frac, args.seed)

    out_parquet, out_csv = table_paths(root / (args.out or OUT_BASENAME))
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"seed={args.seed}  val_frac={args.val_frac}  test_frac={args.test_frac}")
    report(df)
    print(f"\nwrote {out_parquet} + {out_csv}")


if __name__ == "__main__":
    main()
