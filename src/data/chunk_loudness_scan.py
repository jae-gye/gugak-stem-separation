"""chunk_loudness_scan.py — the loudness the dataloader ACTUALLY serves, per class per pool.

Loudness matching needs to know what an ensemble stem sounds like at the scale the model
sees it. The per-file scan (`loudness_crest_scan`) answers a different question: it measures
whole files, while training reads short activity-aware excerpts. The two differ, and they
differ DIFFERENTLY for the two pools, which is exactly the bias matching exists to remove:

  - windowing — a whole-file value averages a song part across its loud and quiet passages,
    while an activity-aware window deliberately lands where the stem is sounding
  - channel presentation — the file scan measures a mono file as one channel, but the
    dataloader duplicates mono to stereo, and BS.1770 sums channel energies, so the same
    audio reads exactly 10·log10(2) = 3.01 dB louder as the dataloader presents it. The
    solo pool is ~79% mono after ingest and the ensemble pool is not, so this lands almost
    entirely on one pool.

So this module measures windows, not files, and it measures them THROUGH THE DATALOADER'S
OWN DRAW PATH (`GugakMixDataset._draw_excerpt`) rather than reimplementing it: same window
length, same activity manifest, same activity criterion, same mono handling, same tail
padding for short clips. Reimplementing would let the reference drift from the thing it
describes — the one failure this table cannot survive.

Output is the full per-window distribution, one row per window, NOT summary statistics:
loudness matching samples empirically from it (per-class loudness is plausibly multimodal —
the same instrument leads one song and provides texture in another), so collapsing it to a
mean and a sigma would throw away the shape the sampler needs.

Regenerate whenever the thing it describes changes: a different window length (BS-RoFormer
trains at 8 s, HTDemucs at 10 s), a new activity criterion, a re-ingest, or a change to how
short clips are placed in the window (pad-with-random-placement will change these values).
The window length is written into the table and into its filename so the two can't be
confused, and the matching sampler refuses a table whose window length disagrees with the
configured segment.

READ-ONLY: reads the ingest store, the manifests and the per-file loudness scan; writes only
its own output table.

Run:
    uv run python src/data/chunk_loudness_scan.py
      --config PATH            experiment config supplying gugak_mix (classes, geometry,
                               manifest paths)     (default: configs/exp001_htdemucs_9stem.yaml)
      --window-seconds S       window length (default: the config's segment_seconds)
      --windows-per-class N    windows drawn per class per pool (default: 1000)
      --datasets D [D ...]     pools to scan (default: 71955 71470)
      --split SPLIT            manifest split to draw from (default: train)
      --seed N                 base seed; every window is independently reproducible
      --source-table PATH      per-file loudness scan joined in for the file-vs-window offset
      --tables-out DIR         output directory (default: experiments/260731_chunk_loudness)
      --out-name NAME          table basename (default: chunk_loudness_reference_<W>s)
      --workers N              parallel processes (default: 4 — the box is shared)
"""
from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyloudnorm
import yaml

try:    # imported as a package module
    from src.data.loudness_match import MONO_DUPLICATION_DB
    from src.data.mix_dataset import GugakMixDataset, MixDatasetConfig
except ModuleNotFoundError:   # imported as a sibling (scripts run from src/data)
    from loudness_match import MONO_DUPLICATION_DB
    from mix_dataset import GugakMixDataset, MixDatasetConfig

# --- defaults (CLI-overridable) ---
DEFAULT_CONFIG = "configs/exp001_htdemucs_9stem.yaml"
DEFAULT_SOURCE_TABLE = "experiments/260730_solo_pool_duration/loudness_crest_scan.parquet"
DEFAULT_TABLES_DIR = "experiments/260731_chunk_loudness"
DEFAULT_DATASETS = ("71955", "71470")
DEFAULT_SPLIT = "train"
DEFAULT_WINDOWS_PER_CLASS = 1000
DEFAULT_SEED = 20260731
DEFAULT_WORKERS = 4
# MONO_DUPLICATION_DB is owned by loudness_match (the runtime consumer), imported above,
# so the scan and the sampler can never disagree about how a mono file is presented


def repo_root(start: Path | None = None) -> Path:
    """Locate the repo root (holds pyproject.toml)."""
    p = Path.cwd() if start is None else start
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists():
            return cand
    raise FileNotFoundError("repo root (pyproject.toml) not found above cwd")


def table_paths(base: Path) -> tuple[Path, Path]:
    """Map an output basename to its (parquet, csv) twin paths, creating parents.

    Args:
        base: table basename without suffix.
    """
    parquet, csv = base.with_suffix(".parquet"), base.with_suffix(".csv")
    parquet.parent.mkdir(parents=True, exist_ok=True)
    return parquet, csv


# --- configuration ----------------------------------------------------------
@dataclass
class ChunkLoudnessScanConfig:
    """Everything the scan needs; assembled from CLI args."""
    experiment_config: Path
    window_seconds: float
    windows_per_class: int
    datasets: tuple[str, ...]
    split: str
    seed: int
    source_table: Path
    tables_dir: Path
    out_name: str
    workers: int


# --- per-process dataset cache (one build per pool per worker) --------------
_DATASET_CACHE: dict = {}


def pool_dataset(dataset_name: str, mix_block: dict, root: str) -> GugakMixDataset:
    """The dataloader itself, restricted to one source pool, matching left off.

    Cached per process: building it reads three manifests, and every window of a pool
    reuses the same draw structures. Matching stays off because this scan measures the
    NATIVE level the pool supplies — it is the input to matching, not its output.

    Args:
        dataset_name: the pool to restrict to, e.g. "71470".
        mix_block: the experiment's gugak_mix block, already window-adjusted.
        root: repo root as a string (worker processes receive plain types).
    """
    key = (dataset_name, mix_block["segment_seconds"], mix_block["split"])
    if key not in _DATASET_CACHE:
        block = dict(mix_block)
        block["datasets"] = [dataset_name]
        block["loudness_match"] = {}
        _DATASET_CACHE[key] = GugakMixDataset(
            MixDatasetConfig.from_mapping(block), Path(root), num_items=0)
    return _DATASET_CACHE[key]


# --- the measurement --------------------------------------------------------
def scan_class_windows(dataset_name: str, class_name: str, class_index: int,
                       mix_block: dict, root: str, windows: int, seed: int,
                       source_lufs_by_path: dict) -> list:
    """Draw and measure `windows` activity-aware excerpts of one class from one pool.

    Each window gets its own generator seeded from (seed, pool, class, index), so the
    table is reproducible regardless of worker count or completion order. Windows whose
    loudness is not finite (a draw that landed on silence) are kept and flagged rather
    than dropped, so the shape of the distribution isn't quietly edited.

    Args:
        dataset_name: pool to draw from.
        class_name: stem class to draw.
        class_index: position of the class in the configured list (seed component).
        mix_block: the experiment's gugak_mix block, already window-adjusted.
        root: repo root as a string.
        windows: how many windows to draw.
        seed: base seed.
        source_lufs_by_path: out_path → whole-file integrated LUFS, for the offset column.
    """
    dataset = pool_dataset(dataset_name, mix_block, root)
    meter = pyloudnorm.Meter(dataset.cfg.sample_rate)
    pool_index = int(dataset_name == mix_block["datasets"][0])

    rows = []
    for index in range(windows):
        rng = np.random.default_rng([seed, pool_index, class_index, index])
        excerpt, entry = dataset._draw_excerpt(rng, class_name)   # the dataloader's path
        window_lufs = float(meter.integrated_loudness(excerpt.T))

        # the file's level in the form the dataloader presents it (mono → duplicated)
        source_lufs = float(source_lufs_by_path.get(entry.out_path, math.nan))
        as_drawn = source_lufs + (MONO_DUPLICATION_DB if entry.source_channels == 1
                                  else 0.0)
        rows.append({
            "dataset": dataset_name, "stem_group": class_name,
            "split": mix_block["split"], "window_seconds": mix_block["segment_seconds"],
            "window_index": index, "out_path": entry.out_path,
            "clip_id": Path(entry.out_path).stem,
            "source_channels": int(entry.source_channels),
            "source_lufs": source_lufs, "source_lufs_as_drawn": as_drawn,
            "window_lufs": window_lufs if math.isfinite(window_lufs) else math.nan,
            "window_minus_source_db": (window_lufs - as_drawn
                                       if math.isfinite(window_lufs) else math.nan),
            "measurable": bool(math.isfinite(window_lufs)),
        })
    return rows


def run_scan(cfg: ChunkLoudnessScanConfig, root: Path) -> pd.DataFrame:
    """Scan every (pool × class) combination and return one row per window.

    Args:
        cfg: the assembled scan configuration.
        root: repo root.
    """
    raw = yaml.safe_load((root / cfg.experiment_config).read_text())
    mix_block = dict(raw["gugak_mix"])
    mix_block["segment_seconds"] = cfg.window_seconds     # the window under test
    mix_block["split"] = cfg.split
    mix_block["datasets"] = list(cfg.datasets)
    classes = list(mix_block["classes"])

    source_scan = pd.read_parquet(root / cfg.source_table)
    source_lufs_by_path = dict(zip(source_scan.out_path,
                                   source_scan.integrated_lufs.astype(float)))

    tasks = [(dataset_name, class_name, class_index)
             for dataset_name in cfg.datasets
             for class_index, class_name in enumerate(classes)]
    arguments = [(dataset_name, class_name, class_index, mix_block, str(root),
                  cfg.windows_per_class, cfg.seed, source_lufs_by_path)
                 for dataset_name, class_name, class_index in tasks]

    rows: list = []
    if cfg.workers <= 1:
        for argument in arguments:
            rows.extend(scan_class_windows(*argument))
            print(f"  {argument[0]} · {argument[1]}: {cfg.windows_per_class} windows")
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
            futures = {executor.submit(scan_class_windows, *argument):
                       (argument[0], argument[1]) for argument in arguments}
            for future in as_completed(futures):
                dataset_name, class_name = futures[future]
                rows.extend(future.result())
                print(f"  {dataset_name} · {class_name}: {cfg.windows_per_class} windows")
    return pd.DataFrame(rows).sort_values(
        ["dataset", "stem_group", "window_index"]).reset_index(drop=True)


# --- reporting --------------------------------------------------------------
def summarize(windows: pd.DataFrame) -> pd.DataFrame:
    """Per (pool × class): window loudness distribution and its offset from file level.

    Args:
        windows: the per-window table produced by run_scan.
    """
    measurable = windows[windows.measurable]
    summary = measurable.groupby(["dataset", "stem_group"]).agg(
        windows=("window_lufs", "size"),
        window_mean_lufs=("window_lufs", "mean"),
        window_std_db=("window_lufs", "std"),
        source_mean_lufs=("source_lufs_as_drawn", "mean"),
        offset_mean_db=("window_minus_source_db", "mean"),
        offset_std_db=("window_minus_source_db", "std"),
        mono_share=("source_channels", lambda s: float((s == 1).mean())),
    ).reset_index()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure dataloader-window loudness per class per pool.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--window-seconds", type=float, default=None,
                    help="default: the config's gugak_mix.segment_seconds")
    ap.add_argument("--windows-per-class", type=int, default=DEFAULT_WINDOWS_PER_CLASS)
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    ap.add_argument("--tables-out", default=DEFAULT_TABLES_DIR)
    ap.add_argument("--out-name", default=None,
                    help="default: chunk_loudness_reference_<window>s")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = ap.parse_args()

    root = repo_root()
    raw = yaml.safe_load((root / args.config).read_text())
    window_seconds = (args.window_seconds if args.window_seconds is not None
                      else float(raw["gugak_mix"]["segment_seconds"]))
    out_name = (args.out_name if args.out_name is not None
                else f"chunk_loudness_reference_{window_seconds:g}s")
    cfg = ChunkLoudnessScanConfig(
        experiment_config=Path(args.config), window_seconds=window_seconds,
        windows_per_class=args.windows_per_class, datasets=tuple(args.datasets),
        split=args.split, seed=args.seed, source_table=Path(args.source_table),
        tables_dir=Path(args.tables_out), out_name=out_name, workers=args.workers)

    print(f"chunk loudness scan · window {cfg.window_seconds:g}s · "
          f"{cfg.windows_per_class} windows/class · pools {list(cfg.datasets)} · "
          f"split {cfg.split} · seed {cfg.seed} · workers {cfg.workers}")
    started = time.time()
    windows = run_scan(cfg, root)

    # --- report: the distribution, and how far it sits from whole-file level ---
    summary = summarize(windows)
    print(f"\n{len(windows):,} windows in {time.time() - started:.0f}s · "
          f"{int((~windows.measurable).sum())} non-finite (silent draws)")
    print("\nper pool × class — window loudness, and its offset from file level "
          "as the dataloader presents it")
    print(summary.round(2).to_string(index=False))

    for dataset_name, group in summary.groupby("dataset"):
        print(f"\n{dataset_name}: window mean {group.window_mean_lufs.mean():+.2f} LUFS · "
              f"mean offset from file level {group.offset_mean_db.mean():+.2f} dB · "
              f"mono share {group.mono_share.mean():.0%}")

    parquet_path, csv_path = table_paths(root / cfg.tables_dir / cfg.out_name)
    windows.to_parquet(parquet_path, index=False)
    windows.to_csv(csv_path, index=False)
    summary_parquet, summary_csv = table_paths(
        root / cfg.tables_dir / f"{cfg.out_name}_summary")
    summary.to_parquet(summary_parquet, index=False)
    summary.to_csv(summary_csv, index=False)
    print(f"\nwrote {parquet_path} (+ csv twin, + summary)")


if __name__ == "__main__":
    main()
