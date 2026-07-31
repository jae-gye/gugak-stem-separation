"""loudness_profile_scan.py — loudness/crest profiling, and why it does NOT flag disguised masters.

⚠️ READ THIS BEFORE REBUILDING THIS DETECTOR. Run 2026-07-30 over all 16,615 files (465 h).
The idea was to catch "a master wearing a stem's filename" even when the bytes differ —
a master copied over a stem and then further processed keeps a master's loudness signature,
which byte comparison misses. **As a binary flag it does not work, and the reason is
structural, not a tuning problem.** Full write-up → docs/solo_pool_duration_analysis.md §6.

MEASURED POPULATIONS (integrated LUFS, crest dB, active_frac)
    masters       n=  903   -13.08 +/- 2.62   14.19 +/- 2.42   0.982 +/- 0.031
    71955 stems   n= 5767   -20.84 +/- 5.11   20.74 +/- 3.53   0.761 +/- 0.248
    71470 clips   n= 9945   -25.52 +/- 8.11   18.73 +/- 5.43   0.859 +/- 0.165

The CENTRES separate as expected — masters are ~7.8 dB louder and ~6.5 dB lower-crest than
ensemble stems. But the stem cloud is twice as wide and swallows the master cloud:
    27.9% of stems fall inside the masters' 95% ellipse
    70.3% of masters fall inside the stems' ellipse
So: a naive likelihood rule flags 2,479 of 15,712 sources (15.8%, useless), while calibrating
with a realistic rare prior (1e-3) flags ZERO — including the known positives. No threshold
catches the known-bad files without dragging in hundreds of legitimate loud dense stems.

THREE THINGS ALREADY TRIED, SO YOU DON'T REPEAT THEM
  1. Adding `active_frac` as a third axis makes it WORSE (masters-inside-stem-ellipse rises
     70.3% -> 89.5%): solo clips are continuously active too.
  2. Pooling the two datasets destroys the ranking. 71470 has no masters at all, so
     "inside the master distribution" is meaningless there, and loud dense solo clips at
     ~-11 LUFS crowd out the real 71955 candidates. ALWAYS score per dataset.
  3. A loudness detector structurally CANNOT catch a solo-piece master-as-stem.
     `0905_창작국악_창작국악`'s 해금 is bit-identical to its own master yet ranks 4165/5767,
     because that master is itself stem-like (Mahalanobis 4.38 from the master centroid,
     more extreme than 99.1% of real masters) — a solo piece was never bus-compressed. The
     premise "masters are louder and more limited" describes ENSEMBLE masters only.

WHAT IT IS STILL GOOD FOR: a per-dataset RANKING for human review. Restricted to 71955 the
three confirmed `0714_민속악_민요` mix-copies rank 1, 2, 3 of 5,767. Use `--stage score` to
regenerate the review list; treat the flag column as advisory, never as a verdict.

Two stages, because stage 1 costs ~20 min and stage 2 costs seconds (mirrors
src/data/activity_scan.py -> src/data/build_activity_manifest.py):
    measure : integrated LUFS (ITU-R BS.1770, gated) + crest dB per file  -> parquet
    score   : fit the two populations, rank every source, report          -> parquet

Note the metrics differ in one important way: LUFS is GATED (silence excluded, so a sparse
박 stem is measured only where it plays) while crest spans the whole file and IS inflated by
silence. The confound therefore has a direction — sparse stems are pushed away from the
master cluster (safe), dense loud stems drift toward it (false-positive risk).

Run:
    uv run python scripts/loudness_profile_scan.py --stage measure [--workers N]
    uv run python scripts/loudness_profile_scan.py --stage score [--prior 1e-3]
    uv run python scripts/loudness_profile_scan.py --stage both
"""
from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

DEFAULT_SOURCE_MANIFEST = "manifests/parquet/source_manifest.parquet"
DEFAULT_MEASURE_OUT = "manifests/loudness_crest_scan"
DEFAULT_SCORE_OUT = "manifests/master_profile_ranking"
DEFAULT_PRIOR = 1e-3          # disguised masters are rare; ~4 suspected in 15.7k sources
DEFAULT_TOP = 15
METRIC_COLUMNS = ["integrated_lufs", "crest_db"]
CARRY_COLUMNS = ["file_id", "dataset", "role", "song_id", "clip_id",
                 "instrument_canonical", "stem_group", "split", "genre_sub",
                 "out_duration", "out_sr", "out_channels"]

# established by byte/correlation tests — a detector that misses these is not trustworthy
KNOWN_POSITIVES = [("0905_창작국악_창작국악", "해금"), ("0714_민속악_민요", "대금"),
                   ("0714_민속악_민요", "아쟁"), ("0714_민속악_민요", "피리")]

_METER = None


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


# --- stage 1: measure every file -------------------------------------------------
def _init_worker(sample_rate: int) -> None:
    """One BS.1770 meter per worker — rebuilding it per file is pure overhead."""
    global _METER
    import pyloudnorm

    warnings.filterwarnings("ignore")
    _METER = pyloudnorm.Meter(sample_rate)


def measure_file(task: dict) -> dict:
    """Integrated LUFS + crest dB for one file. Never raises; errors land in the row.

    Args:
        task: identity columns plus `out_path`.
    """
    row = dict(task)
    row["error"] = ""
    try:
        # read once as float32 — some masters run ~1150 s, so a float64 copy is costly
        audio, _ = sf.read(row["out_path"], dtype="float32", always_2d=True)

        # peak and rms without materializing a second large array
        peak = max(abs(float(audio.max())), abs(float(audio.min())))
        sum_of_squares = float(np.einsum("ij,ij->", audio, audio, dtype=np.float64))
        rms = float(np.sqrt(sum_of_squares / audio.size)) if audio.size else 0.0

        row["measured_peak"] = peak
        row["measured_rms"] = rms
        row["crest_db"] = float(20.0 * np.log10(peak / rms)) if rms > 0.0 and peak > 0.0 else np.nan
        row["integrated_lufs"] = float(_METER.integrated_loudness(audio))
    except Exception as exc:                       # noqa: BLE001 — record, keep scanning
        row["error"] = f"{type(exc).__name__}: {exc}"
        for key in ("measured_peak", "measured_rms", "crest_db", "integrated_lufs"):
            row.setdefault(key, np.nan)
    return row


def run_measure(root: Path, source_manifest: str, out_base: str, workers: int) -> pd.DataFrame:
    """Measure LUFS and crest for every file in the ingest store.

    Args:
        root: repo root.
        source_manifest: manifest path, repo-relative.
        out_base: output basename under manifests/.
        workers: parallel processes. Keep this modest on a shared box — trainings need CPU
            for their dataloaders.
    """
    manifest = pd.read_parquet(root / source_manifest)
    assert manifest["out_sr"].nunique() == 1, "store is not single-rate; the meter needs one rate"
    sample_rate = int(manifest["out_sr"].iloc[0])

    tasks = [{**{c: r[c] for c in CARRY_COLUMNS}, "out_path": r["out_path"]}
             for _, r in manifest.iterrows()]
    print(f"measuring {len(tasks)} files ({manifest['out_duration'].sum() / 3600:.1f} h) "
          f"at {sample_rate} Hz with {workers} workers", flush=True)

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(sample_rate,)) as executor:
        futures = [executor.submit(measure_file, t) for t in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index % 500 == 0:
                print(f"  {index}/{len(tasks)}", flush=True)

    frame = pd.DataFrame(rows)
    parquet, csv = table_paths(root / out_base)
    frame.to_parquet(parquet, index=False)
    frame.to_csv(csv, index=False)
    errors = int((frame["error"] != "").sum())
    silent = int((~np.isfinite(frame["integrated_lufs"])).sum())
    print(f"\nmeasured {len(frame)} files · errors {errors} · non-finite LUFS {silent}")
    print(f"wrote {parquet}")
    return frame


# --- stage 2: fit the populations and rank ---------------------------------------
def fit_gaussian(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean vector and full covariance of an (n, d) point cloud."""
    return points.mean(axis=0), np.cov(points, rowvar=False)


def log_density(points: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Log density of a multivariate normal at each row of `points`."""
    dimension = points.shape[1]
    precision = np.linalg.inv(covariance)
    _, log_determinant = np.linalg.slogdet(covariance)
    centred = points - mean
    quadratic = np.einsum("ij,jk,ik->i", centred, precision, centred)
    return -0.5 * (quadratic + log_determinant + dimension * np.log(2.0 * np.pi))


def fraction_inside_ellipse(points: np.ndarray, mean: np.ndarray,
                            covariance: np.ndarray, chi_square_cut: float) -> float:
    """Fraction of `points` inside a fitted Gaussian's confidence ellipse.

    Args:
        points: (n, d) cloud to test.
        mean: fitted centre.
        covariance: fitted covariance.
        chi_square_cut: chi-square quantile for the desired level and dimensionality.
    """
    precision = np.linalg.inv(covariance)
    centred = points - mean
    return float(np.mean(np.einsum("ij,jk,ik->i", centred, precision, centred) < chi_square_cut))


def score_sources(measurements: pd.DataFrame, prior_master: float) -> pd.DataFrame:
    """Rank every source by how master-like its loudness profile is, PER DATASET.

    Scoring is per dataset on purpose — see trap 2 in the module docstring. The returned
    `master_profile_log_odds` includes the rare prior, so it is a posterior log-odds and
    positive values would mean "more likely a master than a stem". In practice nothing
    crosses zero; use `rank_in_dataset` instead.

    Args:
        measurements: stage-1 output.
        prior_master: prior probability that a given source is a disguised master.
    """
    usable = measurements[(measurements["error"] == "")
                          & np.isfinite(measurements["integrated_lufs"])
                          & np.isfinite(measurements["crest_db"])]
    masters = usable[(usable["dataset"] == "71955") & (usable["role"] == "master")]
    master_mean, master_covariance = fit_gaussian(masters[METRIC_COLUMNS].to_numpy())

    scored_parts = []
    for dataset, group in usable[usable["role"] != "master"].groupby("dataset"):
        stem_mean, stem_covariance = fit_gaussian(group[METRIC_COLUMNS].to_numpy())
        points = group[METRIC_COLUMNS].to_numpy()
        part = group.copy()
        part["master_profile_log_odds"] = (
            log_density(points, master_mean, master_covariance)
            - log_density(points, stem_mean, stem_covariance)
            + np.log(prior_master / (1.0 - prior_master)))
        part = part.sort_values("master_profile_log_odds", ascending=False)
        part["rank_in_dataset"] = np.arange(1, len(part) + 1)
        part["sources_in_dataset"] = len(part)
        scored_parts.append(part)
    return pd.concat(scored_parts, ignore_index=True)


def report_score(measurements: pd.DataFrame, scored: pd.DataFrame, top: int) -> None:
    """Print the populations, their overlap, the per-dataset ranking, and the known positives."""
    pd.set_option("display.width", 250)
    usable = measurements[measurements["error"] == ""]
    masters = usable[(usable["dataset"] == "71955") & (usable["role"] == "master")]

    print("=== populations ===")
    groups = [("masters", masters)]
    groups += [(f"{d} sources", g) for d, g in
               usable[usable["role"] != "master"].groupby("dataset")]
    for label, group in groups:
        print(f"  {label:16s} n={len(group):5d}  "
              f"LUFS {group.integrated_lufs.mean():+7.2f} +/- {group.integrated_lufs.std():5.2f}  "
              f"crest {group.crest_db.mean():6.2f} +/- {group.crest_db.std():5.2f}")

    print("\n=== overlap (the reason this is not a flag) ===")
    master_mean, master_covariance = fit_gaussian(masters[METRIC_COLUMNS].to_numpy())
    ensemble = usable[(usable["dataset"] == "71955") & (usable["role"] == "stem")]
    stem_mean, stem_covariance = fit_gaussian(ensemble[METRIC_COLUMNS].to_numpy())
    chi_square_95_two_dimensions = 5.991
    print(f"  stems inside master 95% ellipse : "
          f"{fraction_inside_ellipse(ensemble[METRIC_COLUMNS].to_numpy(), master_mean, master_covariance, chi_square_95_two_dimensions) * 100:5.1f}%")
    print(f"  masters inside stem 95% ellipse : "
          f"{fraction_inside_ellipse(masters[METRIC_COLUMNS].to_numpy(), stem_mean, stem_covariance, chi_square_95_two_dimensions) * 100:5.1f}%")

    columns = ["rank_in_dataset", "song_id", "clip_id", "instrument_canonical", "stem_group",
               "split", "out_duration", "integrated_lufs", "crest_db", "master_profile_log_odds"]
    for dataset, group in scored.groupby("dataset"):
        print(f"\n=== {dataset}: top {top} review candidates (RANKING, not verdicts) ===")
        print(group.nsmallest(top, "rank_in_dataset")[columns].round(3).to_string(index=False))

    print("\n=== sensitivity check: known positives ===")
    for song_id, instrument in KNOWN_POSITIVES:
        match = scored[(scored["song_id"] == song_id)
                       & (scored["instrument_canonical"] == instrument)]
        if match.empty:
            print(f"  {song_id}/{instrument}: NOT PRESENT")
            continue
        row = match.iloc[0]
        print(f"  {song_id}/{instrument}: rank {int(row.rank_in_dataset):5d}/"
              f"{int(row.sources_in_dataset)}  LUFS {row.integrated_lufs:+7.2f}  "
              f"crest {row.crest_db:6.2f}  log-odds {row.master_profile_log_odds:+7.2f}")
    print("  (0905's 해금 is EXPECTED to rank low — its own master is stem-like; see docstring)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Profile loudness/crest and rank master-like stems (ranking, not a flag).")
    ap.add_argument("--stage", choices=["measure", "score", "both"], default="both")
    ap.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST)
    ap.add_argument("--measure-out", default=DEFAULT_MEASURE_OUT)
    ap.add_argument("--score-out", default=DEFAULT_SCORE_OUT)
    ap.add_argument("--workers", type=int, default=4,
                    help="keep modest on a shared box — trainings need CPU (default: 4)")
    ap.add_argument("--prior", type=float, default=DEFAULT_PRIOR,
                    help="prior probability a source is a disguised master")
    ap.add_argument("--top", type=int, default=DEFAULT_TOP)
    args = ap.parse_args()

    root = find_root()
    measure_parquet, _ = table_paths(root / args.measure_out)

    # stage 1 (expensive) — skip when scoring an existing measurement table
    if args.stage in ("measure", "both"):
        measurements = run_measure(root, args.source_manifest, args.measure_out, args.workers)
    else:
        measurements = pd.read_parquet(measure_parquet)
        print(f"loaded {len(measurements)} measurements from {measure_parquet}")

    # stage 2 (cheap) — fit, rank, report
    if args.stage in ("score", "both"):
        scored = score_sources(measurements, args.prior)
        parquet, csv = table_paths(root / args.score_out)
        scored.to_parquet(parquet, index=False)
        scored.to_csv(csv, index=False)
        report_score(measurements, scored, args.top)
        print(f"\nwrote {parquet}")


if __name__ == "__main__":
    main()
