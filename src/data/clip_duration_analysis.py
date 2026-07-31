"""clip_duration_analysis.py — how much material a training segment length costs us.

A training draw needs a contiguous excerpt of `segment_seconds`. Any source file
SHORTER than that cannot supply one without padding or looping, so the choice of
segment length silently decides how much of a pool is usable. This module measures
that cost BEFORE the policy is chosen: given a segment length and a set of shorter
reference thresholds, it reports how many clips and how many hours sit below each
one — overall, and per stem class, so a rare class losing most of its material is
visible instead of averaged away.

Written for the exp002 question (should the 71470 solo pool join the draw pool, and
what happens to clips shorter than a segment), but nothing here is specific to that
run: the segment length, the thresholds, the class scheme and the datasets are all
parameters.

Design:
  - READ-ONLY, and no audio I/O. `manifests/parquet/source_manifest.parquet` already
    carries `out_duration` per ingested file plus the taxonomy join (`stem_group`), so
    the whole analysis is a manifest read. We never walk directories or open a wav.
  - One row per POOL file: 71470 contributes its `clip` rows, 71955 its `stem` rows.
    Masters are excluded — they are not stems and would double-count hours.
  - `stem_group` is legitimately null for the 71470-only instruments held aside pending
    expert consultation. Those rows are kept and labelled, never silently dropped and
    never merged into a real class.
  - The modelled class list is read from an experiment config (`gugak_mix.classes`)
    rather than hardcoded, so re-running against a different stem scheme is a flag.
  - Every knob is a function parameter with a CLI default; a yaml config could drive
    the same functions later without touching them.

Run:
    uv run python src/data/clip_duration_analysis.py [options]
      --segment-seconds S      training segment length under test (default: 10.0)
      --thresholds S [S ...]   shorter reference thresholds (default: 2 3 5 8 10)
      --dataset {71470,71955,both}
                               which pool(s) to analyse (default: both — 71470 is the
                               subject, 71955 the size reference)
      --source-manifest PATH   input manifest (default: manifests/parquet/source_manifest.parquet)
      --class-config PATH      experiment config supplying gugak_mix.classes
      --tables-out DIR         parquet + csv tables (default: experiments/260730_solo_pool_duration)
      --markdown-out PATH      all tables as markdown (default: <tables-out>/tables.md)
      --figure-out PATH        histogram figure (default: notebooks/fig_solo_clip_durations.png)
      --no-figure              skip the figure
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

# --- defaults (CLI-overridable; a yaml config could supply the same values later) ---
DEFAULT_SOURCE_MANIFEST = "manifests/parquet/source_manifest.parquet"
DEFAULT_CLASS_CONFIG = "configs/exp001_htdemucs_9stem.yaml"
DEFAULT_TABLES_DIR = "experiments/260730_solo_pool_duration"
DEFAULT_FIGURE = "notebooks/fig_solo_clip_durations.png"
DEFAULT_SEGMENT_SECONDS = 10.0
DEFAULT_THRESHOLDS = (2.0, 3.0, 5.0, 8.0, 10.0)
DEFAULT_PERCENTILES = (0.0, 5.0, 25.0, 50.0, 75.0, 95.0, 100.0)

# --- pool definition: which manifest role is the drawable unit of each dataset ---
POOL_ROLE_PER_DATASET = {"71470": "clip", "71955": "stem"}
SUBJECT_DATASET = "71470"          # the pool exp002 would add
REFERENCE_DATASET = "71955"        # the pool exp001 already trains on
HELD_ASIDE_LABEL = "(held aside · stem_group null)"

# --- anomaly-scan sensitivities (parameters, not law) ---
DEFAULT_NEAR_ZERO_SECONDS = 2.0
DEFAULT_OUTLIER_IQR_MULTIPLE = 3.0

SECONDS_PER_HOUR = 3600.0

# --- Okabe-Ito colourblind-safe pair (published palette; the skill's JS validator
#     needs node, which sym8 does not have, so we use pre-validated values) ---
COLOR_BY_COUNT = "#0072B2"
COLOR_BY_HOURS = "#E69F00"
COLOR_MARKER = "#333333"
COLOR_GRID = "#DDDDDD"


# --- repo-root discovery + table paths (mirrors ingest.py / build_source_manifest.py) ---
def find_root(start: Path | None = None) -> Path:
    """Locate the repo root (holds pyproject.toml)."""
    p = Path.cwd() if start is None else start
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists():
            return cand
    raise FileNotFoundError("repo root (pyproject.toml) not found above cwd")


def table_paths(base: Path) -> tuple[Path, Path]:
    """Map an output basename to its (parquet, csv) twin paths, creating parents.

    Mirrors the repo-wide manifest rule (parquet = canonical, csv = eyeball twin) for
    analysis outputs, so the two never drift apart.

    Args:
        base: table basename without suffix, e.g. Path(".../per_class_at_segment").
    """
    parquet = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")
    parquet.parent.mkdir(parents=True, exist_ok=True)
    return parquet, csv


# --- configuration -----------------------------------------------------------
@dataclass
class DurationAnalysisConfig:
    """Everything the analysis needs; assembled from CLI args (or a yaml, later)."""
    segment_seconds: float
    thresholds: tuple[float, ...]
    percentiles: tuple[float, ...]
    datasets: tuple[str, ...]
    modelled_classes: tuple[str, ...]
    source_manifest: Path
    tables_dir: Path
    markdown_out: Path
    figure_out: Path | None
    near_zero_seconds: float
    outlier_iqr_multiple: float


def load_modelled_classes(class_config_path: Path) -> tuple[str, ...]:
    """Read the modelled stem classes from an experiment config's gugak_mix.classes.

    The class scheme is an experiment decision, so it is read from the run config
    rather than restated here; a different scheme is a different --class-config.

    Args:
        class_config_path: path to an experiment yaml carrying gugak_mix.classes.
    """
    raw = yaml.safe_load(Path(class_config_path).read_text())
    classes = raw.get("gugak_mix", {}).get("classes")
    if not classes:
        raise KeyError(f"{class_config_path} has no gugak_mix.classes list")
    return tuple(str(c) for c in classes)


# --- manifest loading (the only input; strictly read-only) -------------------
def load_pool_frame(source_manifest: Path, datasets: Sequence[str]) -> pd.DataFrame:
    """Load the drawable pool files of the requested datasets, with a class label.

    Keeps only each dataset's drawable role (71470 clips, 71955 stems — masters are
    not stems and would double-count hours) and adds `stem_class`, which is
    `stem_group` with the deliberately-null held-aside rows labelled instead of
    dropped.

    Args:
        source_manifest: path to source_manifest.parquet.
        datasets: dataset ids to keep, e.g. ("71470", "71955").
    """
    frame = pd.read_parquet(source_manifest)
    frame["dataset"] = frame["dataset"].astype(str)

    keep_mask = pd.Series(False, index=frame.index)
    for dataset in datasets:
        role = POOL_ROLE_PER_DATASET[dataset]
        keep_mask |= (frame["dataset"] == dataset) & (frame["role"] == role)
    pool = frame[keep_mask].copy()

    pool["stem_class"] = pool["stem_group"].fillna(HELD_ASIDE_LABEL)
    return pool.reset_index(drop=True)


def class_display_order(pool: pd.DataFrame, modelled_classes: Sequence[str]) -> list[str]:
    """Order classes as: modelled scheme first (config order), then extras, then held-aside.

    Args:
        pool: pool frame carrying `stem_class`.
        modelled_classes: the experiment's class list, in config order.
    """
    present = list(pool["stem_class"].unique())
    modelled = [c for c in modelled_classes if c in present]
    extras = sorted(c for c in present if c not in modelled and c != HELD_ASIDE_LABEL)
    tail = [HELD_ASIDE_LABEL] if HELD_ASIDE_LABEL in present else []
    return modelled + extras + tail


# --- core statistics ---------------------------------------------------------
def duration_percentiles(durations: pd.Series,
                         percentiles: Sequence[float]) -> dict[str, float]:
    """Percentiles of a duration series, keyed `p0`, `p5`, … (p0/p100 = min/max).

    Args:
        durations: clip durations in seconds.
        percentiles: percentile positions in 0-100.
    """
    quantiles = durations.quantile([p / 100.0 for p in percentiles])
    return {f"p{p:g}": float(v) for p, v in zip(percentiles, quantiles.to_numpy())}


def summarize_pool(pool: pd.DataFrame,
                   percentiles: Sequence[float] = DEFAULT_PERCENTILES) -> pd.DataFrame:
    """One row per dataset: file count, total hours, mean, and the percentile spread.

    Args:
        pool: pool frame from `load_pool_frame`.
        percentiles: percentile positions in 0-100.
    """
    rows = []
    for dataset, group in pool.groupby("dataset", sort=True):
        durations = group["out_duration"]
        rows.append({
            "dataset": dataset,
            "files": int(len(group)),
            "total_hours": round(float(durations.sum()) / SECONDS_PER_HOUR, 3),
            "mean_seconds": round(float(durations.mean()), 3),
            **{k: round(v, 3) for k, v in duration_percentiles(durations, percentiles).items()},
        })
    return pd.DataFrame(rows)


def cumulative_below_thresholds(durations: pd.Series,
                                thresholds: Sequence[float]) -> pd.DataFrame:
    """Cumulative share of a pool shorter than each threshold, by count and by duration.

    "Shorter than" is strict: a file exactly as long as the threshold still yields one
    full excerpt, so it is not counted as lost.

    Args:
        durations: file durations in seconds.
        thresholds: cut points in seconds, ascending.
    """
    total_files = int(len(durations))
    total_seconds = float(durations.sum())
    rows = []
    for threshold in thresholds:
        below = durations[durations < threshold]
        rows.append({
            "threshold_seconds": float(threshold),
            "files_below": int(len(below)),
            "frac_files_below": (len(below) / total_files) if total_files else 0.0,
            "hours_below": float(below.sum()) / SECONDS_PER_HOUR,
            "frac_hours_below": (float(below.sum()) / total_seconds) if total_seconds else 0.0,
        })
    return pd.DataFrame(rows)


def per_class_thresholds(pool: pd.DataFrame, thresholds: Sequence[float],
                         class_order: Sequence[str]) -> pd.DataFrame:
    """Tidy (class × threshold) cumulative table: files/hours below each cut point.

    Args:
        pool: pool frame (single dataset).
        thresholds: cut points in seconds.
        class_order: class labels in display order.
    """
    frames = []
    for stem_class in class_order:
        group = pool[pool["stem_class"] == stem_class]
        if group.empty:
            continue
        table = cumulative_below_thresholds(group["out_duration"], thresholds)
        table.insert(0, "stem_class", stem_class)
        table.insert(1, "files", int(len(group)))
        table.insert(2, "total_hours", float(group["out_duration"].sum()) / SECONDS_PER_HOUR)
        frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def per_class_at_segment(pool: pd.DataFrame, segment_seconds: float,
                         class_order: Sequence[str]) -> pd.DataFrame:
    """Per-class loss at the segment length, sorted by the share of hours it would cost.

    This is the decision table: one row per class, how much of it is shorter than a
    single training segment. Held-aside rows keep their own line.

    Args:
        pool: pool frame (single dataset).
        segment_seconds: the training segment length under test.
        class_order: class labels in display order (used only to keep unknowns out).
    """
    rows = []
    for stem_class in class_order:
        group = pool[pool["stem_class"] == stem_class]
        if group.empty:
            continue
        durations = group["out_duration"]
        below = durations[durations < segment_seconds]
        rows.append({
            "stem_class": stem_class,
            "files": int(len(group)),
            "total_hours": float(durations.sum()) / SECONDS_PER_HOUR,
            "median_seconds": float(durations.median()),
            "files_below_segment": int(len(below)),
            "frac_files_below": len(below) / len(group),
            "hours_below_segment": float(below.sum()) / SECONDS_PER_HOUR,
            "frac_hours_below": float(below.sum()) / float(durations.sum()),
            "hours_kept_if_reject": float(durations[durations >= segment_seconds].sum())
                                    / SECONDS_PER_HOUR,
        })
    return (pd.DataFrame(rows)
            .sort_values("frac_hours_below", ascending=False)
            .reset_index(drop=True))


def pool_size_by_class(pool: pd.DataFrame, class_order: Sequence[str]) -> pd.DataFrame:
    """File count and total hours per class per dataset, side by side.

    Answers "how much would the solo pool actually add" — the two datasets as columns
    so the relative size per class is one glance.

    Args:
        pool: pool frame spanning both datasets.
        class_order: class labels in display order.
    """
    grouped = pool.groupby(["stem_class", "dataset"])["out_duration"]
    sizes = grouped.agg(files="size", seconds="sum").reset_index()
    sizes["hours"] = sizes["seconds"] / SECONDS_PER_HOUR

    wide = sizes.pivot(index="stem_class", columns="dataset", values=["files", "hours"])
    wide.columns = [f"{metric}_{dataset}" for metric, dataset in wide.columns]
    wide = wide.reindex([c for c in class_order if c in wide.index])

    # relative size of the solo addition, where both pools have material
    files_subject = f"files_{SUBJECT_DATASET}"
    hours_subject = f"hours_{SUBJECT_DATASET}"
    hours_reference = f"hours_{REFERENCE_DATASET}"
    if hours_subject in wide.columns and hours_reference in wide.columns:
        wide["solo_hours_vs_ensemble"] = wide[hours_subject] / wide[hours_reference]
    if files_subject in wide.columns:
        wide[files_subject] = wide[files_subject].fillna(0).astype("int64")
    return wide.reset_index()


# --- anomaly scan (report, never fix) ---------------------------------------
def scan_anomalies(pool: pd.DataFrame, near_zero_seconds: float,
                   outlier_iqr_multiple: float) -> dict[str, pd.DataFrame]:
    """Collect things worth a human look: stub clips, outliers, duplicates, format spread.

    Purely descriptive — nothing is dropped or corrected here. Duplicate *content* is
    inferred from an exact (frames, peak, rms) match, which is a strong fingerprint for
    the same audio ingested twice, not proof.

    Args:
        pool: pool frame from `load_pool_frame`.
        near_zero_seconds: anything shorter than this is flagged as a stub.
        outlier_iqr_multiple: upper fence is q75 + multiple × IQR.
    """
    report: dict[str, pd.DataFrame] = {}
    identity_columns = ["dataset", "file_id", "clip_id", "song_id", "instrument_canonical",
                        "stem_class", "out_duration"]

    # short stubs (one absolute rule) and long outliers (fence PER DATASET — a 12 s clip
    # and a 4-minute ensemble stem must never share one distribution)
    report["stub_files"] = (pool[pool["out_duration"] < near_zero_seconds]
                            [identity_columns].sort_values("out_duration"))
    fence_rows, outlier_frames = [], []
    for dataset, group in pool.groupby("dataset", sort=True):
        q25, q75 = group["out_duration"].quantile([0.25, 0.75])
        upper_fence = float(q75 + outlier_iqr_multiple * (q75 - q25))
        fence_rows.append({"dataset": dataset, "upper_fence_seconds": round(upper_fence, 3),
                           "outlier_iqr_multiple": outlier_iqr_multiple})
        outlier_frames.append(group[group["out_duration"] > upper_fence][identity_columns])
    report["outlier_fence"] = pd.DataFrame(fence_rows)
    report["long_outliers"] = (pd.concat(outlier_frames, ignore_index=True)
                               .sort_values("out_duration", ascending=False))

    # duplicate keys and duplicate-looking content
    report["duplicate_file_ids"] = pool[pool.duplicated("file_id", keep=False)][identity_columns]
    fingerprint = ["dataset", "out_frames", "peak", "rms"]
    duplicate_audio = pool[pool.duplicated(fingerprint, keep=False)]
    report["duplicate_audio_fingerprints"] = (duplicate_audio[identity_columns + ["out_frames"]]
                                              .sort_values(["out_frames", "file_id"]))

    # format spread and content-health flags carried by the QC join
    report["source_formats"] = (pool.groupby(["dataset", "src_sr", "src_subtype", "src_channels"])
                                .size().reset_index(name="files"))
    report["ingested_formats"] = (pool.groupby(["dataset", "out_sr", "out_subtype", "out_channels"])
                                  .size().reset_index(name="files"))
    flagged = pool[pool["dead"].fillna(False) | (pool["error"].fillna("") != "")]
    report["dead_or_errored"] = flagged[identity_columns + ["dead", "active_frac", "error"]]
    quietest_per_dataset = 15
    quietest = [group.nsmallest(min(len(group), quietest_per_dataset), "active_frac")
                for _, group in pool.groupby("dataset", sort=True)]
    report["low_activity_files"] = (pd.concat(quietest, ignore_index=True)
                                    [identity_columns + ["active_frac", "rms"]])
    return report


# --- figure ------------------------------------------------------------------
def plot_duration_histogram(pool: pd.DataFrame, segment_seconds: float,
                            class_losses: pd.DataFrame, out_path: Path,
                            bin_width_seconds: float = 1.0) -> Path:
    """Two-panel figure: duration histogram with a segment marker + per-class loss bars.

    Left panel is the distribution question (where does the mass sit relative to one
    segment); right panel is the fairness question (which classes pay for a reject
    policy). Korean class names render via koreanize-matplotlib.

    Args:
        pool: single-dataset pool frame.
        segment_seconds: marker position and the loss cut point.
        class_losses: output of `per_class_at_segment` (already sorted).
        out_path: png destination.
        bin_width_seconds: histogram bin width.
    """
    import koreanize_matplotlib  # noqa: F401 — registers the Korean font on import
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    durations = pool["out_duration"]
    frac_files_below = float((durations < segment_seconds).mean())
    frac_hours_below = float(durations[durations < segment_seconds].sum() / durations.sum())

    figure, (ax_hist, ax_class) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                               gridspec_kw={"width_ratios": [1.25, 1.0]})

    # left: duration histogram, with the segment length marked
    bins = np.arange(0.0, float(durations.max()) + bin_width_seconds, bin_width_seconds)
    ax_hist.hist(durations, bins=bins, color=COLOR_BY_COUNT, edgecolor="none")
    ax_hist.axvline(segment_seconds, color=COLOR_MARKER, linestyle="--", linewidth=2.0)
    ax_hist.annotate(f"segment = {segment_seconds:g} s\n"
                     f"{frac_files_below:.1%} of clips · {frac_hours_below:.1%} of hours below",
                     xy=(segment_seconds + bin_width_seconds * 3,
                         ax_hist.get_ylim()[1] * 0.92),
                     color=COLOR_MARKER, fontsize=11, va="top")
    ax_hist.set_xlabel("clip duration (seconds)")
    ax_hist.set_ylabel(f"clips per {bin_width_seconds:g} s bin")
    ax_hist.set_title(f"{SUBJECT_DATASET} solo-clip duration distribution", loc="left",
                      fontsize=12)

    # right: per-class share below the segment, by clip count and by hours
    plotted = class_losses.iloc[::-1]          # longest bar at the top
    positions = np.arange(len(plotted))
    bar_height = 0.38
    ax_class.barh(positions + bar_height / 2, plotted["frac_files_below"] * 100.0,
                  height=bar_height, color=COLOR_BY_COUNT, label="% of clips")
    ax_class.barh(positions - bar_height / 2, plotted["frac_hours_below"] * 100.0,
                  height=bar_height, color=COLOR_BY_HOURS, label="% of hours")
    ax_class.set_yticks(positions, plotted["stem_class"])
    ax_class.set_xlabel(f"share shorter than one {segment_seconds:g} s segment (%)")
    ax_class.set_title("what a reject policy would cost, per class", loc="left", fontsize=12)
    ax_class.legend(frameon=False, loc="lower right")

    # shared cosmetics: recessive grid, no box
    for axis in (ax_hist, ax_class):
        axis.grid(axis="x" if axis is ax_class else "y", color=COLOR_GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path


# --- markdown rendering (no tabulate dependency) -----------------------------
def to_markdown_table(frame: pd.DataFrame, float_format: str = "{:.3f}") -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table.

    Args:
        frame: table to render.
        float_format: format string applied to float cells.
    """
    def cell(value: object) -> str:
        if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
            return "—"
        if isinstance(value, float):
            return float_format.format(value)
        return str(value)

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |"
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *body])


def write_tables(tables: dict[str, pd.DataFrame], tables_dir: Path,
                 markdown_out: Path) -> None:
    """Persist every table as parquet + csv, and all of them as one markdown dump.

    Args:
        tables: name → table.
        tables_dir: directory for the parquet/csv twins.
        markdown_out: single markdown file holding every table.
    """
    sections = []
    for name, frame in tables.items():
        parquet_path, csv_path = table_paths(tables_dir / name)
        frame.to_parquet(parquet_path, index=False)
        frame.to_csv(csv_path, index=False)
        sections.append(f"## {name}  ({len(frame)} rows)\n\n{to_markdown_table(frame)}\n")
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text("\n".join(sections), encoding="utf-8")


# --- analysis orchestration --------------------------------------------------
def run_analysis(cfg: DurationAnalysisConfig) -> dict[str, pd.DataFrame]:
    """Build every table for the configured segment length and thresholds.

    Args:
        cfg: assembled analysis configuration.
    """
    pool = load_pool_frame(cfg.source_manifest, cfg.datasets)
    order = class_display_order(pool, cfg.modelled_classes)

    tables: dict[str, pd.DataFrame] = {"pool_summary": summarize_pool(pool, cfg.percentiles)}

    # the subject pool: overall + per-class threshold tables
    subject = pool[pool["dataset"] == SUBJECT_DATASET]
    if not subject.empty:
        overall = cumulative_below_thresholds(subject["out_duration"], cfg.thresholds)
        overall.insert(0, "dataset", SUBJECT_DATASET)
        tables["cumulative_overall"] = overall
        tables["per_class_thresholds"] = per_class_thresholds(subject, cfg.thresholds, order)
        tables["per_class_at_segment"] = per_class_at_segment(subject, cfg.segment_seconds, order)

    # both pools side by side, so the size of the addition is visible per class
    tables["pool_size_by_class"] = pool_size_by_class(pool, order)

    # anomalies noticed in passing
    tables.update({f"anomaly_{k}": v for k, v in
                   scan_anomalies(pool, cfg.near_zero_seconds, cfg.outlier_iqr_multiple).items()})
    return tables


def report(tables: dict[str, pd.DataFrame], cfg: DurationAnalysisConfig) -> None:
    """Print the headline tables and the anomaly row counts to stdout."""
    print(f"\n=== clip duration analysis — segment {cfg.segment_seconds:g} s · "
          f"thresholds {[f'{t:g}' for t in cfg.thresholds]} ===\n")
    for name in ("pool_summary", "cumulative_overall", "per_class_at_segment",
                 "pool_size_by_class"):
        if name in tables:
            print(f"--- {name} ---")
            print(tables[name].to_string(index=False))
            print()
    print("--- anomaly scan (row counts) ---")
    for name, frame in tables.items():
        if name.startswith("anomaly_"):
            print(f"  {name}: {len(frame)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure how much of a source pool falls below a training segment length.")
    ap.add_argument("--segment-seconds", type=float, default=DEFAULT_SEGMENT_SECONDS,
                    help="training segment length under test")
    ap.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
                    help="cumulative-fraction cut points, seconds")
    ap.add_argument("--percentiles", type=float, nargs="+", default=list(DEFAULT_PERCENTILES),
                    help="percentile positions (0-100) for the summary table")
    ap.add_argument("--dataset", choices=["71470", "71955", "both"], default="both")
    ap.add_argument("--source-manifest", default=DEFAULT_SOURCE_MANIFEST)
    ap.add_argument("--class-config", default=DEFAULT_CLASS_CONFIG,
                    help="experiment yaml supplying gugak_mix.classes (the stem scheme)")
    ap.add_argument("--tables-out", default=DEFAULT_TABLES_DIR)
    ap.add_argument("--markdown-out", default=None, help="default: <tables-out>/tables.md")
    ap.add_argument("--figure-out", default=DEFAULT_FIGURE)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--near-zero-seconds", type=float, default=DEFAULT_NEAR_ZERO_SECONDS,
                    help="flag files shorter than this as stubs in the anomaly scan")
    ap.add_argument("--outlier-iqr-multiple", type=float, default=DEFAULT_OUTLIER_IQR_MULTIPLE,
                    help="upper outlier fence = q75 + multiple x IQR")
    args = ap.parse_args()

    # resolve paths + assemble the config
    root = find_root()
    tables_dir = root / args.tables_out
    datasets = ("71470", "71955") if args.dataset == "both" else (args.dataset,)
    cfg = DurationAnalysisConfig(
        segment_seconds=args.segment_seconds,
        thresholds=tuple(sorted(args.thresholds)),
        percentiles=tuple(args.percentiles),
        datasets=datasets,
        modelled_classes=load_modelled_classes(root / args.class_config),
        source_manifest=root / args.source_manifest,
        tables_dir=tables_dir,
        markdown_out=(root / args.markdown_out) if args.markdown_out else tables_dir / "tables.md",
        figure_out=None if args.no_figure else root / args.figure_out,
        near_zero_seconds=args.near_zero_seconds,
        outlier_iqr_multiple=args.outlier_iqr_multiple,
    )

    # run, persist, report
    tables = run_analysis(cfg)
    write_tables(tables, cfg.tables_dir, cfg.markdown_out)
    report(tables, cfg)
    print(f"\nwrote {len(tables)} tables (parquet+csv) to {cfg.tables_dir}")
    print(f"wrote markdown dump to {cfg.markdown_out}")

    if cfg.figure_out is not None and "per_class_at_segment" in tables:
        pool = load_pool_frame(cfg.source_manifest, (SUBJECT_DATASET,))
        figure_path = plot_duration_histogram(pool, cfg.segment_seconds,
                                              tables["per_class_at_segment"], cfg.figure_out)
        print(f"wrote figure to {figure_path}")


if __name__ == "__main__":
    main()
