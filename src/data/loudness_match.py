"""loudness_match.py — draw solo-pool stem loudness from the ensemble distribution.

The 71470 solo clips reach the model at a different level from 71955 ensemble stems, and
the rare classes lean hardest on the solo pool. Left untreated, loudness becomes a proxy
for "which pool did this stem come from", which the model can learn instead of timbre
("quiet ⇒ 양금"). This module removes that cue.

For each solo-sourced stem it DRAWS a target loudness from that class's ENSEMBLE
distribution and returns the gain that puts the clip there. Drawing, not normalizing to
the class mean, is the point: one fixed value would give solo stems zero variance where
ensemble stems have σ ≈ 6 dB — a cleaner shortcut, not no shortcut. Draws are empirical
(sampled from observed values plus small jitter) rather than fitted, because per-class
loudness is plausibly multimodal — the same instrument leads one song and provides texture
in another.

BOTH SIDES ARE COMPARED AT THE SCALE THE MODEL SEES (`reference_grain: chunk`, the
default). Whole-file loudness is the wrong scale, and wrong by a pool-dependent amount:

  - a mono file is measured as one channel by the file scan, but the dataloader duplicates
    it to stereo, and BS.1770 sums channel energies — exactly +3.01 dB. The solo pool is
    ~79% mono after ingest and the ensemble pool is ~1%, so this lands on one pool only
  - an activity-aware window is not a whole file: measured against file level, ensemble
    windows sit ~0.75 dB low (a window catches one passage of a song part) while solo
    windows sit ~0.15 dB low (a phrase clip is nearly all window)

Together those were the measured +2.75 dB residual left by file-level matching. Chunk grain
removes them by construction: targets come from measured ensemble WINDOWS
(`src/data/chunk_loudness_scan.py`), and a solo clip's own level is put on the same scale
before the offset is computed — its file measurement, plus the mono duplication where it
applies, plus that class's measured window-vs-file offset. The per-clip error left by
estimating the last term rather than metering every excerpt at runtime has σ ≈ 1.2 dB
against a target distribution σ ≈ 6 dB, so it widens the matched distribution by ~2% while
preserving each clip's natural relationship between window content and window level.
`reference_grain: file` selects the original whole-file behaviour for comparison.

Tiered by how many ensemble SOURCE FILES back a class: ≥ empirical_min_count → empirical
per-class sampling · ≥ class_mean_min_count → class mean with the pooled spread · below
that (or class absent from the ensemble pool) → the pooled ensemble distribution, with a
warning naming the class. Measured 2026-07-31, all nine exp002 classes sit in the empirical
tier (thinnest: 양금 at 73 stems), so the fallbacks are dormant insurance.

This is pool-level PRE-CONDITIONING applied BEFORE mixing, not per-stem normalization
inside a mixture: the finished mixture is still normalized as one tuple downstream by one
shared gain, so the project's "normalize the mixture, never per-stem" rule is untouched.
Matching COMPOSES with the random gain augmentation rather than replacing it — see
GugakMixDataset._draw_stem_gain for why that ordering matters.

Everything is off unless the experiment YAML says otherwise: an absent or empty
`gugak_mix.loudness_match` block means stock behaviour, byte-for-byte.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# BS.1770 sums per-channel energies with unit weights for L/R, so presenting a mono file
# as duplicated stereo doubles the measured energy — exactly this, never approximately
MONO_DUPLICATION_DB = 10.0 * math.log10(2.0)


# --- config -----------------------------------------------------------------
@dataclass
class LoudnessMatchConfig:
    """The `gugak_mix.loudness_match` block of an experiment YAML (paths repo-relative)."""
    # master switch: absent/empty block = disabled = stock random-gain behaviour
    enabled: bool = False
    # scale both sides are compared at: "chunk" (dataloader windows) | "file" (legacy)
    reference_grain: str = "chunk"
    # where targets come from — must match reference_grain
    reference_table: str = ("experiments/260731_chunk_loudness/"
                            "chunk_loudness_reference_10s.parquet")
    # per-file loudness of every ingested file, for looking up a drawn clip's own level
    source_table: str = ("experiments/260730_solo_pool_duration/"
                         "loudness_crest_scan.parquet")
    # which pool supplies targets, and which pools get matched to it
    reference_dataset: str = "71955"
    matched_datasets: list = field(default_factory=lambda: ["71470"])
    reference_split: str = "train"
    # draw shaping: uniform ±jitter smooths the discrete empirical values
    jitter_db: float = 0.5
    # tier thresholds — ensemble SOURCE FILES per class (not windows)
    empirical_min_count: int = 30
    class_mean_min_count: int = 10
    # gain guards: a huge boost lifts a quiet clip's noise floor into audibility. +25 dB
    # after the 2026-07-31 listening test found +20 dB boosts clean on studio headphones;
    # it cuts the clamp's class-correlated distortion from 0.44 dB spread to 0.09 dB
    max_gain_db: float = 25.0
    min_gain_db: float = -20.0
    # clips at/below this measured loudness have no meaningful level to match
    min_source_lufs_db: float = -70.0

    @classmethod
    def from_mapping(cls, mapping: dict) -> "LoudnessMatchConfig":
        """Build from a plain dict (yaml sub-block); unknown keys error loudly."""
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(mapping) - known
        if unknown:
            raise KeyError(f"unknown loudness_match config keys: {sorted(unknown)}")
        config = cls(**mapping)
        if config.reference_grain not in {"chunk", "file"}:
            raise ValueError(f"reference_grain must be 'chunk' or 'file', "
                             f"got {config.reference_grain!r}")
        return config


# --- per-class reference --------------------------------------------------------
@dataclass(frozen=True)
class ClassLoudnessReference:
    """How one class's target loudness is drawn.

    Args:
        tier: "empirical" | "class_mean" | "pooled" — which rule produced this.
        count: ensemble SOURCE FILES observed for the class (0 = absent from the pool).
        values: observed loudness to sample from; None for the class_mean tier.
        mean_db: class mean (class_mean tier) or pooled mean (fallback bookkeeping).
        spread_db: standard deviation used by the class_mean tier.
    """
    tier: str
    count: int
    values: np.ndarray | None
    mean_db: float
    spread_db: float


class LoudnessTargetSampler:
    """Draws ensemble-distributed loudness targets and turns them into stem gains.

    Built once at dataset init: reads the reference table, assigns every configured class
    a tier, learns the per-class window-vs-file offset for each matched pool, and keeps a
    file_id → measured loudness lookup. All randomness comes from the caller's Generator,
    so draws are reproducible under the run seed.

    Args:
        cfg: the `loudness_match` block.
        repo_root: absolute repo root the table paths resolve against.
        classes: the experiment's class list (every one gets a reference).
        segment_seconds: the training window, checked against a chunk-grain table's own
            window length so a stale reference can't be used silently.
    """

    def __init__(self, cfg: LoudnessMatchConfig, repo_root: Path, classes: list,
                 segment_seconds: float) -> None:
        self.cfg = cfg
        root = Path(repo_root)

        # file_id → measured whole-file loudness, for the clips this sampler re-levels
        source_scan = pd.read_parquet(root / cfg.source_table)
        self.source_lufs_by_file_id: dict = dict(
            zip(source_scan.file_id, source_scan.integrated_lufs.astype(float)))

        # reference population + (chunk grain) the offset that puts a clip on window scale
        reference_table = pd.read_parquet(root / cfg.reference_table)
        if cfg.reference_grain == "chunk":
            values, counts, self.window_offset_db = self._read_chunk_reference(
                reference_table, classes, segment_seconds)
        else:
            values, counts, self.window_offset_db = self._read_file_reference(
                reference_table, classes)
        self.references = self._build_references(values, counts, classes)

        # diagnostics — per process, read by scripts/verify_loudness_match.py
        self.counters: dict = {"matched": 0, "clamped": 0, "unmeasurable": 0}

    # --- init-time reference reading ---
    def _read_chunk_reference(self, table: pd.DataFrame, classes: list,
                              segment_seconds: float) -> tuple[dict, dict, dict]:
        """Ensemble window loudness per class, plus each matched pool's window offset.

        Args:
            table: the per-window table from src/data/chunk_loudness_scan.py.
            classes: the experiment's class list.
            segment_seconds: training window length, which the table must have been
                measured at — a mismatch means the reference describes a different scale.
        """
        window_lengths = set(table.window_seconds.unique())
        if window_lengths != {segment_seconds}:
            raise ValueError(
                f"{self.cfg.reference_table} was measured at window_seconds="
                f"{sorted(window_lengths)} but the experiment trains at "
                f"{segment_seconds}s — regenerate it with "
                f"`--window-seconds {segment_seconds:g}` (src/data/chunk_loudness_scan.py)")
        usable = table[table.measurable & (table.split == self.cfg.reference_split)]

        # targets: the ensemble pool's own measured window loudness
        ensemble = usable[(usable.dataset == self.cfg.reference_dataset)
                          & (usable.stem_group.isin(classes))]
        values = {name: group.window_lufs.to_numpy(dtype=float)
                  for name, group in ensemble.groupby("stem_group")}
        # tiers are about how many SOURCES a class has, not how many windows were drawn
        counts = {name: int(group.out_path.nunique())
                  for name, group in ensemble.groupby("stem_group")}

        # per matched pool and class: how far a window sits from its file's level
        offsets: dict = {}
        matched = usable[usable.dataset.isin(self.cfg.matched_datasets)]
        pooled_offset = float(matched.window_minus_source_db.mean()) if len(matched) else 0.0
        for (dataset_name, class_name), group in matched.groupby(["dataset", "stem_group"]):
            offsets[(dataset_name, class_name)] = float(group.window_minus_source_db.mean())
        for dataset_name in self.cfg.matched_datasets:
            for class_name in classes:
                if (dataset_name, class_name) not in offsets:
                    warnings.warn(
                        f"loudness_match: no {dataset_name} windows for class "
                        f"{class_name!r} in {self.cfg.reference_table} — using the pooled "
                        f"window offset {pooled_offset:+.2f} dB", stacklevel=3)
                    offsets[(dataset_name, class_name)] = pooled_offset
        return values, counts, offsets

    def _read_file_reference(self, table: pd.DataFrame,
                             classes: list) -> tuple[dict, dict, dict]:
        """Legacy whole-file reference: one value per ensemble stem, no scale correction.

        Kept selectable so the file-level and chunk-level variants can be compared; it
        reproduces the original behaviour exactly, mono duplication and windowing offset
        included (which is what made it wrong by ~2.75 dB).

        Args:
            table: the per-file scan (loudness_crest_scan).
            classes: the experiment's class list.
        """
        ensemble = table[(table.dataset == self.cfg.reference_dataset)
                         & (table.role != "master")
                         & (table.split == self.cfg.reference_split)
                         & (table.stem_group.isin(classes))
                         & np.isfinite(table.integrated_lufs)]
        values = {name: group.integrated_lufs.to_numpy(dtype=float)
                  for name, group in ensemble.groupby("stem_group")}
        counts = {name: int(len(group)) for name, group in ensemble.groupby("stem_group")}
        return values, counts, {}

    def _build_references(self, values: dict, counts: dict, classes: list) -> dict:
        """One ClassLoudnessReference per configured class, warning on every fallback.

        Args:
            values: class → observed reference loudness values.
            counts: class → how many ensemble source files back those values.
            classes: the experiment's class list.
        """
        pooled_values = (np.concatenate(list(values.values())) if values
                         else np.empty(0, dtype=float))
        if pooled_values.size == 0:
            raise ValueError(
                f"loudness_match: no reference rows (dataset={self.cfg.reference_dataset}, "
                f"split={self.cfg.reference_split}, classes={classes}) in "
                f"{self.cfg.reference_table}")
        pooled_mean, pooled_spread = float(pooled_values.mean()), float(pooled_values.std())

        references: dict = {}
        for class_name in classes:
            class_values = values.get(class_name, np.empty(0, dtype=float))
            count = int(counts.get(class_name, 0))
            if count >= self.cfg.empirical_min_count:
                references[class_name] = ClassLoudnessReference(
                    tier="empirical", count=count, values=class_values,
                    mean_db=float(class_values.mean()),
                    spread_db=float(class_values.std()))
            elif count >= self.cfg.class_mean_min_count:
                warnings.warn(
                    f"loudness_match: class {class_name!r} has only {count} ensemble "
                    f"stems (< {self.cfg.empirical_min_count}) — falling back to its "
                    "class mean with the pooled ensemble spread", stacklevel=2)
                references[class_name] = ClassLoudnessReference(
                    tier="class_mean", count=count, values=None,
                    mean_db=float(class_values.mean()), spread_db=pooled_spread)
            else:
                warnings.warn(
                    f"loudness_match: class {class_name!r} has only {count} ensemble "
                    f"stems (< {self.cfg.class_mean_min_count}) — falling back to the "
                    "POOLED ensemble loudness distribution", stacklevel=2)
                references[class_name] = ClassLoudnessReference(
                    tier="pooled", count=count, values=pooled_values,
                    mean_db=pooled_mean, spread_db=pooled_spread)
        return references

    # --- per-draw work ---
    def source_loudness(self, file_id: str) -> float:
        """Measured whole-file loudness of one ingested file (nan when unknown)."""
        return float(self.source_lufs_by_file_id.get(file_id, math.nan))

    def level_as_drawn(self, dataset_name: str, class_name: str, source_lufs: float,
                       source_channels: int) -> float:
        """A clip's own level on the same scale the reference is measured at.

        Under chunk grain that means the level of a drawn WINDOW of this clip: its file
        measurement, plus the mono→stereo duplication where the dataloader applies one,
        plus that pool-and-class's measured window-vs-file offset. Under file grain it is
        the file measurement unchanged (the legacy comparison).

        Args:
            dataset_name: the pool the clip comes from.
            class_name: the stem class being drawn.
            source_lufs: the clip's whole-file integrated loudness.
            source_channels: channels in the ingested file (1 = duplicated on draw).
        """
        if self.cfg.reference_grain != "chunk":
            return source_lufs
        duplication = MONO_DUPLICATION_DB if source_channels == 1 else 0.0
        return (source_lufs + duplication
                + self.window_offset_db.get((dataset_name, class_name), 0.0))

    def draw_target_lufs(self, rng: np.random.Generator, class_name: str) -> float:
        """Draw one target loudness for `class_name` from its ensemble reference.

        Empirical and pooled tiers pick an observed value and jitter it uniformly by
        ±jitter_db, so draws aren't restricted to the exact values measured; the
        class_mean tier draws from a normal instead, which is already continuous.

        Args:
            rng: the caller's Generator — the only source of randomness.
            class_name: the stem class being drawn.
        """
        reference = self.references[class_name]
        if reference.values is None:
            return float(rng.normal(reference.mean_db, reference.spread_db))
        observed = float(reference.values[rng.integers(reference.values.size)])
        return observed + float(rng.uniform(-self.cfg.jitter_db, self.cfg.jitter_db))

    def draw_gain(self, rng: np.random.Generator, class_name: str, dataset_name: str,
                  source_lufs: float, source_channels: int) -> float | None:
        """Linear gain putting one clip at a freshly drawn ensemble-like loudness.

        Returns None — consuming no randomness — when the clip has no measurable level to
        match (missing measurement, or near-silence where loudness is degenerate); the
        caller then falls back to the ordinary random-gain treatment rather than applying
        an absurd or infinite boost. Gains are clamped to [min_gain_db, max_gain_db]: the
        clamp costs a little match accuracy on the quietest clips and buys not amplifying
        their noise floor into a class cue.

        Args:
            rng: the caller's Generator.
            class_name: the stem class being drawn.
            dataset_name: the pool the clip comes from.
            source_lufs: the clip's whole-file integrated loudness.
            source_channels: channels in the ingested file.
        """
        if not math.isfinite(source_lufs) or source_lufs < self.cfg.min_source_lufs_db:
            self.counters["unmeasurable"] += 1
            return None
        own_level = self.level_as_drawn(dataset_name, class_name, source_lufs,
                                        source_channels)
        required_db = self.draw_target_lufs(rng, class_name) - own_level
        gain_db = float(np.clip(required_db, self.cfg.min_gain_db, self.cfg.max_gain_db))
        self.counters["matched"] += 1
        if gain_db != required_db:
            self.counters["clamped"] += 1
        return float(10.0 ** (gain_db / 20.0))

    # --- reporting ---
    def tier_summary(self) -> pd.DataFrame:
        """One row per class: tier, ensemble source count, reference mean/spread."""
        return pd.DataFrame([
            {"stem_group": name, "tier": ref.tier, "n_ensemble_stems": ref.count,
             "reference_mean_lufs": ref.mean_db, "reference_spread_db": ref.spread_db}
            for name, ref in self.references.items()])
