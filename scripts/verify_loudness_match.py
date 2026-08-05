"""verify_loudness_match.py — does solo-pool loudness matching actually close the gap?

Three checks, none of which touch a GPU, a manifest, or the ingest store's contents:
  1. TIERS — which rule each class's targets come from, and off how many ensemble stems
  2. TARGETS — drawn targets vs the ensemble reference values, per class (mean, σ, KS)
  3. AUDIO — measured loudness of real drawn excerpts, solo-sourced (matching ON) vs
     ensemble-sourced, per class, before and after each pool's full gain treatment

Check 3 is the honest one: it builds two datasets from the same experiment config —
one restricted to the ensemble pool, one to the solo pool with matching enabled — and
pushes real excerpts through the real draw path (`_draw_excerpt` → `_augment_stem`), so
what it measures is what the trainer would see, random gain augmentation included.
Matching means the two POPULATIONS agree, not that every excerpt is pinned to one value:
each side keeps its own within-pool variation, and 3b's per-class spread — not the mean
alone — is where a class-correlated cue would show up.

No config file is modified: the solo pool and the loudness_match block are injected
into an in-memory copy of the YAML.

Run:
    uv run python scripts/verify_loudness_match.py
      --config configs/exp001_htdemucs_9stem.yaml   base geometry + class list
      --draws 150        excerpts per class per pool (check 3; the slow part)
      --targets 20000    synthetic target draws per class (check 2)
      --seed 7
      --out <path.parquet>   optional: write the per-class table
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyloudnorm
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.mix_dataset import GugakMixDataset, MixDatasetConfig  # noqa: E402


# --- statistics -------------------------------------------------------------
def ks_two_sample(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov statistic and its asymptotic p-value.

    Hand-rolled rather than pulled from scipy, which this project does not declare as
    a dependency. Large p = the two samples are consistent with one distribution.

    Args:
        left: first sample.
        right: second sample.
    """
    left, right = np.sort(np.asarray(left)), np.sort(np.asarray(right))
    grid = np.concatenate([left, right])
    cdf_left = np.searchsorted(left, grid, side="right") / left.size
    cdf_right = np.searchsorted(right, grid, side="right") / right.size
    statistic = float(np.abs(cdf_left - cdf_right).max())

    effective_n = np.sqrt(left.size * right.size / (left.size + right.size))
    scaled = (effective_n + 0.12 + 0.11 / effective_n) * statistic
    terms = np.arange(1, 101)
    p_value = float(np.clip(
        2.0 * np.sum((-1.0) ** (terms - 1) * np.exp(-2.0 * terms ** 2 * scaled ** 2)),
        0.0, 1.0))
    return statistic, p_value


# --- dataset construction ---------------------------------------------------
def build_dataset(raw_config: dict, datasets: list, matching: bool) -> GugakMixDataset:
    """Dataset restricted to one source pool, with loudness matching on or off.

    Args:
        raw_config: parsed experiment YAML (copied, never written back).
        datasets: the source pools to draw from, e.g. ["71470"].
        matching: whether to enable the loudness_match block.
    """
    block = dict(raw_config["gugak_mix"])
    block["datasets"] = list(datasets)
    block["loudness_match"] = {"enabled": bool(matching)}
    return GugakMixDataset(MixDatasetConfig.from_mapping(block), REPO_ROOT, num_items=0)


def measure_excerpts(dataset: GugakMixDataset, class_name: str, draws: int,
                     seed: int, meter: pyloudnorm.Meter
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Loudness of `draws` real excerpts of one class, before and after gain treatment.

    "Before" is the excerpt at its native level — the thing the ensemble reference
    distribution describes. "After" is what the mixer receives: matched loudness for
    solo-sourced stems, native level × random gain for ensemble ones.

    Args:
        dataset: the dataset whose pool and gain treatment are exercised.
        class_name: the stem class to draw.
        draws: how many excerpts to pull.
        seed: base seed — each draw gets its own reproducible stream.
        meter: the loudness meter (44.1 kHz).
    """
    native, treated = [], []
    for index in range(draws):
        rng = np.random.default_rng([seed, index])
        excerpt, entry = dataset._draw_excerpt(rng, class_name)
        stem = dataset._augment_stem(rng, excerpt, class_name, entry)
        before, after = (meter.integrated_loudness(excerpt.T),
                         meter.integrated_loudness(stem.T))
        if np.isfinite(before) and np.isfinite(after):
            native.append(before)
            treated.append(after)
    return np.asarray(native, dtype=float), np.asarray(treated, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify solo-pool loudness matching against the ensemble pool.")
    ap.add_argument("--config", default="configs/exp001_htdemucs_9stem.yaml")
    ap.add_argument("--draws", type=int, default=150,
                    help="excerpts measured per class per pool")
    ap.add_argument("--targets", type=int, default=20000,
                    help="synthetic target draws per class")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="optional parquet for the summary table")
    args = ap.parse_args()

    raw = yaml.safe_load((REPO_ROOT / args.config).read_text())
    classes = list(raw["gugak_mix"]["classes"])
    ensemble_pool = build_dataset(raw, ["71955"], matching=False)
    solo_pool = build_dataset(raw, ["71470"], matching=True)
    sampler = solo_pool.loudness_sampler
    meter = pyloudnorm.Meter(raw["gugak_mix"]["sample_rate"])

    # --- check 1: tier assignment ---
    print("=== 1. target tiers (which rule each class draws from) ===")
    tiers = sampler.tier_summary().set_index("stem_group").loc[classes]
    print(tiers.round(2).to_string())
    fallbacks = tiers[tiers.tier != "empirical"]
    print(f"\nclasses on a fallback tier: "
          f"{list(fallbacks.index) if len(fallbacks) else 'none'}")

    # --- check 2: drawn targets vs the ensemble reference ---
    print(f"\n=== 2. drawn targets vs ensemble reference ({args.targets:,}/class) ===")
    print(f"{'class':<8} {'ref n':>6} {'ref mean':>9} {'ref σ':>6} "
          f"{'draw mean':>10} {'draw σ':>7} {'KS':>6} {'p':>7}")
    target_rows = []
    for slot, class_name in enumerate(classes):
        rng = np.random.default_rng([args.seed, slot])
        drawn = np.array([sampler.draw_target_lufs(rng, class_name)
                          for _ in range(args.targets)])
        reference = sampler.references[class_name].values
        statistic, p_value = ks_two_sample(drawn, reference)
        print(f"{class_name:<8} {reference.size:>6d} {reference.mean():>9.2f} "
              f"{reference.std():>6.2f} {drawn.mean():>10.2f} {drawn.std():>7.2f} "
              f"{statistic:>6.3f} {p_value:>7.3f}")
        target_rows.append({"stem_group": class_name, "reference_n": int(reference.size),
                            "reference_mean_lufs": float(reference.mean()),
                            "reference_std_db": float(reference.std()),
                            "target_mean_lufs": float(drawn.mean()),
                            "target_std_db": float(drawn.std()),
                            "target_ks": statistic, "target_ks_p": p_value})

    # --- check 3: real excerpts through the real draw path ---
    started, measured = time.time(), {}
    for class_name in classes:
        measured[class_name] = (
            measure_excerpts(ensemble_pool, class_name, args.draws, args.seed, meter)
            + measure_excerpts(solo_pool, class_name, args.draws, args.seed + 1, meter))

    print(f"\n=== 3a. untreated levels: the gap matching has to close "
          f"({args.draws} excerpts/class/pool) ===")
    print("both pools' windows at their NATIVE level, before any gain")
    print(f"{'class':<8} {'ensemble':>9} {'σ':>6} {'solo':>9} {'σ':>6} {'raw gap':>8}")
    audio_rows = []
    for class_name in classes:
        ensemble_native, ensemble_treated, solo_native, solo_treated = measured[class_name]
        print(f"{class_name:<8} {ensemble_native.mean():>9.2f} "
              f"{ensemble_native.std():>6.2f} {solo_native.mean():>9.2f} "
              f"{solo_native.std():>6.2f} "
              f"{solo_native.mean() - ensemble_native.mean():>+8.2f}")
        statistic, p_value = ks_two_sample(solo_treated, ensemble_treated)
        audio_rows.append({"stem_group": class_name,
                           "ensemble_native_mean_lufs": float(ensemble_native.mean()),
                           "ensemble_native_std_db": float(ensemble_native.std()),
                           "ensemble_treated_mean_lufs": float(ensemble_treated.mean()),
                           "ensemble_treated_std_db": float(ensemble_treated.std()),
                           "solo_native_mean_lufs": float(solo_native.mean()),
                           "solo_native_gap_db": float(solo_native.mean()
                                                       - ensemble_native.mean()),
                           "solo_treated_mean_lufs": float(solo_treated.mean()),
                           "solo_treated_std_db": float(solo_treated.std()),
                           "solo_treated_gap_db": float(solo_treated.mean()
                                                        - ensemble_treated.mean()),
                           "excerpt_ks": statistic, "excerpt_ks_p": p_value})
    audio = pd.DataFrame(audio_rows)

    print("\n=== 3b. what the mixer actually receives — the test that matters ===")
    print("both pools after their FULL gain treatment: ensemble = random gain · "
          "solo = matching composed with the same random gain. A model that could read "
          "the pool off the level would need these to differ.")
    print(f"{'class':<8} {'ensemble':>9} {'σ':>6} {'solo':>9} {'σ':>6} {'gap':>7} "
          f"{'KS':>6} {'p':>7}")
    for row in audio.itertuples():
        print(f"{row.stem_group:<8} {row.ensemble_treated_mean_lufs:>9.2f} "
              f"{row.ensemble_treated_std_db:>6.2f} {row.solo_treated_mean_lufs:>9.2f} "
              f"{row.solo_treated_std_db:>6.2f} {row.solo_treated_gap_db:>+7.2f} "
              f"{row.excerpt_ks:>6.3f} {row.excerpt_ks_p:>7.3f}")
    print(f"\nmean |gap| across classes: untreated "
          f"{audio.solo_native_gap_db.abs().mean():.2f} dB → treated "
          f"{audio.solo_treated_gap_db.abs().mean():.2f} dB "
          f"(mean signed {audio.solo_treated_gap_db.mean():+.2f} dB)")
    print(f"per-class spread of the gap: untreated "
          f"{audio.solo_native_gap_db.max() - audio.solo_native_gap_db.min():.2f} dB → "
          f"treated {audio.solo_treated_gap_db.max() - audio.solo_treated_gap_db.min():.2f}"
          f" dB   (this is the class-correlated cue)")
    print(f"mean σ across classes: ensemble {audio.ensemble_treated_std_db.mean():.2f} dB"
          f" · solo {audio.solo_treated_std_db.mean():.2f} dB")

    # --- counters ---
    counters = sampler.counters
    clamp_rate = counters["clamped"] / max(counters["matched"], 1)
    print(f"sampler counters: {counters} (clamp rate {clamp_rate:.1%})")
    print(f"elapsed {time.time() - started:.0f} s")

    if args.out:
        table = pd.DataFrame(target_rows).merge(audio, on="stem_group")
        table = table.merge(tiers.reset_index(), on="stem_group")
        table.to_parquet(REPO_ROOT / args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
