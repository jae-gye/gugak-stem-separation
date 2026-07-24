# experiments/

Per-experiment home for **light, tracked** artifacts: metric tables + notes. Heavy audio
(mixtures, model predictions) stays under `data/_val/` (storage symlink, gitignored) and is
**never** copied here.

## Folder naming
Every experiment folder carries a `YYMMDD` date stamp:
- **Named experiments:** `<name>_<date>` — e.g. `zeroshot_baseline_260719`.
- **Numbered experiments:** `exp<NNN>_<date>_<rest>` — date goes *after* the number, before the
  run descriptor. e.g. `exp001_260722_htdemucs_4stem`.

(The `exp<NNN>_<model>_<stemscheme>_<key-hparam>` run-naming convention is in the root CLAUDE.md;
the date is inserted right after `exp<NNN>`.)

## What lives in each experiment folder
```
<exp>/
  metrics.parquet   # source of truth for figures  → TRACKED
  metrics.csv       # diff-able mirror              → TRACKED
  README.md         # setup + headline findings     → TRACKED
  figures/          # generated pngs                → GITIGNORED (regenerable)
```

## Tracking policy
- **Tracked:** `*.parquet`, `*.csv`, `*.md` — small, and the true inputs to every chart.
- **Not tracked:** `*.png` (and any figure image). They regenerate from the tracked metrics +
  the plotting script, so committing them would just bloat git with opaque binaries.
- **To reproduce a figure:** re-run its script (e.g. `uv run python scripts/fig_zeroshot.py`) —
  reads the tracked `metrics.parquet`, writes into `<exp>/figures/`.
