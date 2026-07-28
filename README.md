# gugak-stem-separation

Stem separation on a traditional Korean music (gugak) multitrack dataset
(AI Hub **국악합주곡 디지털 음원 데이터**, datasetkey 71955). See `CLAUDE.md` for
the full project brief, conventions, and dataset gotchas.

## Environment (uv)

```bash
uv sync          # create/refresh .venv from pyproject.toml + uv.lock
uv run <cmd>     # run a command inside the env (no manual activate needed)
```

## Layout

```
configs/     one YAML per experiment
src/data/    manifest building, dataset class, mixing
src/models/  MSST wrappers (later)
scripts/     data acquisition / extraction one-offs
notebooks/   EDA only
manifests/   source-of-truth tables: parquet/ (committed) + csv/ eyeball twins (mostly disk-only)
data/        -> symlink to /home/jae.gye/storage/nia-gugak (dataset, not tracked)
```
