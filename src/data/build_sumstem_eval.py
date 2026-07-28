"""build_sumstem_eval.py — render the frozen Σstem eval variant to disk, MSST-native.

For every song of the requested eval split, builds the controlled evaluation version:
class targets = sum of the song's stems per stem class, mixture = sum of those targets.
Output layout is exactly what MSST's valid.py consumes (`<song>/mixture.flac` +
`<song>/<class>.flac`), so validation needs zero custom eval code.

The frozen mixture recipe (Notion · Training Data Strategy / mixture recipe):
  1. take the song's stems belonging to the experiment's class scheme — quarantined
     classes (e.g. 편종·편경·방향 in exp001) never enter, keeping mixture ≡ Σ(targets)
  2. trim every stem to the song's shortest before summing (판소리 tail rule; identity
     for equal-length songs) — never pad
  3. same-base multi stems (피리1/2/3 …) land in one class target by construction
  4. peak-normalize the mixture to 0.99 and apply the SAME gain to every target —
     linear, so the training identity survives (never clip)
Absent classes get NO file: valid.py then scores each class only over songs that
contain it (MoisesDB-style filtering). Silent-target eval is a live design question
(→ Notion legal pad) — revisit here if it lands differently.

Mono stems are center-duplicated to stereo before summing. Output FLAC PCM_24 —
bit-transparent to the store, ~half the disk of wav.

Run:
    uv run python src/data/build_sumstem_eval.py --config configs/exp001_htdemucs_9stem.yaml
      --split val         eval split to render (val | test)
      --workers N         parallel song builders (default 8)
      --verify N          read back N random songs and check the identity (default 5)
      --overwrite         rebuild songs whose output already exists
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile
import yaml

from mix_dataset import MixDatasetConfig   # sibling module (scripts run from src/data)


# --- repo-root discovery (cwd-agnostic, mirrors ingest.py) ---
def find_root(start: Path | None = None) -> Path:
    """Locate the repo root (holds pyproject.toml)."""
    p = Path.cwd() if start is None else start
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists():
            return cand
    raise FileNotFoundError("repo root (pyproject.toml) not found above cwd")


# --- per-song builder (runs in worker processes) ---
def build_song(task: dict) -> dict:
    """Render one song's Σstem mixture + class targets.

    Args:
        task: song_id, out_dir, sample_rate, subtype, and `stems` — a list of
              (audio_path, frames, stem_group) for every included stem.
    """
    out_dir = Path(task["out_dir"])
    stems: list[tuple[str, int, str]] = task["stems"]

    # trim-to-shortest: read exactly min_frames from every stem (never pad)
    min_frames = min(frames for _, frames, _ in stems)

    class_targets: dict[str, np.ndarray] = {}
    for audio_path, _, stem_group in stems:
        audio, _ = soundfile.read(audio_path, frames=min_frames,
                                  dtype="float32", always_2d=True)
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)        # centered mono
        if stem_group in class_targets:
            class_targets[stem_group] = class_targets[stem_group] + audio
        else:
            class_targets[stem_group] = audio

    mixture = np.sum(list(class_targets.values()), axis=0)

    # peak-norm mixture to 0.99, SAME linear gain on every target (identity-preserving)
    peak = float(np.abs(mixture).max())
    gain = np.float32(0.99 / peak) if peak > 0 else np.float32(1.0)
    mixture = mixture * gain

    out_dir.mkdir(parents=True, exist_ok=True)
    soundfile.write(out_dir / f"mixture.{task['extension']}", mixture,
                    task["sample_rate"], subtype=task["subtype"])
    for class_name, target in class_targets.items():
        soundfile.write(out_dir / f"{class_name}.{task['extension']}", target * gain,
                        task["sample_rate"], subtype=task["subtype"])

    return {"song_id": task["song_id"], "n_classes": len(class_targets),
            "seconds": min_frames / task["sample_rate"]}


# --- verification (read back what was written) ---
def verify_songs(out_root: Path, song_ids: list[str], extension: str,
                 tolerance: float = 1e-4) -> None:
    """Check mixture ≡ Σ(class files) within PCM_24 quantization tolerance."""
    for song_id in song_ids:
        song_dir = out_root / song_id
        mixture, _ = soundfile.read(song_dir / f"mixture.{extension}", dtype="float32")
        total = np.zeros_like(mixture)
        for stem_file in sorted(song_dir.glob(f"*.{extension}")):
            if stem_file.stem == "mixture":
                continue
            total += soundfile.read(stem_file, dtype="float32")[0]
        error = float(np.abs(mixture - total).max())
        peak = float(np.abs(mixture).max())
        status = "OK" if error < tolerance else "FAIL"
        print(f"  {status}  {song_id}: max|mixture - Σtargets| = {error:.2e}, "
              f"peak = {peak:.3f}")
        if error >= tolerance:
            raise RuntimeError(f"{song_id}: Σstem identity broken (error {error:.2e})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the Σstem eval variant (MSST layout).")
    ap.add_argument("--config", default="configs/exp001_htdemucs_9stem.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = find_root()
    cfg = yaml.safe_load((root / args.config).read_text())
    mix_cfg = MixDatasetConfig.from_mapping(cfg["gugak_mix"])   # resolves defaults
    classes: list[str] = list(mix_cfg.classes)
    eval_cfg = cfg["sumstem_eval"]
    out_root = root / eval_cfg["out_root"] / args.split
    extension, subtype = eval_cfg["extension"], eval_cfg["subtype"]
    sample_rate = mix_cfg.sample_rate

    # song list + included stems from the one manifest dataloaders read
    manifest = pd.read_parquet(root / mix_cfg.source_manifest)
    stems = manifest[(manifest.dataset == "71955") & (manifest.split == args.split)
                     & (manifest.role != "master") & (manifest.stem_group.isin(classes))]

    tasks = []
    skipped = 0
    for song_id, group in stems.groupby("song_id"):
        song_dir = out_root / str(song_id)
        if not args.overwrite and (song_dir / f"mixture.{extension}").exists():
            skipped += 1
            continue
        tasks.append({
            "song_id": str(song_id), "out_dir": str(song_dir),
            "sample_rate": sample_rate, "extension": extension, "subtype": subtype,
            "stems": [(str(root / r.out_path), int(r.out_frames), str(r.stem_group))
                      for r in group.itertuples()],
        })
    print(f"split={args.split}: {stems.song_id.nunique()} songs · "
          f"{len(stems)} stems -> {len(tasks)} to build ({skipped} already exist)")

    results = []
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(build_song, t) for t in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if i % 10 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)}")

    if results:
        frame = pd.DataFrame(results)
        print(f"\nbuilt {len(frame)} songs · {frame.seconds.sum() / 3600:.2f} h · "
              f"classes per song: median {frame.n_classes.median():.0f} "
              f"(min {frame.n_classes.min()}, max {frame.n_classes.max()})")

    if args.verify > 0:
        song_dirs = sorted(d.name for d in out_root.iterdir() if d.is_dir())
        rng = np.random.default_rng(42)
        sample = list(rng.choice(song_dirs, size=min(args.verify, len(song_dirs)),
                                 replace=False))
        print(f"\nverifying {len(sample)} songs:")
        verify_songs(out_root, sample, extension)
    print(f"\ndone -> {out_root}")


if __name__ == "__main__":
    main()
