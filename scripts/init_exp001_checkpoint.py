"""init_exp001_checkpoint.py — build exp001's warm-start checkpoint (spec: model init).

Instantiates the 9-source HTDemucs exactly as MSST training will (same config, same
param names), then transfers every name+shape-compatible parameter from the official
pretrained `htdemucs` checkpoint (4 Western sources, via the demucs package). Layers
whose shape depends on the source count — the final source-splitting layers of both
branches — keep their fresh initialization. Logs exactly which parameter groups
loaded vs re-initialized (spec requirement) to stdout AND a log file.

Output is a PLAIN state_dict: MSST's `--load_only_compatible_weights` path performs a
strict `model.load_state_dict(torch.load(path))`, so training starts from precisely
this tensor set — no tolerant-load ambiguity.

Run:
    uv run python scripts/init_exp001_checkpoint.py
      --config configs/exp001_htdemucs_9stem.yaml
      --out experiments/exp001_260728_htdemucs_9stem/checkpoints/start_checkpoint.ckpt
      --log experiments/exp001_260728_htdemucs_9stem/checkpoint_init_log.txt
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "msst"))


def group_of(param_name: str) -> str:
    """Collapse a parameter name to its reporting group (e.g. 'decoder.3')."""
    parts = param_name.split(".")
    return ".".join(parts[:2]) if len(parts) > 1 and parts[1].isdigit() else parts[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Warm-start checkpoint for exp001.")
    ap.add_argument("--config", default="configs/exp001_htdemucs_9stem.yaml")
    ap.add_argument("--out",
                    default="experiments/exp001_260728_htdemucs_9stem/checkpoints/"
                            "start_checkpoint.ckpt")
    ap.add_argument("--log",
                    default="experiments/exp001_260728_htdemucs_9stem/"
                            "checkpoint_init_log.txt")
    args = ap.parse_args()

    # target: the 9-source model, built by MSST itself -> param names match training
    from utils.settings import get_model_from_config
    target_model, config = get_model_from_config("htdemucs",
                                                 str(REPO_ROOT / args.config))
    target_state = target_model.state_dict()

    # source: official pretrained htdemucs (4 sources) from the demucs package
    from demucs.pretrained import get_model as get_pretrained
    source_model = get_pretrained("htdemucs").models[0]
    source_state = source_model.state_dict()

    # transfer everything name+shape-compatible; classify the rest
    transferred, reinitialized, absent_in_source = [], [], []
    for name, tensor in target_state.items():
        if name in source_state and source_state[name].shape == tensor.shape:
            target_state[name] = source_state[name].clone()
            transferred.append(name)
        elif name in source_state:
            reinitialized.append(
                f"{name}  target{tuple(tensor.shape)} vs "
                f"pretrained{tuple(source_state[name].shape)}")
        else:
            absent_in_source.append(name)

    # report by parameter group (spec: log loaded vs re-initialized groups)
    def summarize(names: list[str]) -> dict:
        counts: dict = defaultdict(int)
        for n in names:
            counts[group_of(n.split("  ")[0])] += 1
        return dict(sorted(counts.items()))

    n_params_total = sum(v.numel() for v in target_state.values())
    n_params_moved = sum(target_state[n].numel() for n in transferred)
    lines = [
        f"instruments ({len(config.training.instruments)}): "
        f"{list(config.training.instruments)}",
        f"pretrained source model: demucs 'htdemucs' "
        f"({list(source_model.sources)})",
        f"tensors: {len(target_state)} total | {len(transferred)} transferred | "
        f"{len(reinitialized)} shape-mismatch (fresh init) | "
        f"{len(absent_in_source)} absent in pretrained (fresh init)",
        f"parameters transferred: {n_params_moved:,} / {n_params_total:,} "
        f"({100 * n_params_moved / n_params_total:.1f}%)",
        "", "TRANSFERRED groups (count):",
        *(f"  {g}: {c}" for g, c in summarize(transferred).items()),
        "", "RE-INITIALIZED groups — shape mismatch (count):",
        *(f"  {g}: {c}" for g, c in summarize(reinitialized).items()),
        "", "RE-INITIALIZED shape-mismatch detail:",
        *(f"  {d}" for d in reinitialized),
        "", "ABSENT in pretrained (fresh init):",
        *(f"  {n}" for n in absent_in_source),
    ]
    report = "\n".join(lines)
    print(report)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target_state, out_path)
    log_path = REPO_ROOT / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(report + "\n")
    print(f"\nwrote {out_path}\nlog   {log_path}")


if __name__ == "__main__":
    main()
