#!/usr/bin/env bash
# launch.sh — exp001.1_bf16_probe: bf16 numerics probe on GPU 1.
# Run inside tmux from the repo root:
#   tmux new -s exp001_bf16_probe 'bash experiments/exp001.1_260730_bf16_probe/launch.sh'
#
# Probe, not a training run (spec → Notion EXP001, fp16 cliff findings):
#   Q1 does bf16 survive the ~−1.5 dB danger zone that killed fp16 twice?
#   Q2 what speed/memory does bf16 buy vs the fp32 arm?
# Mirrors launch_exp001_resume.sh exactly (same ep9 best checkpoint READ-ONLY, same
# fresh-optimizer resume flags, same seed/loader settings) except:
#   - GPU 1 (GPU 0 = the running fp32 arm — untouchable; GPU 2 reserved for seed twin)
#   - config configs/exp001.1_bf16_probe.yaml (precision block only diff)
#   - checkpoints/logs under experiments/exp001.1_260730_bf16_probe/ only
#   - NO crash-resume supervisor: a probe crash is a finding, preserve and stop.
set -u
cd /home/jae.gye/userdata/repos/gugak_stem_separation

RUN_DIR=experiments/exp001.1_260730_bf16_probe
RESULTS=$RUN_DIR/checkpoints
LOG=$RUN_DIR/train.log
BEST_CKPT=experiments/exp001_260728_htdemucs_9stem/checkpoints/model_htdemucs_ep_9_si_sdr_-1.5460.ckpt
export PYTHONPATH=/home/jae.gye/userdata/repos/gugak_stem_separation
export CUDA_VISIBLE_DEVICES=1

mkdir -p "$RESULTS"

# provenance: exact commit + dirty-file list alongside the run artifacts
git rev-parse HEAD > $RUN_DIR/git_commit.txt
git status --short >> $RUN_DIR/git_commit.txt

# GPU-memory sampler: GPU 1 usage every 30 s for the Q2 peak-memory number
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 30 -i 1 \
  > $RUN_DIR/gpu_mem.csv &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null' EXIT

echo "=== exp001.1_bf16_probe launch $(date -Is) ===" | tee -a "$LOG"
uv run python external/msst/train.py \
  --model_type htdemucs \
  --config_path configs/exp001.1_bf16_probe.yaml \
  --results_path "$RESULTS" \
  --data_path dummy \
  --valid_path data/gugak_ensemble_71955/sumstem_9stem/val \
  --num_workers 8 --pin_memory --persistent_workers \
  --seed 42 --device_ids 0 \
  --loss l1_loss \
  --metrics si_sdr --metric_for_scheduler si_sdr \
  --wandb_offline \
  --start_check_point "$BEST_CKPT" \
  --load_epoch --load_best_metric --load_all_metrics --load_all_losses \
  2>&1 | tee -a "$LOG"
code=${PIPESTATUS[0]}

echo "=== exp001.1_bf16_probe exited (code $code) $(date -Is) ===" | tee -a "$LOG"
exit "$code"
