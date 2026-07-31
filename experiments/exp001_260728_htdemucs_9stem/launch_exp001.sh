#!/usr/bin/env bash
# launch_exp001.sh — exp001 training with a built-in resume-once supervisor.
# Run inside tmux from the repo root:  tmux new -s exp001 'bash experiments/exp001_260728_htdemucs_9stem/launch_exp001.sh'
#
# Policy (overnight brief): if training crashes, resume ONCE from the latest
# checkpoint; a second crash stops everything and leaves CRASHED_TWICE as a marker.
set -u
cd /home/jae.gye/userdata/repos/gugak_stem_separation

EXP_DIR=experiments/exp001_260728_htdemucs_9stem
RESULTS=$EXP_DIR/checkpoints
LOG=$EXP_DIR/train.log
export PYTHONPATH=/home/jae.gye/userdata/repos/gugak_stem_separation
export CUDA_VISIBLE_DEVICES=0

# provenance: exact commit + dirty-file list alongside the run artifacts
git rev-parse HEAD > $EXP_DIR/git_commit.txt
git status --short >> $EXP_DIR/git_commit.txt

# flags shared by fresh start and resume
COMMON_FLAGS=(
  --model_type htdemucs
  --config_path configs/exp001_htdemucs_9stem.yaml
  --results_path "$RESULTS"
  --data_path dummy
  --valid_path data/gugak_ensemble_71955/sumstem_9stem/val
  --num_workers 8 --pin_memory --persistent_workers
  --seed 42 --device_ids 0
  --loss l1_loss
  --metrics si_sdr --metric_for_scheduler si_sdr
  --wandb_offline
)

run_fresh() {
  uv run python external/msst/train.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$RESULTS/start_checkpoint.ckpt" \
    --load_only_compatible_weights \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

run_resume() {
  # tolerant load unwraps model_state_dict from the full last-checkpoint dict
  uv run python external/msst/train.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$RESULTS/last_htdemucs.ckpt" \
    --load_optimizer --load_scheduler --load_epoch \
    --load_best_metric --load_all_metrics --load_all_losses \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

echo "=== exp001 launch $(date -Is) ===" | tee -a "$LOG"
run_fresh
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== exp001 finished clean $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp001 CRASHED (exit $code) $(date -Is) — resuming once ===" | tee -a "$LOG"
if [ -f "$RESULTS/last_htdemucs.ckpt" ]; then
  run_resume
else
  echo "(no last checkpoint yet -> fresh restart counts as the one resume)" | tee -a "$LOG"
  run_fresh
fi
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== exp001 finished clean after resume $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp001 CRASHED TWICE (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
touch "$EXP_DIR/CRASHED_TWICE"
exit "$code"
