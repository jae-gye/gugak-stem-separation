#!/usr/bin/env bash
# launch_exp001_continue.sh — resume exp001 exactly where the pre-shutdown pause left it.
#
#   tmux new -s exp001 'bash experiments/exp001_260728_htdemucs_9stem/launch_exp001_continue.sh'
#
# Continues from the ep33 best checkpoint (val SI-SDR +1.0409, 85k optimizer steps) with
# optimizer + scheduler + epoch counter + metric history all restored, so training picks
# up at epoch 34 as if never interrupted. This is the OPPOSITE of launch_exp001_resume.sh,
# which deliberately discards optimizer state (that one was the post-cliff recovery).
#
# Stop policy: SIGINT (Ctrl-C or `kill -INT <python pid>`) is treated as an INTENTIONAL
# stop — exit 130 ends this script instead of triggering the crash-resume. That mistake
# cost us a relaunch during the 2026-07-31 pause; never again. Genuine crashes still get
# exactly one automatic resume from `last`, then stop with a CRASHED_TWICE marker.
set -u
cd /home/jae.gye/userdata/repos/gugak_stem_separation

EXP_DIR=experiments/exp001_260728_htdemucs_9stem
RESULTS=$EXP_DIR/checkpoints
LOG=$EXP_DIR/train.log
RESUME_CKPT=${RESUME_CKPT:-$RESULTS/model_htdemucs_ep_33_si_sdr_1.0409.ckpt}
export PYTHONPATH=/home/jae.gye/userdata/repos/gugak_stem_separation
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

git rev-parse HEAD > $EXP_DIR/git_commit.txt
git status --short >> $EXP_DIR/git_commit.txt

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
  # full state restore — the point of this script
  --load_optimizer --load_scheduler --load_epoch
  --load_best_metric --load_all_metrics --load_all_losses
)

run_training() {
  uv run python external/msst/train.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$1" 2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

echo "=== exp001 CONTINUE $(date -Is) from $(basename "$RESUME_CKPT") ===" | tee -a "$LOG"
run_training "$RESUME_CKPT"
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== exp001 finished clean $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi
if [ "$code" -eq 130 ]; then
  echo "=== exp001 stopped by SIGINT $(date -Is) — intentional, not resuming ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp001 CRASHED (exit $code) $(date -Is) — resuming once from last ===" | tee -a "$LOG"
run_training "$RESULTS/last_htdemucs.ckpt"
code=$?
if [ "$code" -eq 0 ] || [ "$code" -eq 130 ]; then
  echo "=== exp001 ended after resume (exit $code) $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp001 CRASHED TWICE (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
touch "$EXP_DIR/CRASHED_TWICE"
exit "$code"
