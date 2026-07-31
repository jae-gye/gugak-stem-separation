#!/usr/bin/env bash
# launch_exp001_resume.sh — resume exp001 from the pre-excursion best checkpoint.
# Run inside tmux from the repo root:
#   tmux new -s exp001 'bash experiments/exp001_260728_htdemucs_9stem/launch_exp001_resume.sh'
#
# Context (diagnosis 2026-07-29, → docs/exp001_overnight_report.md): epoch 10 hit a
# destabilization excursion (가야금/대금/양금 heads collapsed in one epoch). Recovery:
#   - weights from the ep9 best (si_sdr −1.546), NOT the damaged last
#   - FRESH optimizer/scheduler (Adam moments carry the excursion's momentum)
#   - lr halved to 5e-5 (config), grad clip 5.0 kept
#   - epoch counter / best metric / metric history preserved for continuity
# Crash policy unchanged: resume once from last, second crash stops (CRASHED_TWICE).
set -u
cd /home/jae.gye/userdata/repos/gugak_stem_separation

EXP_DIR=experiments/exp001_260728_htdemucs_9stem
RESULTS=$EXP_DIR/checkpoints
LOG=$EXP_DIR/train.log
BEST_CKPT=$RESULTS/model_htdemucs_ep_9_si_sdr_-1.5460.ckpt
export PYTHONPATH=/home/jae.gye/userdata/repos/gugak_stem_separation
export CUDA_VISIBLE_DEVICES=0

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
)

run_from_best() {
  # tolerant load unwraps model_state_dict; optimizer/scheduler deliberately FRESH
  uv run python external/msst/train.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$BEST_CKPT" \
    --load_epoch --load_best_metric --load_all_metrics --load_all_losses \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

run_resume_last() {
  uv run python external/msst/train.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$RESULTS/last_htdemucs.ckpt" \
    --load_optimizer --load_scheduler --load_epoch \
    --load_best_metric --load_all_metrics --load_all_losses \
    2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

echo "=== exp001 RESUME-FROM-BEST $(date -Is) (lr 5e-5, fresh optimizer) ===" | tee -a "$LOG"
run_from_best
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== exp001 finished clean $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp001 CRASHED (exit $code) $(date -Is) — resuming once from last ===" | tee -a "$LOG"
run_resume_last
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== exp001 finished clean after resume $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp001 CRASHED TWICE (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
touch "$EXP_DIR/CRASHED_TWICE"
exit "$code"
