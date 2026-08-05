#!/usr/bin/env bash
# launch.sh — exp001.2: continue exp001 from its paused best checkpoint, in bf16.
#
#   tmux new -s exp001.2 'bash experiments/exp001.2_260802_htdemucs_9stem_resumed/launch.sh'
#
# Continues from exp001's epoch-33 best (val SI-SDR +1.0409, 85k optimizer steps) with
# optimizer + scheduler + epoch counter + metric history restored, so training resumes at
# epoch 34 exactly as if the shutdown had not happened. Only numerics changed: bf16
# autocast instead of pure fp32 (→ ../exp001.1_260730_bf16_probe/BF16_ADOPTION.md).
#
# ISOLATION: every output — checkpoints, log, provenance — lands in THIS folder. The
# exp001 folder is read exactly once, for the resume checkpoint, and never written to;
# its artifacts back the presentation and must stay untouched. Checkpoint filenames start
# at epoch 34, so they cannot collide with exp001's set even if the folders are merged later.
#
# STOP POLICY: SIGINT (Ctrl-C or `kill -INT <python pid>`) is an intentional stop — exit
# 130 ends this script rather than triggering the crash-resume. Real crashes get exactly
# one automatic resume from this run's own `last`, then stop with a CRASHED_TWICE marker.
set -u
cd /home/jae.gye/userdata/repos/gugak_stem_separation

EXP_DIR=experiments/exp001.2_260802_htdemucs_9stem_resumed
RESULTS=$EXP_DIR/checkpoints
LOG=$EXP_DIR/train.log
# read-only source: exp001's frozen best
RESUME_CKPT=${RESUME_CKPT:-experiments/exp001_260728_htdemucs_9stem/checkpoints/model_htdemucs_ep_33_si_sdr_1.0409.ckpt}
export PYTHONPATH=/home/jae.gye/userdata/repos/gugak_stem_separation
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p "$RESULTS"
git rev-parse HEAD > $EXP_DIR/git_commit.txt
git status --short >> $EXP_DIR/git_commit.txt
git -C external/msst rev-parse HEAD >> $EXP_DIR/git_commit.txt

COMMON_FLAGS=(
  --model_type htdemucs
  --config_path configs/exp001.2_htdemucs_9stem_resumed.yaml
  --results_path "$RESULTS"
  --data_path dummy
  --valid_path data/gugak_ensemble_71955/sumstem_9stem/val
  --num_workers 8 --pin_memory --persistent_workers
  --seed 42 --device_ids 0
  --loss l1_loss
  --metrics si_sdr --metric_for_scheduler si_sdr
  --wandb_offline
  --load_optimizer --load_scheduler --load_epoch
  --load_best_metric --load_all_metrics --load_all_losses
)

run_training() {
  uv run python external/msst/train.py "${COMMON_FLAGS[@]}" \
    --start_check_point "$1" 2>&1 | tee -a "$LOG"
  return "${PIPESTATUS[0]}"
}

echo "=== exp001.2 START $(date -Is) from $(basename "$RESUME_CKPT") (bf16) ===" | tee -a "$LOG"
run_training "$RESUME_CKPT"
code=$?
if [ "$code" -eq 0 ]; then
  echo "=== exp001.2 finished clean $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi
if [ "$code" -eq 130 ]; then
  echo "=== exp001.2 stopped by SIGINT $(date -Is) — intentional, not resuming ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp001.2 CRASHED (exit $code) $(date -Is) — resuming once from last ===" | tee -a "$LOG"
run_training "$RESULTS/last_htdemucs.ckpt"
code=$?
if [ "$code" -eq 0 ] || [ "$code" -eq 130 ]; then
  echo "=== exp001.2 ended after resume (exit $code) $(date -Is) ===" | tee -a "$LOG"
  exit 0
fi

echo "=== exp001.2 CRASHED TWICE (exit $code) $(date -Is) — stopping per policy ===" | tee -a "$LOG"
touch "$EXP_DIR/CRASHED_TWICE"
exit "$code"
