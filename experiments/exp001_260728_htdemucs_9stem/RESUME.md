# exp001 — how to resume after the 2026-08-01 shutdown

Paused 2026-07-31 18:51 KST, mid-improvement, for the sym8 maintenance power-off.
Metrics snapshot → `metrics/SUMMARY.md`. Full history → `docs/exp001_overnight_report.md`.

## One command

```bash
tmux new -s exp001 'bash experiments/exp001_260728_htdemucs_9stem/launch_exp001_continue.sh'
```

Runs on GPU 0 by default; override with `CUDA_VISIBLE_DEVICES=1 tmux new -s exp001 '...'`.
Verify a free card with `nvidia-smi` first (shared machine).

## What it resumes from — verified 2026-07-31

`checkpoints/model_htdemucs_ep_33_si_sdr_1.0409.ckpt` — the best checkpoint, whose
weights are **byte-identical** to `last_htdemucs.ckpt` (both are post-epoch-33 training);
it additionally carries the epoch-33 eval in its metric history, so it is the better
resume source.

| Field | Value |
|---|---|
| epoch recorded | 33 → training restarts at **epoch 34** |
| learning rate | **2.5e-5** (ReduceLROnPlateau had halved it from 5e-5) |
| optimizer state | 533/533 tensors present (Adam moments intact) |
| scheduler state | present (`best` 0.8779, `num_bad_epochs` 0) |
| model tensors | 533, zero NaN/Inf |
| metric history | 34 eval cycles |

**Live-tested on 2026-07-31** (a ~1 h trial run on GPU 2, then stopped): it resumed at
epoch 34 with LR 2.5e-5 and training loss ~0.00483 — matching the 0.00486 from just
before the pause, which is the proof that optimizer state really was restored rather
than silently re-initialised. No checkpoint was overwritten by the trial (saves happen
only at epoch end).

## Sanity checks on restart

1. Log prints `Train epoch: 34 Learning rate: 2.5e-05`.
2. Loss starts near **0.0048**. A jump to ~0.0065+ means optimizer state did NOT load —
   stop and investigate rather than let it run.
3. First eval should land near **+1.04** (it was still climbing ~+0.16 dB/cycle).
4. Per-class tripwire: any class dropping >5 dB in one eval ⇒ stop and escalate
   (see the instability history in the overnight report).

## Known bookkeeping quirk

Both checkpoints record `best_metric = 0.8779` — MSST saves the *previous* best inside
the file it is currently writing. On resume the bar therefore loads as 0.8779, so the
first eval above that value gets filed as "best" even if it is below 1.0409. Cosmetic
only; the real bar to beat is **+1.0409**, and the metric history in the checkpoint has
the true values.

## Stop it properly

```bash
kill -INT <python pid>      # or Ctrl-C in the tmux pane
```

`launch_exp001_continue.sh` treats exit 130 (SIGINT) as an intentional stop and will
**not** relaunch. (The older `launch_exp001_resume.sh` does not — it read the 2026-07-31
pause SIGINT as a crash and restarted training. Use the continue script.)

Genuine crashes still get exactly one automatic resume from `last_htdemucs.ckpt`; a
second crash stops and leaves a `CRASHED_TWICE` marker file.

## Other checkpoints in this folder

| File | What it is |
|---|---|
| `model_htdemucs_ep_33_si_sdr_1.0409.ckpt` | **best — resume from this** |
| `last_htdemucs.ckpt` | same weights, shorter metric history |
| `model_htdemucs_ep_*.ckpt` | earlier bests, kept for the trajectory |
| `damaged_ep11_*.ckpt` | the two fp16 collapse states, kept as forensic evidence |
| `start_checkpoint.ckpt` | the original 9-source warm start from pretrained htdemucs |
