# bf16 ADOPTED as the project default — decision record, 2026-08-02

**This probe's recommendation was accepted.** From exp001.2 onward, training runs bf16
autocast instead of pure fp32. This file is the pointer anyone lands on when they ask
"why does this run say `amp_dtype: bfloat16`?"

## The decision

| | before (exp001, from epoch 10) | from exp001.2 |
|---|---|---|
| training numerics | pure fp32 (`use_amp: false`) | **bf16 autocast** (`use_amp: true`, `amp_dtype: bfloat16`) |
| gradient scaler | n/a | auto-disabled (bf16 needs none) |
| STFT / iSTFT | fp32 | **fp32** — unchanged, pinned in the model |
| evaluation | fp32 | **fp32** — unchanged (`inference_amp_dtype: float32`) |
| throughput | ~1.56 it/s | ~2.2 it/s expected (the ~30% fp32 penalty returned) |

Only the *training* forward/backward changed. Evaluation deliberately stays fp32 so val
numbers remain measured by the same instrument as exp001 — the curves stay directly
comparable across the pause rather than needing a caveat.

## Why bf16 and not "fp16 but more carefully"

fp16 killed exp001 twice. Both collapses hit the same classes (가야금 · 대금 · 양금, plus
아쟁 in the second) and both triggered at a *quality level* near −1.5 dB rather than at a
fixed time, so resuming the epoch-9 checkpoint dropped a run straight back into the trap.
Halving the learning rate and resetting the optimizer did not prevent the second collapse;
changing the numerics did.

bf16 fixes the mechanism rather than the symptom: it carries **fp32's exponent range in
16 bits**, trading mantissa precision for headroom. The overflow that fp16 suffers is
structurally impossible, which is why bf16 also needs no loss scaling.

## The evidence (full write-up → `docs/exp001.1_bf16_probe_report.md`)

- **0 non-finite steps in 40,000** — against fp16's 5–8 per 10,000
- Executed **both** historical trip transitions (ep9→10 and ep10→11, the exact step where
  the fp16 recovery attempt fell 13 dB) and the val curve *rose* through both:
  −1.5460 → −1.0468 → −0.7736
- **Beat the certified-clean fp32 arm at all four matched step counts** (+0.28, +0.14,
  +0.19, +0.45 dB)
- Faster and smaller than fp32

⚠️ **The probe's own stated verdict was "AMBIGUOUS by rule, CLEAN by evidence"** — the
briefed success criterion (6 consecutive evals beyond −1.2 dB) was never satisfiable,
because the fp32 control arm does not meet it either; the metric's natural epoch-to-epoch
swing reaches 1.40 dB. The rule was calibrated tighter than the measurement's noise, so
failing it carries no information. Adoption rests on the substantive measurements above,
and that distinction should be stated honestly if the choice is ever questioned.

## Caveat carried forward

fp32 was not perfectly stable either — it had a **milder excursion at epoch 18**
(+0.04 → −1.36, 가야금 and 양금 only) which then self-healed over ~10 cycles. So the
numerics fix removed the *catastrophic* failure mode, not the underlying one. Head
competition over contested plucked-string energy remains the leading hypothesis, and the
가야금/거문고 asymmetry (−5.9 vs +1.5 dB at the best checkpoint) is the evidence for it.

**Watch for it in exp001.2:** if 가야금 or 양금 drops sharply again under bf16, that
confirms the mechanism is not numerical and the fix belongs in the loss or the class
scheme instead.

## Where it is configured

- `configs/exp001.2_htdemucs_9stem_resumed.yaml` — `use_amp: true`, `amp_dtype: bfloat16`,
  `inference_amp_dtype: float32`
- Fork support: `external/msst` branch `gugak-patches`, commit `8dbc33e`
  ("add bf16 autocast, fp32 spectral path, and step telemetry")
- `configs/exp001_htdemucs_9stem.yaml` is left as-is (`use_amp: false`) — it is the record
  of what exp001 actually ran and should not be retro-edited.
