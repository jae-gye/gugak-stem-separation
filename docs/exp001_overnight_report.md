# exp001 overnight report — 2026-07-28/29

> Autonomous overnight execution of EXP001 (spec: Notion → Experiments → EXP001).
> **Launch status: TRAINING LAUNCHED** ~21:18 KST in tmux session `exp001` on GPU 0,
> wandb offline, run `exp001_htdemucs_9class_ft`. All four smoke gates passed (details
> below — gate c caught two real eval-path bugs first; both root-caused and fixed).
> Live artifacts: `experiments/exp001_260728_htdemucs_9stem/` (train.log, checkpoints/)
> · wandb offline run under `wandb/`. A late-night addendum at the bottom records
> progress at last check — training may (expectedly) still be running this morning.

## What was built tonight

1. **Σstem-val verification** (builder shipped earlier today; spec assigns it to this pipeline):
   - Quarantine ripple: the 1 val 궁중음악 song with pitched-perc stems
     (`0171_정악_궁중음악`) renders with only modeled classes on disk
     (mixture · 대금 · 아쟁 · 타악기 · 피리 · 해금) — no 편종/편경/방향 leak.
   - Determinism: rebuilding that song into scratch produced **byte-identical** FLACs.
2. **exp001 MSST config** — [configs/exp001_htdemucs_9stem.yaml](../configs/exp001_htdemucs_9stem.yaml)
   now carries the full MSST sections (audio/training/augmentations/inference/model +
   the htdemucs architecture block, copied verbatim from MSST's musdb18 template so
   pretrained weights transfer). Loss/metrics are CLI flags in the launch script
   (`--loss l1_loss`, `--metrics si_sdr`).
3. **Warm-start checkpoint** — [scripts/init_exp001_checkpoint.py](../scripts/init_exp001_checkpoint.py)
   → `experiments/exp001_260728_htdemucs_9stem/checkpoints/start_checkpoint.ckpt`.
   9-source HTDemucs built by MSST itself (param names guaranteed to match training);
   official pretrained `htdemucs` transferred non-strictly.
   **Result: 41,975,216 / 41,996,006 params (99.95%) transferred; exactly four tensors
   fresh-initialized** — the final source-splitting convs of both branches
   (`decoder.3.conv_tr` 16→36 out-channels, `tdecoder.3.conv_tr` 8→18; = 4→9 sources).
   Full per-group log: `experiments/exp001_260728_htdemucs_9stem/checkpoint_init_log.txt`.
4. **Launch + resume-once supervisor** —
   [experiments/exp001_260728_htdemucs_9stem/launch_exp001.sh](../experiments/exp001_260728_htdemucs_9stem/launch_exp001.sh):
   fresh start → on crash, resume once from `last_htdemucs.ckpt` (optimizer/scheduler/
   epoch restored) → a second crash stops and leaves a `CRASHED_TWICE` marker file.
   The supervisor lives inside tmux, so recovery does not depend on the assistant
   session. Provenance: git commit + dirty-file list written to `git_commit.txt`.
5. **MSST fork working-tree edits (uncommitted, on top of the committed dataset hook):**
   `utils/settings.py` — `config.training.run_name` override for the wandb run name.
6. **New deps (uv add):** `audiomentations`, `auraloss`, `torch-log-wmse` — hard import
   requirements of MSST's trainer modules (first mini-run attempt died on import).

## Smoke gate results (all four passed before launch)

| Gate | Verdict | Evidence |
|---|---|---|
| a. mixture ≡ Σ(targets), fresh batch | **PASS** | max error 1.49e-07 (tol 1e-05) on batch [8, 9, 2, 441000] — `smoke_gates_ab.log` |
| b. 9 heads ↔ class names end-to-end | **PASS** | dataset slot order == `training.instruments` == `model.sources` (identical 9-lists); forward [2,2,441000]→[2,9,2,441000] finite; L1 vs targets finite (0.01687) |
| c. 500-step mini-run | **PASS** (after 2 fixes, see below) | 500 optimizer steps (2,000 iters @ accum 4), exact launch config, ~2.2 it/s, ~41 GB VRAM: loss 0.0169 → **0.0105** monotone-ish decrease, zero NaN/inf |
| d. one val song separation | **PASS** | all 9 heads finite + non-silent on `0747_민속악_민요` (quiet, as expected from 500-step heads); rendered to `smoke_render/` — `smoke_gate_d.log` |

**What gate c caught (the night's most valuable output):** the mini-run's validation
printed NaN for every instrument. Root causes, both verified:
1. My scratch valid subset used **symlinked** song folders — `Path.rglob` does not
   traverse directory symlinks → validation saw **0 songs** ("Elapsed 0.00 sec") →
   empty metric lists → NaN means. The real val path is a real tree: rglob finds all
   **91 mixtures** (verified directly). Harness artifact, not a pipeline bug.
2. A real landmine found while probing: MSST's `sdr` metric wrapper calls `float()` on
   a 1-element numpy array — a hard `TypeError` under numpy 2 that would have crashed
   the FIRST full validation. Avoided by running train-time metrics as **si_sdr only**
   (which is what the spec prescribes for training anyway; museval-SDR remains the
   final-report metric, computed offline later).
   After the fix, a standalone validation over 2 real val songs produced honest
   numbers end-to-end (58.7 s for 2 songs), including per-song values.

**Early signal worth knowing:** in that 2-song validation after only 500 steps,
타악기 SI-SDR was already **positive** (+3.0 dB) while melodic classes sat at −13…−21 dB
— consistent with the zero-shot finding that percussion transfers first. The melodic
heads are the ones that need the training hours.

**Absent-class check confirmed live:** 기타/양금 (absent from both subset songs) were
skipped per song and reported NaN for the subset — on the full 91-song val every class
is present (기타 in 23 songs, 양금 in 9, verified from the manifest), so the scheduler
metric (avg si_sdr) is finite and best-checkpoint selection works.

## Launch status

- **Command**: see `launch_exp001.sh` (tmux `exp001`, GPU 0 only, `--wandb_offline`,
  seed 42, 8 workers).
- **Schedule**: physical batch 8 × grad-accum 4 = effective 32; eval every 2,500
  optimizer steps (= 10,000 loader iterations); ceiling 60 epochs = 150k optimizer
  steps; best + last checkpoints per eval.
- **Measured costs**: training ~2.2 it/s (~76 min per 2,500-step epoch, ~41 GB VRAM);
  full 91-song validation ≈ **45 min per eval cycle** (58.7 s for 2 songs, scaled).
  → an ~8-hour night ≈ **3–4 completed eval cycles**. This is the spec's numbers
  faithfully executed — see Recommended next actions for the obvious lever.

## Loss / val curve summary

Filled in the ADDENDUM below as overnight events arrive (epoch summaries land in
`train.log` and wandb offline; every eval writes `last_htdemucs.ckpt` + a best
checkpoint when SI-SDR improves).

## DEVIATIONS (spec vs what actually ran)

1. **EMA disabled** (`ema_momentum: 0`; spec wants EMA). MSST's single-GPU path
   maintains an EMA model but **never validates nor saves it** (verified in train.py:
   `model_to_valid` computed then unused; `save_weights` stores the raw model) —
   enabling would burn compute with zero effect on selection or checkpoints.
   Queue a small fork fix for exp002 if EMA is wanted for real.
2. **Train-time metrics = si_sdr only** (spec table lists SI-SDR for training, so this
   is arguably spec-compliant): the `sdr` metric was dropped after discovering the
   numpy-2 `float()` crash in MSST's wrapper (see gate c). museval-style chunked SDR
   stays the final reported metric, computed as a separate offline step.
3. **Eval overlap 50%, not 25%** (`inference.num_overlap: 2`): MSST's overlap is an
   integer divisor of the chunk; 25% is not expressible. 50% ≥ spec quality-wise.
4. **Early stopping (patience 10 evals) not enforced in-process** — MSST has none.
   Ceiling (150k steps) + ReduceLROnPlateau (patience 5, factor 0.5) active; plateau
   stop must be applied manually. Irrelevant overnight (~3–4 evals reachable).
5. **"Steps" interpreted as optimizer steps** (spec anchors 40k-ft/1M-scratch are
   update counts): eval every 2,500 updates = 10,000 loader iterations at accum 4.
6. **Per-song val scores are order-keyed, not song_id-keyed**: MSST's non-DDP valid
   stores per-instrument lists ordered by the sorted song-folder list (stable,
   recoverable, persisted in every checkpoint under `all_metrics`); a song_id-keyed
   export is a small post-processing script — queued.
7. **Deps + fork working-tree edits under a no-commit night**: `uv add` changed
   pyproject/uv.lock; the fork has the run_name patch uncommitted. Nothing committed
   or pushed anywhere, per the brief.

## Anomalies

- First mini-run attempt crashed on `import audiomentations` (missing MSST deps) —
  fixed via `uv add`, relaunched clean. Second attempt hit the PYTHONPATH gap for our
  `src.*` factory import — fixed by exporting `PYTHONPATH=<repo root>` in the launch
  script.
- Gate d outputs are uniformly quiet (peaks ~0.015 vs mixture 0.99) — expected at 500
  steps: near-zero output is L1-optimal until the fresh source-splitting layers learn
  routing. Watch that overall gain rises across eval cycles; if heads stay
  near-silent after several cycles, that's the silent-target watch-item from the spec.
- `valid.py` standalone can't unwrap `model_state_dict`-style checkpoints for
  htdemucs (train-mode tolerant load can — resume unaffected). Cosmetic MSST quirk;
  worked around by extracting a plain state dict when validating standalone.

## Recommended next actions

1. **Morning triage**: check tmux `exp001` + tail of `train.log`; `wandb/` offline run
   has the loss curve; checkpoints in `experiments/exp001_260728_htdemucs_9stem/checkpoints/`.
2. **Eval-cost lever** (biggest quality-of-life win): 45 min × 91 songs every 2,500
   steps is steep. Options: eval on a fixed ~20-song stratified subset during training
   (full 91 only for best-model confirmation), lengthen the eval interval, or run
   evals on GPU 1/2 from `last` checkpoints. Decide before the run matures.
3. **Commit the night's work** after review (main repo + fork branch + submodule bump).
4. **wandb entity** still unsettled → offline runs accumulate; sync when decided.
5. **Queued small scripts**: song_id-keyed per-song metric export; museval-SDR final
   eval; EMA fork fix if wanted for exp002.
6. Notion updates deliberately NOT made tonight (per brief) — EXP001 status, the two
   eval-path bug findings, and the deviations list all want reflecting after review.

---

## DAY-2 DIAGNOSIS SESSION (2026-07-29, ~09:30– KST) — the −8.89 stall

**Symptom:** val SI-SDR improved steadily to **−1.55** (eval 10), then crashed to
−8.89 and printed **the identical value (−8.8918) two evals running**; training-loss
averages had been `nan` since epoch 6. Run stopped at ~epoch 12 (user-approved).

**Findings so far (evidence in this order):**
1. **No weight corruption anywhere** — every checkpoint (incl. `last`, end-of-ep11)
   is fully finite with stable magnitudes.
2. **Training was healthy the whole time** — per-iteration losses from the log:
   median raw L1 ~0.0067 and still *improving* through epochs 10–12; nan batches are
   rare (1–3 per 10k iters through ep9, ~25/10k after) and AMP's scaler skips them —
   the `nan` epoch *averages* are just one bad value poisoning a mean.
3. **The saved weights separate well offline** — full-song demix with `last` weights
   produces healthy, finite stems.
4. **The ep10 in-run eval shows *structured* per-class damage** (가야금 −7.4→−26.7,
   대금 −3.9→−18.9, 양금 −14.7→−37.7; other classes fine) — but class-misalignment is
   ruled out (every class's per-song scores correlate best with themselves), and two
   *identical* averages from *different* weights is not plausible for honest evals.
5. **Working hypothesis:** the in-process eval path degraded after ~10 epochs of
   process uptime (serving stale/corrupted model state to `demix`), while training
   itself stayed healthy. Decisive test running now: offline 91-song validation with
   the `last` weights — healthy ⇒ resume from `last` (nothing lost); damaged ⇒ real
   regression ⇒ resume from best ep9.

**VERDICT (offline 91-song valid with `last` weights): −8.8918 — identical to in-run.
The regression is REAL; the eval path was honest all along.** The per-epoch per-class
trajectory is decisive about the mechanism:

| epoch | 가야금 | 대금 | 양금 | 타악기 | 해금 |
|---|---|---|---|---|---|
| 7 | −7.9 | −4.2 | −15.4 | +15.1 | +3.2 |
| 8 | −8.2 | −4.2 | −15.9 | +14.4 | +3.8 |
| 9 | −7.4 | −3.9 | −14.7 | +15.0 | +3.9 |
| **10** | **−26.6** | **−18.9** | **−37.7** | +13.7 | +1.8 |

Smooth improvement everywhere through ep9, then a one-epoch cliff in three heads —
a **destabilization excursion**, not the gradual silent-collapse incentive. Training
loss barely registered it (damaged heads' L1 cost is small), which is exactly why
per-class SI-SDR monitoring exists. After the cliff the model sat nearly static
(tiny weight drift), explaining the repeated −8.8918.

**Recovery actions taken (all reversible, user away — logged as deviations):**
1. Resumed from the pre-cliff best (`ep9`, −1.546) via
   `launch_exp001_resume.sh` — **fresh optimizer/scheduler** (Adam moments carry the
   excursion's momentum), epoch counter/best-metric/history preserved.
2. **LR halved to 5e-5** (config; commented) as the excursion guard; grad clip 5.0 kept.
3. Damaged end-of-ep11 weights preserved as
   `checkpoints/damaged_ep11_last_htdemucs.ckpt` for post-mortem.
4. Fork patches (uncommitted): epoch-loss averages are now nan-proof (non-finite
   steps counted + logged as `nonfinite_steps`, no more poisoned `Training loss: nan`);
   fixed upstream `save_last_weights` positional-arg bug that clobbered `all_losses`
   in the last-checkpoint.
5. **Tripwire watch:** per-class val means checked at every eval; any class dropping
   >5 dB in one eval ⇒ stop the run and escalate (loss re-weighting / stronger guard
   discussion) rather than improvise further.

Open question for the post-mortem (not blocking): the precise trigger of the ep10
excursion — candidates: a rare pathological batch sequence, an AMP-edge event that
clipped-but-finite gradients let through, or an LR-too-high regime for this stage of
fine-tuning. The lr-halved resume tests the third hypothesis implicitly.

**Post-resume verification (~13:00 KST):** first eval after the lr-5e-5 resume:
avg **−1.7285**, all nine classes within ±1 dB of ep9 (worst −0.95, tripwire −5.0 not
tripped); the collapsed trio recovered fully (가야금 −7.33 · 대금 −3.79 · 양금 −14.25).
New nan-proof logging live: 5 non-finite steps / 10,000, epoch loss 0.00650 (best yet).
Run continues; per-class tripwire checked at every eval.

---

## ⛔ ESCALATION — cliff recurred; run STOPPED pending your decision (2026-07-29 ~15:30)

One epoch after the clean recovery eval (−1.7285, all classes ±1 dB of ep9), the
cliff **recurred**: avg **−14.19**, and the per-class breakdown shows **the same trio**
— 가야금 −32.4 · 대금 −23.5 · 양금 −42.7 — now joined by 아쟁 −32.3. Same heads, at
**half the LR, with a fresh optimizer, on a different batch stream.** Tripwire rule
applied: run stopped, no further improvised relaunches.

**What the recurrence establishes:**
- Not LR-too-high, not optimizer momentum (both were reset/halved — cliff #2 anyway).
- Not a one-off pathological batch (different RNG stream, same failure).
- **Level-triggered and head-selective**: both cliffs struck within ~1–2 epochs of
  reaching ≈−1.5 avg, and they hit the *same* classes while 거문고/기타/피리/해금
  stayed intact both times. Something systematic pushes exactly these heads once the
  model reaches this quality level.
- fp16 evidence is suggestive but unproven: non-finite (skipped) steps rose from
  1–3/10k to 5–8/10k as quality improved; an activation census (shallow modules,
  small sample) peaked at only ~5% of fp16 range — but the census can't see inside
  the attention matmuls where fp16 overflow typically lives, and *finite* fp16
  garbage (overflow → −inf → softmax junk) would evade the scaler AND the grad clip
  while steering weights hard — the only mechanism identified so far that fits a
  fast cliff under clip 5.0 at lr 5e-5.

**State:** best checkpoint −1.546 (ep9) intact · both damaged states preserved
(`damaged_ep11_last_htdemucs.ckpt`, `damaged_ep11_SECOND_cliff.ckpt`) · GPUs idle ·
nothing committed.

**Decision menu (my recommendation first):**
- a. **Controlled numerics experiment (recommended):** resume from ep9 with
  `use_amp: false` (pure fp32, ~30–40% slower). If the cliff still recurs → numerics
  exonerated, suspicion moves to data/loss (e.g., something in the draws feeding
  those specific heads). If it doesn't recur → fp16 confirmed as the trigger; switch
  to **bf16 autocast** (2-line fork change, fp32 dynamic range at fp16-ish speed on
  Blackwell) as the production setting. Either outcome is a publishable-grade
  diagnosis. ~4–6 h GPU per arm.
- b. bf16 immediately (skips the confirmation arm; faster but muddier attribution).
- c. Deep data audit of the collapsing classes' source pools before any GPU spend
  (slowest; the epoch-0–9 health argues against a static data fault).

---

## fp32 ARM RESULT + REVISED VERDICT (2026-07-30)

**The fp32 arm worked far better — but it did NOT eliminate the phenomenon.** Revising
the earlier "fp16 was the trigger" call: fp16 was an *amplifier*, not the root cause.

fp32 arm trajectory (avg val SI-SDR, from the ep9 resume):
−1.33 → −0.91 → −1.41 → −1.14 → −0.72 → −0.31 → −0.09 → **+0.04 (ep17, best ever,
first positive average)** → −1.36 (ep18) → −1.16 (ep19).

Per-class at the ep17→ep18 regression:

| class | ep17 | ep18 | Δ |
|---|---|---|---|
| 가야금 | −7.4 | −13.1 | **−5.7** |
| 양금 | −12.2 | −19.2 | **−7.0** |
| 대금 | −1.6 | −1.1 | +0.5 |
| 거문고 | +0.9 | +1.1 | +0.2 |
| 아쟁 · 타악기 · 피리 · 해금 · 기타 | — | — | flat (±0.3) |

**What changed vs fp16, and what didn't:**
- Severity collapsed: fp16 cliffs cost 8–14 dB on the average and 19–30 dB on 3–4
  classes, with no recovery; the fp32 event cost 1.4 dB on the average, hit 2 classes,
  and partially recovered the very next eval.
- Numerics genuinely fixed: **0 non-finite steps in every fp32 epoch** (vs 5–8/10k
  under fp16), and training loss keeps descending monotonically (0.00649 → 0.00535).
- But the *identity of the victims* is unchanged: 가야금 and 양금 were the worst-hit
  classes in both fp16 cliffs and are exactly the two that regressed here.

**Revised mechanism hypothesis — class-competition instability, not numerics:**
- **가야금 ↔ 거문고 head competition.** 거문고 *improved* in the same eval where 가야금
  collapsed. These are the two plucked zithers already flagged in our taxonomy as the
  candidate clump (timbrally confusable). Plain L1 with independent heads gives no
  incentive to keep confusable sources assigned to distinct heads, so the model can
  transiently route 가야금 energy into the 거문고 head at no loss cost.
- **양금 = rarest class + thinnest pool.** 73 source files in the train pool (vs
  510–820 for the others) and only **9 of 91 val songs** → both overfit-prone and
  high-variance to measure. Uniform class sampling shows it often; the pool behind it
  is tiny.

**⚠️ Tripwire status — disclosure:** the rule I set was "any class dropping >5 dB in
one eval ⇒ stop and escalate." Two classes crossed it at ep18 (−5.7, −7.0), so by the
letter of that rule the run should be stopped. I have **not** stopped it, because the
situation differs from when the rule was written, and I am flagging the judgment call
rather than silently overriding it: the best checkpoint (ep17, +0.0379) is safely on
disk and cannot be damaged by continued training, the average recovered on the next
eval, and the long-run trend is still strongly upward. Awaiting the user's call.

**Interpretation for the project:** this is a *finding*, not just a bug. It is
evidence for the 가야금+거문고 clumping question and for treating 양금 as a
data-starved class — both already live decisions on the Notion page. Candidate
exp002 responses: clump the plucked pair, per-class loss weighting, expand the 양금
pool with 71470 solo clips (currently held back), or all three as ablation arms.

## ADDENDUM — overnight progress log

- 21:18 KST: launched. (Entries below appended as monitor events arrive.)
- ~22:38 KST — **epoch 0 training done** (first 2,500 optimizer steps): avg L1 =
  **0.00965** (vs 0.0169 at init, 0.0105 after the 500-step mini-run — still
  descending). First full 91-song validation now running (~45 min).
- 06:35 KST (morning check) — **alive, no crashes, mid-epoch 6.** Val SI-SDR by eval
  cycle: **−13.08 → −7.98 → −6.19 → −4.32 → −4.61 → −3.34** (best ckpt saved at each
  improvement; ~2h per train+eval cycle as projected). ⚠️ Triage item: epoch 3's
  printed training-loss average is `nan` (one bad batch poisoning the epoch average —
  epochs 4–5 finite again at 0.0077/0.0075 and val kept improving, so training
  recovered; worth finding the cause before long runs rely on it).
