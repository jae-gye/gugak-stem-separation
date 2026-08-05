# exp001 — metrics snapshot at pause (2026-07-31, pre-shutdown)

Frozen before the sym8 maintenance shutdown. Everything here is regenerable from the
checkpoint via `scripts/export_exp001_metrics.py`; tables live beside this file
(`eval_trajectory` · `per_song_best` · `per_genre_best`, parquet + csv).

## Headline

| Item | Value |
|---|---|
| Best val SI-SDR (Σstem-val, 91 songs) | **+1.0409 dB** |
| Best checkpoint | `checkpoints/model_htdemucs_ep_33_si_sdr_1.0409.ckpt` |
| Reached at | epoch 33 = **85,000 optimizer steps** (34 eval cycles × 2,500) |
| Effective batch | 32 (physical 8 × grad-accum 4) → ~2.72 M training mixes seen |
| Status when paused | **still improving** — best epoch *is* the last epoch |
| Zero-shot reference | this is the first fine-tuned number; the old 4-class zero-shot floor is not directly comparable |

Compute note: the *lineage* is 85k steps (epochs 0–9 under fp16, epochs 10–33 under
fp32 resumed from the ep9 checkpoint). Roughly 4 further epochs of fp16 compute were
spent on the two excursion attempts and discarded — they are not part of this lineage.

## Was it still improving? — yes, unambiguously

| Signal | Value |
|---|---|
| Last-5-eval slope | **+0.160 dB / eval cycle** |
| Last-10-eval slope | +0.133 dB / eval cycle |
| Consecutive new bests at pause | **8** |
| Early-stop counter (patience 10) | **0** — never started counting |
| Training loss at pause | 0.004864, still descending every epoch |

Deltas over the final 8 cycles: +0.16, +0.04, +0.20, +0.03, +0.23, +0.21, +0.06, +0.16.
No flattening — the run was stopped by the shutdown schedule, not by convergence.
Honest framing for the deck: *"paused mid-improvement at 85k of a planned 150k steps."*

## Per-class at the best checkpoint (mean over scored val songs)

| Stem class | mean SI-SDR | median | std | songs |
|---|---|---|---|---|
| 타악기 | **+15.43** | +16.57 | 6.43 | 88 |
| 해금 | +4.77 | +4.62 | 6.57 | 67 |
| 피리 | +3.28 | +4.07 | 8.37 | 71 |
| 대금 | +2.64 | +3.20 | 4.79 | 72 |
| 기타 | +1.91 | +3.45 | 5.17 | 23 |
| 거문고 | +1.48 | +0.64 | 5.50 | 64 |
| 아쟁 | −1.01 | −1.13 | 3.94 | 59 |
| 가야금 | −5.88 | −5.66 | 3.30 | 68 |
| 양금 | −13.26 | −11.33 | 6.78 | 9 |

Reading:
- **타악기 is trivially separable** (+15.4) — percussion is timbrally orthogonal to the
  melodic bulk, matching the zero-shot finding that drums transfer first.
- **가야금 (−5.9) vs 거문고 (+1.5) is a 7.4 dB asymmetry** between two plucked zithers.
  A confusion between the pair would show exactly this shape (one head wins the
  contested energy). Direct test for exp002: the clumping ablation these numbers argue
  for, or a confusion-matrix analysis at fixed weights.
- **양금 (−13.3) rests on 9 val songs** — high variance, weakest evidence per the table;
  treat as indicative, not a headline number.
- Both weak classes are also the two that collapsed during the instability events
  (below) — recovery was real but may still be incomplete.

## Per-genre — the unison-hardness argument, quantified

Genre mean SI-SDR against the *measured* audible density (mean number of simultaneously
audible classes per 10 s val window, from `chunk_activities`):

| Genre | audible density | mean SI-SDR |
|---|---|---|
| 산조 | 2.00 | **+12.22** |
| 대풍류 | 4.94 | +8.67 |
| 창작국악 | 5.68 | −0.75 |
| 풍류음악 | 6.39 | +2.04 |
| 판소리 | 6.99 | +3.48 |
| 민요 | 7.00 | +0.29 |
| 궁중음악 | 7.28 | **−1.16** |

**Pearson r = −0.86; slope ≈ −2.4 dB per additional simultaneous instrument.**

This is the quantified version of the heterophony claim: separability falls steeply and
almost monotonically with ensemble density. 산조 — structurally a duo (solo instrument +
장구) — is 13 dB easier than 궁중음악, where seven instruments play heterophonic variants
of one melody. Caveats worth stating on the slide: only 7 genre points, density and
genre are confounded (each genre has a characteristic 편성), and the per-genre means
average over different class sets.

## Stability events (context for reading the curve)

The trajectory contains two visible disturbances; both are documented in
`docs/exp001_overnight_report.md`:

1. **fp16 cliffs (epochs 10–11, twice)** — catastrophic: −1.55 → −8.89, then again
   −1.73 → −14.19 after resuming with a halved LR and a fresh optimizer. Same classes
   each time (가야금 · 대금 · 양금 · 아쟁). Not in this trajectory: the fp32 lineage
   overwrote those epoch entries; the damaged weights are preserved as
   `checkpoints/damaged_ep11_*.ckpt`.
2. **fp32 excursion (epoch 18)** — milder: +0.04 → −1.36, 가야금 and 양금 only, and it
   **self-healed** over the following ~10 cycles (both classes then exceeded their
   pre-excursion levels). Visible directly in `eval_trajectory`.

Interpretation: fp32 removed the catastrophic failure mode but not a milder underlying
one, so the mechanism is probably not purely numerical — head competition over
contested plucked-string energy is the leading candidate, consistent with the
가야금/거문고 asymmetry above. Open question for exp002, not a settled finding.
