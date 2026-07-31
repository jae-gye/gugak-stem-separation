# Solo pool (71470) duration analysis — what a 10 s segment costs

**Question:** exp002 would add the 71470 solo-clip pool to the training draw pool (exp001 is
71955-ensemble-only). Solo clips are short, exp001's training segment is 10 s. Before we
decide what to do with clips shorter than one segment — pad, loop, or reject — we need to
know how much material each choice costs, and *which classes* pay for it.

**How this was produced.** `src/data/clip_duration_analysis.py`, run read-only against
`manifests/parquet/source_manifest.parquet` (which already carries `out_duration` per
ingested file plus the taxonomy join, so no audio was opened and no re-scan was needed):

```bash
uv run python src/data/clip_duration_analysis.py          # segment 10 s, thresholds 2 3 5 8 10
```

Segment length, thresholds, percentiles, the class scheme and the datasets are all
parameters — re-running for a different segment length is `--segment-seconds 8`, nothing
else changes. Tables (parquet + csv twins) land in
[experiments/260730_solo_pool_duration/](../experiments/260730_solo_pool_duration/); the
figure in [notebooks/](../notebooks/).

**One scope note before the numbers.** The brief said "the current 9-class taxonomy".
`configs/stem_taxonomy.yaml` actually defines **11** working groups; exp001's modelled nine
(`gugak_mix.classes`) are those eleven minus `pitched_percussion` (quarantined from exp001,
not folded into 타악기) and minus `voice` (no ensemble material exists). Since `voice` is
**27.5 % of the solo pool by clip count**, collapsing it out would have hidden the single
biggest thing the solo pool adds. So every table below reports all groups present, marked
▸ for the exp001 nine, and keeps `pitched_percussion`, `voice` and the held-aside
`stem_group: null` rows on their own lines.

---

## 1. Overall duration distribution — 71470

| | |
|---|---|
| clips | **9,945** |
| total | **39.78 h** |
| mean | 14.400 s |

| min | p5 | p25 | **p50** | p75 | p95 | max |
|---|---|---|---|---|---|---|
| 1.242 s | 5.069 s | 8.643 s | **12.042 s** | 19.333 s | 27.412 s | 77.521 s |

For scale, the ensemble pool it would join (71955 stems only, masters excluded):
5,767 stems / **368.78 h**, median 216.5 s, min 6.0 s, max 1,153.2 s. Two pools an order of
magnitude apart in per-file length.

## 2. Cumulative share of 71470 shorter than each threshold

"Shorter than" is strict — a clip of exactly 10.000 s still yields one full segment, so it
is not counted as lost.

| threshold | clips below | % of clips | hours below | **% of hours** |
|---|---|---|---|---|
| 2 s | 7 | 0.07 % | 0.003 | 0.01 % |
| 3 s | 48 | 0.48 % | 0.032 | 0.08 % |
| 5 s | 434 | 4.36 % | 0.507 | 1.27 % |
| 8 s | 2,018 | 20.29 % | 3.401 | 8.55 % |
| **10 s (segment)** | **3,427** | **34.46 %** | **6.913** | **17.38 %** |

**The key number: at a 10 s segment, a reject policy throws away 34.5 % of solo clips but
only 17.4 % of solo hours.** The two figures differ by a factor of two because the discarded
clips are, by construction, the short ones — a third of the files carry a sixth of the
audio.

Restricted to just the exp001 nine (excluding voice, pitched_percussion, held-aside):
6,954 clips / 28.48 h, of which 2,378 clips / 4.73 h sit below the segment → **23.74 h would
survive a reject policy**. Against the ensemble pool's 365.97 h in those same nine classes,
the solo pool is a **+7.8 % addition in hours, falling to +6.5 % if short clips are
rejected**. Either way it is a garnish on the ensemble pool by duration — its value is
diversity (clean single-instrument targets, 9,945 independent phrase recordings), not volume.

Note the drop-off is steep between 8 s and 10 s: relaxing the segment to 8 s alone would cut
the loss from 17.4 % to 8.6 % of hours. Segment length is itself a lever here, not a
constant.

## 3. Per stem class — loss at the 10 s segment

Sorted by the share of *hours* each class would lose. ▸ = one of exp001's modelled nine.

| class | clips | hours | median | clips < 10 s | % clips | hours < 10 s | **% hours** | hours kept if reject |
|---|---|---|---|---|---|---|---|---|
| pitched_percussion | 34 | 0.080 | 8.48 s | 30 | 88.2 % | 0.068 | **84.8 %** | 0.012 |
| ▸ 양금 | 72 | 0.245 | 9.05 s | 40 | 55.6 % | 0.084 | **34.3 %** | 0.161 |
| ▸ 기타 | 639 | 2.137 | 9.51 s | 353 | 55.2 % | 0.711 | **33.3 %** | 1.426 |
| (held aside · `stem_group` null) | 222 | 0.684 | 9.78 s | 114 | 51.4 % | 0.221 | **32.4 %** | 0.462 |
| ▸ 타악기 | 686 | 2.074 | 10.26 s | 330 | 48.1 % | 0.628 | **30.3 %** | 1.446 |
| ▸ 거문고 | 825 | 3.261 | 12.00 s | 321 | 38.9 % | 0.659 | **20.2 %** | 2.602 |
| voice | 2,735 | 10.541 | 11.87 s | 905 | 33.1 % | 1.890 | **17.9 %** | 8.650 |
| ▸ 아쟁 | 353 | 1.306 | 11.52 s | 106 | 30.0 % | 0.228 | **17.5 %** | 1.078 |
| ▸ 가야금 | 1,356 | 5.657 | 13.09 s | 468 | 34.5 % | 0.932 | **16.5 %** | 4.725 |
| ▸ 해금 | 1,291 | 5.598 | 13.83 s | 389 | 30.1 % | 0.747 | **13.3 %** | 4.851 |
| ▸ 피리 | 916 | 4.424 | 16.30 s | 207 | 22.6 % | 0.414 | **9.4 %** | 4.009 |
| ▸ 대금 | 816 | 3.775 | 14.88 s | 164 | 20.1 % | 0.330 | **8.7 %** | 3.445 |

**The loss is strongly class-dependent, and it lands hardest exactly where we have least.**
The ordering is close to an inverse of pool size: the two smallest groups
(pitched_percussion at 0.08 h, 양금 at 0.25 h) are also the two shortest-clip groups, so a
reject policy takes 85 % and 34 % of them respectively. Meanwhile the classes that would
barely notice (대금, 피리) are ones already well covered by the ensemble pool.

Full (class × threshold) detail is in
[`per_class_thresholds.parquet`](../experiments/260730_solo_pool_duration/per_class_thresholds.csv).
Two things worth pulling out of it: at every threshold **below 5 s the loss is negligible in
every class** (worst case 해금 at 1.2 % of clips, 0.2 % of hours), and at 8 s
pitched_percussion has already lost 31 % of its hours while 대금 has lost 4.9 %. The
divergence between classes is created almost entirely in the 8–10 s band.

## 4. Relative size of what the solo pool adds, per class

71955 = the ensemble stems exp001 already trains on; 71470 = the solo pool exp002 would add.

**This already exists** — `notebooks/EDA.ipynb` cell 16 ("Cross-dataset stem-group
distribution") computes per-stem-group hours and per-dataset file counts from the same two
sources of truth (`source_manifest.parquet` ⋈ `configs/stem_taxonomy.yaml`), and
`build_source_manifest.py` prints a similar pivot at build time. Neither persists its
numbers, and a notebook cell isn't importable, so the module recomputes the same groupby
from the same manifest — the figures below agree with the notebook's by construction, and
add the ratio column, which is the part that was missing.

| class | solo clips | ensemble stems | solo hours | ensemble hours | solo ÷ ensemble (hours) |
|---|---|---|---|---|---|
| ▸ 가야금 | 1,356 | 692 | 5.657 | 43.945 | 0.129 |
| ▸ 거문고 | 825 | 636 | 3.261 | 40.525 | 0.080 |
| ▸ 기타 | 639 | 254 | 2.137 | 15.642 | 0.137 |
| ▸ 대금 | 816 | 759 | 3.775 | 48.825 | 0.077 |
| ▸ 아쟁 | 353 | 709 | 1.306 | 45.663 | 0.029 |
| ▸ 양금 | 72 | 95 | 0.245 | 6.341 | 0.039 |
| ▸ 타악기 | 686 | 1,091 | 2.074 | 69.155 | 0.030 |
| ▸ 피리 | 916 | 762 | 4.424 | 49.282 | 0.090 |
| ▸ 해금 | 1,291 | 718 | 5.598 | 46.589 | 0.120 |
| pitched_percussion | 34 | 51 | 0.080 | 2.813 | 0.028 |
| voice | 2,735 | — | 10.541 | — | n/a (no ensemble material) |
| (held aside · `stem_group` null) | 222 | — | 0.684 | — | n/a |

By hours the solo pool adds between 2.9 % (아쟁, 타악기) and 13.7 % (기타) per modelled
class — small everywhere. The exception is **`voice`, where the solo pool is not an addition
but the entire supply**: 2,735 clips / 10.54 h against zero ensemble hours. If the stem
scheme ever gains a voice class, this analysis stops being a marginal-gain question and
becomes the only material there is — and 17.9 % of it is shorter than a segment.

## 5. Figure

![71470 clip duration distribution](../notebooks/fig_solo_clip_durations.png)

[notebooks/fig_solo_clip_durations.png](../notebooks/fig_solo_clip_durations.png) — left:
duration histogram (1 s bins) with the 10 s segment marked; right: per-class share below the
segment, by clip count and by hours. Regenerable from the module; per repo convention the
png is not tracked.

## 6. Anomalies noticed in passing

Reported, not fixed. Detail tables are the `anomaly_*` files in the output directory.

**Worth acting on**

1. **CONFIRMED (2026-07-30, audio-verified): three "stems" of `0714_민속악_민요` are copies
   of the publisher master, not stems.** Flagged first from the manifest (대금, 아쟁, 피리
   sharing `out_frames` 14,288,400, `peak` 0.990000, `rms` 0.209530 and `active_frac`
   0.999913 to six decimals), then verified in the audio after the finding was confirmed by
   ear:
   - The three files are **bit-identical to each other** — md5 of the *decoded samples*
     matches exactly (`3704fd14…`). Their file-level md5s differ only in a header/metadata
     chunk, which is why a naive file hash misses this.
   - That audio **is the master**: zero-lag correlation 0.9965, and a single least-squares
     gain (0.746 on the raw sources) reconstructs **99.30 %** of the master's energy. The
     remaining 0.7 % is the mastering EQ/limiting, consistent with 민요 being the
     "≈ clean sum" genre.
   - **The real 대금/아쟁/피리 parts are unrecoverable.** Σ(가야금 + 거문고 + 장구 + 해금),
     the four genuine stems, explains only **63.1 %** of the master; the residual is 60.8 %
     of master RMS and correlates 0.58 with the mix. The wind/string content is audible in
     the master and present in no stem file. The publisher shipped the mixdown three times
     *instead of* delivering those stems.
   - The four remaining stems are genuine — mutually decorrelated at |r| < 0.01.
   - Song is in **train** (confirmed against the frozen `eval_manifest`), so no eval metric
     is affected. Current exp001 exposure is one poisoned source out of 759 대금 / 709 아쟁 /
     762 피리 → **~0.13 % of draws per class** hands the model a full ensemble mix as a
     single-instrument target. Real label noise, too small to justify interrupting a run.
   - **Recommended action:** exclude the three mix-copy files as targets, keep the four
     genuine stems. Since training mixes are incoherent per-class draws rather than
     whole-song sums, the song simply stops contributing to those three classes.
   - *Not* related to the epoch-10 head collapse despite 대금 appearing in both lists —
     가야금 and 양금 collapsed but are unaffected here, 아쟁/피리 are affected but did not
     collapse, and 0.13 % contamination is not a plausible mechanism.

2. **CONFIRMED (2026-07-30): `0885_창작국악_창작국악` is `0886_창작국악_창작국악` repeated
   exactly 7 times.** Found by a dataset-wide audio scan (below). All 13 files in each song
   share identical `peak`, `rms` and `active_frac` to six decimals while the durations differ
   7-fold (336.0 s vs 48.0 s); segment-by-segment comparison confirms every one of the 7
   repetitions matches, with residuals only at the quantization floor (max 3.3e-5, below
   −89 dBFS — attributable to soxr resampling a long looped file versus a short one, so this
   is **publisher-side, not an ingest artifact**: `src_frames` is already exactly 7× at
   48 kHz PCM_24).
   - Both songs are in **train**, so again no eval leakage — but they are the same 48 s of
     music presented as 384 s across two song ids.
   - **Consequence for the draw pool:** every excerpt drawn from 0885 comes from only 48 s of
     unique material, and 0886 duplicates it again. The pair inflates the ensemble stem pool
     by ~1.12 h of the 368.78 h total (**0.30 %**) — negligible for the headline hours in §1
     and §4, but those figures are "what the manifest says", not unique content.
   - **Recommended action:** keep one of the two, and treat 0885's effective length as 48 s
     for any duration- or excerpt-weighted sampling.

3. **`0905_창작국악_창작국악`: master and its single 해금 stem are bit-identical — probably
   legitimate, worth knowing anyway.** It is one of only two single-stem songs in 71955, so
   for a genuinely solo piece the mix *is* the instrument. No defect. But it sits in **val**,
   which means its Σstem variant and its master variant are the same audio: it contributes
   zero mastering-residual signal to the master-vs-Σstem comparison, and is a one-class val
   song.

4. **A 71470 clip pair with identical audio but two different instrument labels.**
   `BP_CR1_03429` (labelled 아쟁) and `BP_CR1_03602` (labelled 징) share frames, peak, rms
   and `active_frac` exactly. One of the two labels is wrong. 징 is one of the held-aside
   instruments, so this also slightly contaminates that group's count.
5. **Two exact-duplicate clip pairs inside 71470:** `AP_F01_00238`/`AP_F01_00614` (both
   해금) and `AP_F01_00228`/`AP_F01_00252` (both 대금) — identical fingerprints, so the same
   phrase is in the pool twice under two clip ids.
6. **Two near-duplicate 좌고 pairs in 71955** (`0015`/`0016` and `0126`/`0127`
   정악_풍류음악/궁중음악): identical frames and rms to six decimals, differing only in the
   eighth decimal of DC offset. Weaker evidence than #1 — 좌고 sits at a fixed fader
   (peak ≈ 0.1520 across *all* 좌고 stems, so peak carries no information here) and it's a
   sparse instrument — but adjacent song ids with matching content is the signature of a
   reused take. Not audio-verified; the dataset-wide scan below did **not** flag these, so
   they are likely distinct takes at the same fader setting.

**Context for the pad/loop/reject decision**

7. **7 stub clips below 2 s**, shortest 1.242 s (해금); 5 of the 7 are 해금. At a 10 s
   segment these would need looping ~8× — the same audio repeating eight times inside one
   training example.
8. **79 % of solo clips are single-channel after ingest** (7,867 of 9,945), which is correct
   per the ingest channel rule (dual-mono and anti-phase collapse to one channel), but it
   varies wildly by class: 피리 100 % mono, 기타 98 %, 대금 93 %, 양금 44 %,
   pitched_percussion 0 %. The model input is 2-channel and `channel_swap_prob: 0.5` is a
   no-op on a mono source, so the solo pool is effectively a *centred* source pool. Not a
   duration issue, but it is an exp002 integration issue.
9. **Sparse content compounds short duration.** The quietest solo clips are 징 at 8 %
   `active_frac` and 장구 at 5.7 % — a 12 s 징 clip at 8 % activity holds under a second of
   actual sound. Per the standing rule, sparse ≠ dead and these must not be dropped, but a
   "long enough" filter on duration alone doesn't guarantee a usable excerpt. The
   activity-aware draw in `mix_dataset.py` already handles this for 71955 and would need to
   for the solo pool too.
10. **36 long outliers in 71470** above the per-dataset fence of 51.4 s (q75 + 3×IQR),
   longest 77.521 s (가야금), and concentrated in the plucked/bowed melodic classes
   (가야금 13, 피리 11, 해금 8). Long, not absurd — no action, but they
   are the only clips that can supply more than one non-overlapping segment.
11. **The histogram is multi-modal, not smooth** — visible spikes around 10, 20 and 24 s
   suggest clip boundaries were cut on musical phrase lengths (or an editing template),
   rather than being a natural continuum. Relevant because it means the mass sitting just
   *below* 10 s is a genuine cluster, not tail noise: the choice of exactly 10 s lands on a
   cliff edge.

**Clean — checked and nothing to report**

- No duplicate `file_id` (0), no `dead` files (0), no ingest errors (0) across either pool.
- Ingest format unification holds: every one of the 15,712 pool files is 44.1 kHz / PCM_24,
  the only variation being channel count. The solo pool's source-side heterogeneity was real
  — 15 distinct (sample-rate, subtype, channels) combinations including 96 kHz PCM_32 and
  float wavs — and is fully resolved in the store.
- 71955 is not immune to the short-file question, just nearly so: 10 of 5,767 stems are
  under 10 s (shortest 6.0 s, in 산조 songs). Negligible, but it means exp001's loader
  already meets this case, and whatever policy we choose should be checked against how it
  handles those ten today.

### The dataset-wide audio scan behind findings 1–3

Findings 1–3 came from a full audio scan of all 903 ensemble songs / 5,767 stems, run
2026-07-30 (~1 min; a 30 s excerpt per file from the ingest store, which is uniformly
44.1 kHz so nothing needs resampling mid-comparison). The detector is tracked code:
**`scripts/stem_duplicate_scan.py`** — re-run it whenever the ingest store is rebuilt. Two
tests per song: md5 of the decoded excerpt (exact identity), and best-fit gain of the master
onto each stem plus the fraction of master energy explained (gain-invariant, so mastering
level differences do not matter). Verdict: **exactly one genuine mix-copy defect (0714), one
whole-song duplication (0885/0886), one benign single-stem song (0905). Nothing else.**

Three methodological notes for whoever re-runs this:

- **Correlation alone does not identify a mix-copy.** The scan's next-highest stems after the
  confirmed cases are a cluster of 산조 해금 stems explaining 83–96 % of their masters, and
  they are perfectly legitimate — a 2-stem 산조 song is 해금 plus 장구, so the 해금 *is* most
  of the mix. Exact audio identity is what separates a defect from a dominant instrument;
  the energy-explained figure only ranks candidates.
- **Silent excerpts hash alike and masquerade as duplicates.** The raw scan produced two
  duplicate "groups" of 35 and 36 files spanning unrelated songs and instruments. All were
  digital silence: every sample in the excerpt is one constant value, either exactly zero or
  a ±1-LSB residue (1.192e-07 = 2⁻²³) left by ingest's DC removal — different residues
  landing in different hash buckets. That is 박 and friends being sparse by design, per the
  standing sparse ≠ dead rule. **Any future duplicate scan must test `n_unique > 1` before
  believing a hash collision.** After filtering, 58 apparent duplicates collapsed to the 3
  real groups above.
- **Known gap: self-loops within a single file are not detected.** 0885 was only caught
  because a shorter twin (0886) existed to hash against. A song that is an N× loop of itself,
  with no twin, would pass every test here. Detecting that needs a different scan (compare
  each file against its own first 1/N for small integer N) — not yet run.

### NEGATIVE RESULT — loudness/crest profiling does not detect disguised masters

**Read this before rebuilding this detector.** Tracked as
**`scripts/loudness_profile_scan.py`**, kept precisely so the negative result is not
rediscovered the hard way.

**The idea.** Byte comparison misses a master that was copied over a stem and then *further
processed*. But a master should still carry a master's loudness signature — louder and more
limited than a stem. So: measure integrated LUFS (ITU-R BS.1770) and crest factor
(20·log₁₀(peak/rms)) for every file, and flag any stem whose profile sits inside the master
distribution rather than the stem distribution.

**What was measured** — all 16,615 files, 465.4 h, ~20 min on 4 workers, zero errors and
zero silent files (nothing was unscoreable):

| population | n | integrated LUFS | crest dB | active_frac |
|---|---|---|---|---|
| masters | 903 | −13.08 ± 2.62 | 14.19 ± 2.42 | 0.982 ± 0.031 |
| 71955 stems | 5,767 | −20.84 ± 5.11 | 20.74 ± 3.53 | 0.761 ± 0.248 |
| 71470 clips | 9,945 | −25.52 ± 8.11 | 18.73 ± 5.43 | 0.859 ± 0.165 |

**The premise holds for the centres and fails at the tails.** Masters really are ~7.8 dB
louder and ~6.5 dB lower-crest than ensemble stems. But the stem cloud is *twice as wide*
(σ 5.11 vs 2.62 LUFS), and a wide cloud swallows a narrow one:

- **27.9 %** of stems fall inside the masters' 95 % ellipse
- **70.3 %** of masters fall inside the stems' ellipse

Consequently there is **no usable threshold**. A naive likelihood rule (master more probable
than stem) flags 2,479 of 15,712 sources — 15.8 %, meaningless. Calibrating with a realistic
rare prior (10⁻³; we know of ~4 candidates in 15.7 k files) flags **zero, including the known
positives**. Nothing in between catches the known-bad files without dragging in hundreds of
legitimate loud, dense stems. **As a binary flag this detector produces no defensible output.**

**Three things already tried — do not repeat them:**

1. **Adding `active_frac` as a third axis makes it worse**, not better: masters-inside-stem-
   ellipse rises 70.3 % → 89.5 %. Solo clips are continuously active too, so density does not
   discriminate.
2. **Pooling the two datasets destroys the ranking.** 71470 has no masters at all, so "inside
   the master distribution" is not even a meaningful question there, and loud dense solo clips
   at ~−11 LUFS crowd out the real 71955 candidates (they filled the entire pooled top 25).
   Always score **per dataset**.
3. **A loudness detector structurally cannot catch a solo-piece master-as-stem.**
   `0905_창작국악_창작국악`'s 해금 is bit-identical to its own master, yet ranks 4,165/5,767 —
   correctly, because *that master is itself stem-like*: −18.08 LUFS, 22.69 dB crest,
   Mahalanobis 4.38 from the master centroid, i.e. more extreme than **99.1 % of real
   masters**. A solo piece was never bus-compressed or limited. The premise "masters are
   louder and more limited" describes **ensemble** masters only, and this detector can only
   ever catch *an ensemble mix* wearing a stem's filename.

**What it is still good for: a per-dataset ranking for human review.** Restricted to 71955
the three confirmed 0714 mix-copies rank **1, 2, 3 of 5,767** — strong signal for that
failure mode. Cross-checked against the independent duplicate scan, the two detectors agree
only weakly (Spearman 0.269), so they catch genuinely different things; only five files score
highly on both (the 0714 trio plus `0394`/`0393_민속악_산조` 해금, the legitimately-dominant
2-stem 산조 cases).

Review candidates 4–10 in 71955, none approaching 0714's signature — these read as
hot-mixed dominant instruments rather than copies:

| rank | song | stem | split | LUFS | crest | master energy explained |
|---|---|---|---|---|---|---|
| 4 | `0788_민속악_민요` | 아쟁 | train | −12.98 | 14.40 | 52.4 % |
| 5 | `0898_창작국악` | 대금 | train | −11.43 | 13.96 | 19.1 % |
| 8 | `0147_정악_궁중음악` | 해금 | **val** | −13.84 | 14.62 | 11.9 % |
| 9 | `0953_창작국악` | 대금 | train | −13.93 | 14.56 | 65.0 % |
| 10 | `0751_민속악_민요` | 아쟁 | **test** | −14.09 | 14.25 | 51.0 % |

Ranks 8 and 10 were listened to (they sit in val and test, where a mix-copy would inflate a
per-stem SDR rather than merely add training noise). One systematic pattern worth noting:
ranks 4, 10 and 14 are all **아쟁 in 7-stem 민요 songs** with `active_frac` ≈ 1.0 and ~50 % of
master energy explained — either 아쟁 is consistently mixed hot in 민요 or it doubles the
melody. Observation, not a defect.

**Overall verdict on the data.** Across three independent detectors (manifest fingerprints,
exact-audio identity + master correlation, loudness/crest profiling) the ensemble set yields
**two genuine defects in 903 songs, both confined to `train`**. No eval metric is affected by
either. That is a clean dataset by any reasonable standard — the earlier alarm did not
survive contact with the evidence.

---

## What this implies for pad / loop / reject

*My reading of the numbers, not a decision — the call is yours.*

The headline is that **reject is cheaper than it looks in aggregate and more expensive than
it looks per class**. Globally it costs 17.4 % of solo hours, and since the solo pool is only
a +7.8 % addition to the ensemble pool anyway, that's ~1.2 % of total training hours — a
rounding error. If the argument for the solo pool were volume, reject would be obviously
fine.

But the argument for the solo pool isn't volume, it's *coverage of the thin classes*, and
that's precisely where reject bites: it removes 85 % of pitched_percussion, 34 % of 양금,
33 % of 기타 and 30 % of 타악기. Those are the classes where exp001 has the least ensemble
material and where a handful of extra clean phrase recordings plausibly matters most. Reject
takes the solo pool's best contribution and discards the majority of it. (양금 is
particularly awkward: 72 clips is already thin, and reject leaves 32.)

Between the two ways of keeping short clips, I'd expect **loop to be worse than pad, and for
the risk to be a periodicity artifact rather than a data-volume one**. Looping a 1.2 s clip
into a 10 s segment repeats it eight times at an exactly constant interval — a perfectly
periodic target the model can learn to expect, and one that never occurs in real gugak.
Padding with silence has the opposite failure mode: it teaches the model that a class's
target is silent for part of a segment, which is at least *true* of sparse instruments (박,
징) and is behaviour the incoherent-mix design already produces, since a mix with n < 9
classes drawn leaves the undrawn heads at zero for the whole segment. Padding a solo clip
is a smaller extrapolation from what the loader already does than looping is.

The two things I'd want to know before choosing, both cheap:

- **Is the segment length itself negotiable?** Moving from 10 s to 8 s halves the problem
  (17.4 % → 8.6 % of hours) and reduces pitched_percussion's loss from 85 % to 31 %. Since
  10 s is the pretrained checkpoint's native training length that's not free, but it makes
  "reject at a shorter segment" a real third option rather than a compromise.
- **Does the choice have to be uniform?** Reject is nearly free for 대금, 피리, 해금, 가야금
  and voice (8–18 % of hours, all with adequate material). It is expensive only for
  pitched_percussion, 양금, 기타 and 타악기. A per-class rule — reject where we're rich, pad
  where we're poor — captures most of the material at the cost of a class-conditional
  code path. Worth considering, though it does make the draw distribution harder to reason
  about, and pitched_percussion is quarantined from exp001 anyway so it may be moot.

The one option I'd argue against outright is **looping short clips of the sparse percussion
classes**, which is the intersection of both risks: sparse content, so the loop repeats
mostly silence with a periodic transient, at exactly the classes whose activity statistics
we least want to distort.
