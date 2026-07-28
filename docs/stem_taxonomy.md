# Instrument taxonomy — gugak stem separation

Human-facing reference for how instruments map across the two datasets.
**Canonical machine-readable mapping → [`configs/stem_taxonomy.yaml`](../configs/stem_taxonomy.yaml)** —
that file is the single source of truth; the tables here are its readable rendering.
`src/data/build_source_manifest.py` reads the YAML and stamps `instrument_canonical`,
`stem_group`, `pitched`, and `instrument_group_71955` onto every row of
`manifests/parquet/source_manifest.parquet`.

> ⚠️ **Working scheme, not final.** This is the publisher's grouping plus two deliberate
> additions. It is revisited after the taxonomy lit review, and again with the prof
> (→ Notion · Stem Class Scheme). There is deliberately **no "acoustic family" axis** —
> that stays out until the review lands.

---

## 1. The 11 working stem groups

The publisher's 9 classes (`instrumentMajor`), plus `pitched_percussion` split out of
타악기, plus `voice` which exists only because of 71470.

| Stem group | 71955 stems | 71470 clips | Total hours |
|---|---|---|---|
| **타악기** | 1,091 | 686 | 71.2 |
| **피리** | 762 | 916 | 53.7 |
| **대금** | 759 | 816 | 52.6 |
| **해금** | 718 | 1,291 | 52.2 |
| **아쟁** | 709 | 353 | 47.0 |
| **가야금** | 692 | 1,356 | 49.6 |
| **거문고** | 636 | 825 | 43.8 |
| **기타** ("other") | 254 | 639 | 17.8 |
| **양금** | 95 | 72 | 6.6 |
| **pitched_percussion** *(new)* | 51 | 34 | 2.9 |
| **voice** *(new)* | **0** | 2,735 | 10.5 |
| *held aside — see §6* | — | 222 | 0.7 |

Two things the publisher scheme does that are easy to misread: **기타 means "other", not
guitar** — it holds four pitched winds and no percussion. And the six single-instrument
groups exist because those instruments are *common*, not from any acoustic principle.

**`voice` is the one group with zero ensemble material.** 71955 has no vocal stem anywhere,
including 판소리, so all 10.5 h come from solo clips.

### What the solo pool actually adds

The solo clips change the shape of the pool far more in **source count** than in duration —
71470 clips have a median length of ~12 s against ~216 s for ensemble stems.

| Stem group | Sources 71955 → total | ×count | Hours 71955 → total | Δhours |
|---|---|---|---|---|
| 가야금 | 692 → 2,048 | **×2.96** | 43.9 → 49.6 | +13% |
| 해금 | 718 → 2,009 | **×2.80** | 46.6 → 52.2 | +12% |
| 기타 | 254 → 893 | **×3.52** | 15.6 → 17.8 | +14% |
| 거문고 | 636 → 1,461 | ×2.30 | 40.5 → 43.8 | +8% |
| 피리 | 762 → 1,678 | ×2.20 | 49.3 → 53.7 | +9% |
| 대금 | 759 → 1,575 | ×2.08 | 48.8 → 52.6 | +8% |
| 양금 | 95 → 167 | ×1.76 | 6.3 → 6.6 | +4% |
| pitched_percussion | 51 → 85 | ×1.67 | 2.8 → 2.9 | +4% |
| 타악기 | 1,091 → 1,777 | ×1.63 | 69.2 → 71.2 | +3% |
| 아쟁 | 709 → 1,062 | ×1.50 | 45.7 → 47.0 | +3% |
| **voice** | 0 → 2,735 | — | 0 → 10.5 | **all of it** |

→ **Do not read the ×count column as "more training data."** It is roughly +10% audio.
What it actually buys is **draw diversity** for incoherent-mix augmentation: the mixer draws
a random source per stem class, so tripling the number of *independent* 가야금 sources
triples the variety of that draw even though total 가야금 audio barely moves.

→ The ensemble-only class ordering is also not the ordering the mixer sees. By source count
가야금 and 해금 overtake 타악기, which is the largest class by hours.

---

## 2. Deviation 1: `pitched_percussion`

**Decision 2026-07-27.** 편종 · 편경 · 방향 are precisely-tuned idiophones that the publisher
files under 타악기 alongside drums and clappers. Split into their own group.

| Instrument | Romanized | What it is | 71955 stems | Duration | Songs |
|---|---|---|---|---|---|
| **편종** | *pyeonjong* | 16 tuned bronze bell-chimes | 17 | 56.3 min | 17 |
| **편경** | *pyeongyeong* | 16 tuned L-shaped stone chimes | 17 | 56.3 min | 17 |
| **방향** | *banghyang* | 16 tuned iron slabs | 17 | 56.3 min | 17 |
| | | | **51** | **168.8 min (2.81 h)** | **17** |

Evidence behind the split, and why it is flagged provisional:

- **One fixed ensemble, not three independent instruments** — all three occur in *exactly
  the same 17 songs*; pairwise intersection is 17/17 in every direction.
- **All 17 songs are 궁중음악**, out of only 47 궁중음악 songs in the dataset → the only
  **genre-locked** group in the scheme.
- **Thin split coverage:** train 12 / val 1 / test 4 per instrument. Validation rests on
  **a single song**, so per-stem val SDR will be near-unmonitorable and its early-stop
  curve noise.
- **Track count binds harder than hours** — 17 songs is the real constraint, not 2.81 h.

**양금 is deliberately NOT folded in.** It is pitched percussion by construction (struck-string
dulcimer; 71470 codes it `PT01`, same family as 편종/편경), but the publisher gave it its own
top-level group and we preserve that pending prof discussion.

## 3. Deviation 2: `voice`

71470 supplies clean solo vocal clips that have no counterpart in the ensemble set.

| Code | Instrument | Clips | Hours |
|---|---|---|---|
| `VF01` | 여성소리 (female) | 1,647 | 6.21 |
| `VM02` | 남성소리 (male) | 1,078 | 4.30 |
| `VH03` | 합창 (chorus) | 10 | 0.04 |
| | | **2,735** | **10.5** |

---

## 4. Two labelling systems, one canonical name space

| Dataset | How instruments are labelled | Where it lives |
|---|---|---|
| **71955** (ensemble) | Korean names in the filename (`<song>_피리2.wav`) + a publisher **9-class group** (`instrumentMajor`) | `labels/*.json` — only **297 of 903** songs carry the tags |
| **71470** (solo) | Publisher **code** only (`instrument_cd`: `SP01`, `PN04`, …), no name field | `labels/*.json`, all 9,945 clips |

**Provenance of the mapping:**
- `instrumentMajor` groups recovered from the 297 tagged label JSONs. `metadata.csv` covers
  all 1,004 songs but lists only `악기구성` (sub-names) — **no group column**.
- 71470 codes recovered by **phrase-name mining** over all 9,945 label JSONs
  (`phrs_nm_kor` + `music_nm_kor`). Every code except `WN05` is confirmed by a direct name
  match in the phrase text, **not** inferred from clip counts.
- `방울` never appears in the 297 tagged songs → its publisher group is **inferred**.

---

## 5. Assignment record: which 71470 instrument went into which stem group

⚠️ **These assignments are by NAME, not by TIMBRE, and are provisional.** A solo clip routed
into a group because it shares an instrument name with a 71955 stem is not guaranteed to
belong there acoustically. This table exists so any instrument can be pulled back out and
regrouped on sonic grounds without re-deriving anything.

**The clearest candidate for later regrouping is 소금** — filed under 기타 purely because
that is where 71955 puts it, though it is a small transverse flute and may sit closer to
**대금** acoustically. 단소 (end-blown vertical flute) has the same question mark.

| Code | Instrument | Clips | Hours | → stem_group | Basis |
|---|---|---|---|---|---|
| `SP01` | 가야금 | 1,356 | 5.66 | **가야금** | same-name 71955 group |
| `SP02` | 거문고 | 825 | 3.26 | **거문고** | same-name 71955 group |
| `SR01` | 해금 | 1,291 | 5.60 | **해금** | same-name 71955 group |
| `SR02` | 아쟁 | 353 | 1.31 | **아쟁** | same-name 71955 group |
| `WN01` | 대금 | 816 | 3.77 | **대금** | same-name 71955 group |
| `WR01` | 피리 | 916 | 4.42 | **피리** | same-name 71955 group |
| `PT01` | 양금 | 72 | 0.25 | **양금** | same-name 71955 group |
| `WN02` | 소금 | 209 | 0.61 | **기타** | 71955 files 소금 under 기타 — ⚠️ regroup candidate (→ 대금?) |
| `WN03` | 단소 | 131 | 0.30 | **기타** | 71955 files 단소 under 기타 — ⚠️ regroup candidate |
| `WR02` | 태평소 | 154 | 0.58 | **기타** | 71955 files 태평소 under 기타 |
| `WR03` | 생황 | 145 | 0.65 | **기타** | 71955 files 생황 under 기타 |
| `PN01` | 장구 | 445 | 1.50 | **타악기** | unpitched perc, in 71955 타악기 |
| `PN03` | 북 | 122 | 0.31 | **타악기** | unpitched perc, in 71955 타악기 |
| `PN02` | 꽹과리 | 117 | 0.26 | **타악기** | unpitched perc, in 71955 타악기 |
| `PN06` | 목탁 | 2 | 0.00 | **타악기** | unpitched perc, in 71955 타악기 |
| `PT02` | 편종 | 17 | 0.04 | **pitched_percussion** | tuned idiophone (§2) |
| `PT03` | 편경 | 17 | 0.04 | **pitched_percussion** | tuned idiophone (§2) |
| `VF01` | 여성소리 | 1,647 | 6.21 | **voice** | new group (§3) |
| `VM02` | 남성소리 | 1,078 | 4.30 | **voice** | new group (§3) |
| `VH03` | 합창 | 10 | 0.04 | **voice** | new group (§3) |

**Assigned: 9,723 of 9,945 clips.** Note `방향` has no 71470 code — it exists only in 71955.

---

## 6. Held aside — assignment deliberately deferred (222 clips)

None of these has a 71955 counterpart, so no assignment follows from the ensemble scheme.
Resolution comes from **expert consultation + listening + lit review**; per instrument the
outcome is either (a) fold into an existing group, or (b) a new others/misc group. In the
manifest these carry `stem_group = null`, which the builder reports explicitly as
HELD ASIDE rather than letting it pass as a silent gap.

### 문묘제례악 ritual instruments (Confucian rite)

| Code | Instrument | Romanized | What it is | Clips | Minutes |
|---|---|---|---|---|---|
| `SP05` | 금 | *geum* | 7-string ritual zither (Chinese-derived) | 8 | 2.6 |
| `SP06` | 슬 | *seul* | 25-string ritual zither | 7 | 2.1 |
| `WN07` | 소 | *so* | panpipes | 6 | 1.8 |
| `WN06` | 지 | *ji* | transverse ritual flute | 4 | 1.0 |
| `PN10` | 축 | *chuk* | wooden box struck to **start** the rite | 1 | 0.2 |
| `PN11` | 어 | *eo* | tiger-shaped scraper to **end** the rite | 1 | 0.1 |

### Percussion with no 71955 counterpart

| Code | Instrument | Romanized | What it is | Clips | Minutes |
|---|---|---|---|---|---|
| `PN04` | 징 | *jing* | large suspended gong | 127 | 20.9 |
| `PN08` | 소고 | *sogo* | small handheld frame drum (농악) | 5 | 0.9 |
| `PN09` | 정주 | *jeongju* | small struck bowl-bell | 3 | 0.5 |
| `PN07` | 종 | *jong* | bell | 2 | 0.4 |

### Winds with no 71955 counterpart

| Code | Instrument | Romanized | What it is | Clips | Minutes |
|---|---|---|---|---|---|
| `WN04` | 퉁소 | *tungso* | end-blown notched flute | 53 | 8.3 |
| `WR04` | 나발 | *nabal* | long straight trumpet (대취타 signalling) | 1 | 0.2 |
| `WR05` | 나각 | *nagak* | conch horn (대취타 signalling) | 1 | 0.3 |

### Unresolved

| Code | Instrument | Clips | Minutes |
|---|---|---|---|
| `WN05` | *unknown* — all 3 clips have an empty `phrs_nm_kor` | 3 | 1.6 |

**징 and 퉁소 are 180 of the 222 held-aside clips (81%)** — resolving just those two clears
most of the backlog. 징 is unambiguously unpitched percussion, so 타악기 is the natural home;
퉁소 is the genuinely open one (a flute — 대금? 기타?). Everything else is single- or
low-double-digit clip counts, where the choice barely moves the pool either way.

---

## 7. Open items (→ Notion)

- **Raise with prof:** whether `pitched_percussion` earns a class at all, given it appears
  in only 17 songs — alternatives are dropping the class or folding it into 기타/misc.
- **Raise with prof:** whether 양금 stays its own group or merges with `pitched_percussion`.
- **Sonic regrouping pass** over the §5 assignments once listening + lit review are done —
  starting with 소금 and 단소.
- **Held-aside resolution** (§6) — expert consultation + listening.
- **Pitch-shift eligibility** — the `pitched` flag exists for this; unpitched percussion
  excluded is settled, 편종/편경/방향 still open.
- **Final revision** of this grouping happens after the taxonomy lit review.

## 8. Known gaps

- **`방울`** — publisher group inferred as 타악기 (absent from the 297 tagged songs).
- **비파** — absent from **both** datasets (0 clips). Sourcing is an open question → Notion.
- **피리1/2/3** — same-base multi-instrument tracks are digit-stripped and merged at ingest.
  Whether they are the same instrument or 향/세/당피리 family members is unconfirmed → Notion.
