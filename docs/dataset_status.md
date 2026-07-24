# 🎼 Gugak Ensemble Stem-Separation Dataset — Status

> **Source:** AI Hub · *국악합주곡 디지털 음원 데이터* (datasetkey `71955`)
**Status:** ✅ Downloaded → extracted → verified → **manifested**. Ready for audio EDA.
**Last updated:** 2026-07-21
> 

| Metric | Value |
| --- | --- |
| Songs (our working set) | **903** |
| — master mixes | 903 (1 per song) |
| — instrument stems | 5,767 |
| **Total audio tracks (WAV)** | **6,670** |
| Total audio | ~57 h (masters) · ~369 h (stems) |
| Stems per song | 1 – 18 (median 7) |
| Audio format | 48 kHz / 24-bit / stereo *(130 files at 96 kHz — see notes)* |
| Split (ours, seed 42) | **train 721 · val 91 · test 91** |

### **Songs available & completeness**

- The published dataset lists 1,004 songs, but the publisher withholds the audio for the 101-song test split (metadata only, not downloadable).
    
    → Available audio = 903 songs. Our dataset is branched from exactly these.
    
    - Every one of the 903 songs has its master mix and all of its stems on disk + metadata

### Genre distribution (`genreSub`)

| Genre | Songs |
| --- | --- |
| 판소리 (pansori) | 269 |
| 산조 (sanjo) | 200 |
| 창작국악 (contemporary) | 187 |
| 풍류음악 (pungnyu) | 104 |
| 민요 (folk song) | 75 |
| 궁중음악 (court music) | 47 |
| 대풍류 (daepungnyu) | 21 |

### Instruments

9 groups / 24 sub-instruments, long-tailed. Stems per group:
타악기 1,142 · 피리 762 · 대금 759 · 해금 718 · 아쟁 709 · 가야금 692 · 거문고 636 · 기타 254 · 양금 95.

- No vocal / 소리 stem exists anywhere — even 판소리 is represented instrumentally (its masters also appear voiceless; to be confirmed by the master-vs-sum test).
- Some songs contain multiple players of one instrument (e.g. 피리1 / 피리2 / 피리3).

### Data notes (to handle in preprocessing)

- 96 kHz outliers: 130 tracks (≈10 창작국악 songs) are 96 kHz rather than 48 kHz → resample to a common rate.
- 판소리 length mismatches: 28 songs where the master is slightly longer than its stems (~0.05–0.5 s) → align/trim before summing.

### Note — publisher split files (mystery solved 2026-07-21)

The publisher ships three overlapping, mutually-inconsistent indexes — `genre_split.csv`, `instrument_split.csv`, and the `Training`/`Validation` folders. They disagree on which songs are train/val/test, and each omits a different subset of songs (e.g. `genre_split.csv` covers only 838 of our 903).

**We now know why.** The two CSVs are not a designed dataset split at all — they are **artifacts generated at training time by the publisher's two validation scripts** (`train.py` and `train_instrument.py` in the submitted evaluation package, confirmed via the package README). Each script performs its own independent 9:1 random hold-out for its task (genre tagging vs instrument tagging), so the two files were never meant to agree with each other, nor with the download folders.

Rather than inherit that confusion, we branched a clean dataset off the available audio — the one complete, self-consistent source — and generated our own reproducible, genre-stratified split (seed 42, 80/10/10). A single frozen manifest (`manifests/songs.parquet` + `stems.parquet`) is now the source of truth for everything downstream. The discovery above retroactively confirms this was the right call: no canonical split ever existed to inherit.

### Note — publisher validation models (checked 2026-07-21)

The dataset's official validation ("AI 모델 개발") consists of **tagging models only**: two ResNet(18)-based classifiers by 뉴튠 — genre tagging on master audio and instrument tagging on per-stem audio, each with an ACC-70% target and a 9:1 hold-out. Documentation is sparse (no detailed methodology or reported results in the distributed manual).

→ **No source-separation model or baseline exists for this dataset.** Our project establishes the first separation results (and the first separation eval protocol) on it. Full prior-art context: Notion → *Survey: gugak_stem_separation* → Strand 0.