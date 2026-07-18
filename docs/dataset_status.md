# 🎼 Gugak Ensemble Stem-Separation Dataset — Status

> **Source:** AI Hub · *국악합주곡 디지털 음원 데이터* (datasetkey `71955`)
**Status:** ✅ Downloaded → extracted → verified → **manifested**. Ready for audio EDA.
**Last updated:** 2026-07-18
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
    
    - Every one of the 903 songs has its master mix and all of its stems on dis + metadata

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

### Note

The publisher ships three overlapping, mutually-inconsistent indexes — `genre_split.csv`, `instrument_split.csv`, and the `Training`/`Validation` folders. They disagree on which songs are train/val/test, and each omits a different subset of songs (e.g. `genre_split.csv` covers only 838 of our 903). Rather than inherit that confusion, we branched a clean dataset off the available audio — the one complete, self-consistent source — and generated our own reproducible, genre-stratified split (seed 42, 80/10/10). A single frozen manifest (`manifests/songs.parquet` + `stems.parquet`) is now the source of truth for everything downstream.