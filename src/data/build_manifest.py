#!/usr/bin/env python3
"""Build the frozen manifest for the gugak stem-separation dataset.

Source of truth = the filesystem (`data/extracted/<dl_folder>/source/<song>/`) joined with
`data/metadata/메타데이터.csv` (per-song metadata, covers all 903 downloaded songs).

Split: we IGNORE the publisher's `genre_split.csv` / `instrument_split.csv` / Training-
Validation folders (mutually inconsistent, incomplete — see CLAUDE.md) and generate OUR OWN
song-level split, stratified by genreSub, fixed seed. `split` is a logical label; the audio
files are not moved (manifest paths point to their real location).

Outputs (parquet = pipeline-authoritative, csv = human/git mirror) to `manifests/` (committed)
and `data/metadata/`:  songs.parquet/csv (903 rows), stems.parquet/csv (5767 rows).

Run:  uv run python src/data/build_manifest.py
"""
from __future__ import annotations
import csv, glob, io, json, os, random, re, sys, unicodedata
from pathlib import Path
import pandas as pd
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
EXTRACTED = REPO / "data" / "extracted"
META_DIR = REPO / "data" / "metadata"
OUT_DIRS = [REPO / "manifests", META_DIR]

# --- split config (freeze these; changing them changes the frozen split) ---
SEED = 42
VAL_FRAC = 0.10
TEST_FRAC = 0.10   # train gets the remainder

N = lambda s: unicodedata.normalize("NFC", s)                 # Korean NFC-normalize keys
BASE = lambda inst: re.sub(r"\d+$", "", inst)                 # 피리3 -> 피리 (strip player index)


def read_csv_rows(path: Path) -> list[dict]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp949", "euc-kr", "utf-8"):
        try:
            return list(csv.DictReader(io.StringIO(raw.decode(enc))))
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"could not decode {path}")


# provisional 4-stem scheme (major -> group); revisit in Phase 5 (기타 = winds grab-bag)
MAJOR_TO_4STEM = {
    "타악기": "타악",
    "대금": "관악", "피리": "관악", "기타": "관악",
    "해금": "찰현", "아쟁": "찰현",
    "가야금": "발현", "거문고": "발현", "양금": "발현",
}


def build_sub_to_major() -> dict[str, str]:
    """Derive instrument-sub(base) -> major from JSON tags + instrument_split labels."""
    acc: dict[str, set] = {}
    for r in read_csv_rows(META_DIR / "instrument_split.csv"):
        major, sub = r["label"].split("_", 1)
        acc.setdefault(N(BASE(sub)), set()).add(N(major))
    for jp in glob.glob(str(EXTRACTED / "*/labels/*.json")):
        for s in json.loads(Path(jp).read_text(encoding="utf-8"))["stems"]:
            t = s["tags"]
            acc.setdefault(N(BASE(t["instrumentSub"])), set()).add(N(t["instrumentMajor"]))
    mapping, conflicts = {}, {}
    for sub, majors in acc.items():
        (mapping if len(majors) == 1 else conflicts)[sub] = (
            next(iter(majors)) if len(majors) == 1 else majors)
    if conflicts:
        print(f"  ! sub->major conflicts: {conflicts}")
    return mapping


def stratified_split(songs: pd.DataFrame) -> dict[str, str]:
    """Song-level split stratified by genre_sub; deterministic given SEED."""
    rng = random.Random(SEED)
    out = {}
    for genre, grp in songs.groupby("genre_sub", sort=True):
        ids = sorted(grp["song_id"].tolist())
        rng.shuffle(ids)
        n = len(ids)
        n_test, n_val = round(n * TEST_FRAC), round(n * VAL_FRAC)
        for i, sid in enumerate(ids):
            out[sid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def probe(path: Path, errors: list) -> dict:
    try:
        i = sf.info(str(path))
        return dict(sr=i.samplerate, ch=i.channels, frames=i.frames,
                    dur_s=round(i.frames / i.samplerate, 3), subtype=i.subtype)
    except Exception as e:
        errors.append((str(path), repr(e)))
        return dict(sr=None, ch=None, frames=None, dur_s=None, subtype=None)


def main() -> int:
    meta = {N(r["project_name"]): r for r in read_csv_rows(META_DIR / "메타데이터.csv")}
    sub2major = build_sub_to_major()
    print(f"loaded: 메타데이터={len(meta)} songs | sub(base)->major={len(sub2major)}")

    song_rows, stem_rows, header_errors, unmapped = [], [], [], set()
    for d in sorted(g for g in glob.glob(str(EXTRACTED / "*/source/*")) if os.path.isdir(g)):
        song = N(os.path.basename(d))
        m = meta[song]                                         # all 903 disk songs are in meta
        num_id, genre_sub = song.split("_", 1)[0], m["genreSub"]
        is_pansori = genre_sub == "판소리"

        stem_fps = sorted(p for p in glob.glob(f"{d}/*.wav") if not p.endswith("_master.wav"))
        instruments = []
        for sp in stem_fps:
            sub = N(os.path.basename(sp))[len(song) + 1:-4]    # <song>_<sub>.wav
            instruments.append(sub)
            major = sub2major.get(BASE(sub))
            if major is None:
                unmapped.add(sub)
            stem_rows.append(dict(
                song_id=song, num_id=num_id, genre_sub=genre_sub,
                instrument=sub, instrument_base=BASE(sub), instrument_major=major,
                stem_group_4=MAJOR_TO_4STEM.get(major), is_pansori=is_pansori,
                stem_path=str(Path(sp).relative_to(REPO)), **probe(Path(sp), header_errors)))

        master_fp = Path(d) / f"{song}_master.wav"
        mh = probe(master_fp, header_errors)
        song_rows.append(dict(
            song_id=song, num_id=num_id,
            genre_major=m["genreMajor"], genre_sub=genre_sub, genre_detail=m["genreDetail"],
            tempo_bpm=m["tempoBpm"], time_signature=m["timeSignature"], western_key=m["western_key"],
            moods=m["mood_count"],
            duration_meta_s=float(m["오디오 길이(초)"]) if m.get("오디오 길이(초)") else None,
            n_stems=len(stem_fps), instruments=sorted(instruments),
            instruments_meta=sorted(x.strip() for x in m.get("악기구성", "").split(",") if x.strip()),
            is_pansori=is_pansori, master_path=str(master_fp.relative_to(REPO)),
            master_sr=mh["sr"], master_ch=mh["ch"], master_frames=mh["frames"],
            master_dur_s=mh["dur_s"], master_subtype=mh["subtype"]))

    songs = pd.DataFrame(song_rows).sort_values("song_id").reset_index(drop=True)
    stems = pd.DataFrame(stem_rows).sort_values(["song_id", "instrument"]).reset_index(drop=True)

    # our own stratified split
    split_map = stratified_split(songs)
    songs["split"] = songs["song_id"].map(split_map)
    stems["split"] = stems["song_id"].map(split_map)

    # --- validation ---
    print("\n=== VALIDATION ===")
    print(f"songs: {len(songs)} | stems: {len(stems)}")
    print(f"split totals: {songs['split'].value_counts().to_dict()}")
    print("per-genre x split:")
    print(pd.crosstab(songs["genre_sub"], songs["split"]).loc[:, ["train", "val", "test"]])
    print(f"songs n_stems != len(악기구성): {int((songs['n_stems'] != songs['instruments_meta'].str.len()).sum())}")
    print(f"unmapped instrument->major: {sorted(unmapped)}")
    print(f"header read errors: {len(header_errors)}")
    print(f"sample-rate counts (stems): {stems['sr'].value_counts().to_dict()} | masters: {songs['master_sr'].value_counts().to_dict()}")
    fr = stems.groupby("song_id")["frames"].agg(["min", "max"])
    mf = songs.set_index("song_id")["master_frames"]
    misalign = [s for s in fr.index if fr.loc[s, "min"] != fr.loc[s, "max"] or mf.get(s) != fr.loc[s, "max"]]
    print(f"songs with master/stem length mismatch: {len(misalign)} (recorded; handle in preprocessing)")

    # --- write ---
    print("\n=== WRITING ===")
    for od in OUT_DIRS:
        od.mkdir(parents=True, exist_ok=True)
        songs.to_parquet(od / "songs.parquet", index=False)
        stems.to_parquet(od / "stems.parquet", index=False)
        songs.to_csv(od / "songs.csv", index=False)
        stems.to_csv(od / "stems.csv", index=False)
        print(f"  wrote songs/stems .parquet+.csv -> {od.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
