#!/usr/bin/env python3
"""Extract all gugak dataset zips -> parallel `extracted/` tree. Resumable, parallel, progress-logged.
Keeps original zips intact (user decides later whether to discard)."""
import os, sys, glob, time, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = "/home/jae.gye/storage/nia-gugak"
SRC  = f"{BASE}/29.국악_합주곡_디지털_음원_데이터/3.개방데이터/2.데이터(NIA)"
OUT  = f"{BASE}/extracted"
MARK = f"{OUT}/.markers"

# (zip glob, output subdir)
JOBS = [
    (f"{SRC}/Training/01.원천데이터/TS_*.zip",   f"{OUT}/train/source"),
    (f"{SRC}/Training/02.라벨링데이터/TL_*.zip", f"{OUT}/train/labels"),
    (f"{SRC}/Validation/01.원천데이터/VS_*.zip", f"{OUT}/val/source"),
    (f"{SRC}/Validation/02.라벨링데이터/VL_*.zip", f"{OUT}/val/labels"),
]

def extract_one(task):
    zpath, outdir = task
    marker = os.path.join(MARK, os.path.basename(zpath) + ".done")
    if os.path.exists(marker):
        return (zpath, "skip", 0)
    try:
        os.makedirs(outdir, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            n = len(zf.namelist())
            zf.extractall(outdir)
        with open(marker, "w") as f:
            f.write("ok\n")
        return (zpath, "ok", n)
    except Exception as e:
        return (zpath, f"ERROR: {e}", 0)

def main():
    os.makedirs(MARK, exist_ok=True)
    tasks = []
    for pat, outdir in JOBS:
        for z in sorted(glob.glob(pat)):
            tasks.append((z, outdir))
    total = len(tasks)
    workers = min(12, (os.cpu_count() or 4))
    print(f"[{time.strftime('%H:%M:%S')}] extracting {total} zips -> {OUT}  ({workers} workers)", flush=True)

    done = ok = skip = err = files = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(extract_one, t) for t in tasks]
        for fut in as_completed(futs):
            zpath, status, n = fut.result()
            done += 1
            if status == "ok":   ok += 1;  files += n
            elif status == "skip": skip += 1
            else:
                err += 1
                print(f"  !! {os.path.basename(zpath)} -> {status}", flush=True)
            if done % 25 == 0 or done == total:
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (total - done) / rate if rate else 0
                print(f"[{time.strftime('%H:%M:%S')}] {done}/{total}  ok={ok} skip={skip} err={err}  "
                      f"files={files}  {rate:.1f} zip/s  ETA {eta/60:.1f} min", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] DONE  ok={ok} skip={skip} err={err}  "
          f"extracted_files={files}  elapsed={ (time.time()-t0)/60:.1f} min", flush=True)
    sys.exit(1 if err else 0)

if __name__ == "__main__":
    main()
