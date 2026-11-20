import argparse
import csv
import os
import random
import sys
import time
import multiprocessing as mp

import cv2
import numpy as np

SPLITS = ("train", "val", "test")
CLAHE_CLIP, CLAHE_TILE = 2.0, (8, 8)
UNSHARP_SIGMA, UNSHARP_AMOUNT = 1.0, 0.5


def cache_path(cache_dir, label, photo_id):
    return os.path.join(cache_dir, label.replace(" ", "_"), f"{photo_id}.jpg")


def read_rows(splits_dir, splits=SPLITS):
    rows = []
    for s in splits:
        with open(os.path.join(splits_dir, f"{s}.csv"), encoding="utf-8-sig", newline="") as f:
            rows += [dict(r, split=s) for r in csv.DictReader(f)]
    return rows


def load_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def apply_ip(bgr, sigma=UNSHARP_SIGMA):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), sigma)
    return cv2.addWeighted(out, 1 + UNSHARP_AMOUNT, blur, -UNSHARP_AMOUNT, 0)


def _write_jpeg(path, img, quality):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.jpg"
    if not cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"cannot write {path}")
    os.replace(tmp, path)


def _work(job):
    src, dst, quality = job
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return "skip", dst
    try:
        _write_jpeg(dst, apply_ip(load_bgr(src)), quality)
        return "ok", dst
    except Exception as e:
        return f"failed: {type(e).__name__}: {e}", dst


def _init_worker():
    cv2.setNumThreads(1)


def make_sheet(rows, path, n, seed):
    rng = random.Random(seed)
    tiles = []
    for r in rng.sample(rows, min(n, len(rows))):
        src = load_bgr(r["path"])
        pair = []
        for im in (src, apply_ip(src)):
            h, w = im.shape[:2]
            s = 240 / h
            pair.append(cv2.resize(im, (int(w * s), 240), interpolation=cv2.INTER_AREA))
        pair = np.hstack([pair[0], np.full((240, 6, 3), 255, np.uint8), pair[1]])
        for color, width in (((255, 255, 255), 2), ((0, 0, 0), 1)):
            cv2.putText(pair, r["label"], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, width, cv2.LINE_AA)
        tiles.append(pair)
    width = max(t.shape[1] for t in tiles)
    tiles = [np.hstack([t, np.full((240, width - t.shape[1], 3), 255, np.uint8)]) for t in tiles]
    grid = np.vstack([np.vstack([t, np.full((6, width, 3), 255, np.uint8)]) for t in tiles])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, grid)
    print(f"saved {len(tiles)} before/after examples: {path} (left = original, right = after IP)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits_dir", required=True, help="folder with train.csv, val.csv, test.csv")
    ap.add_argument("--out_dir", required=True, help="output folder, e.g. cache/ip500")
    ap.add_argument("--splits", default="train,val,test", help="comma-separated splits to process")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality of the written images")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit", type=int, default=0, help="testing: process only the first N images")
    ap.add_argument("--sheet", default="", help="save a before/after IP sheet as PNG")
    ap.add_argument("--sheet_n", type=int, default=12)
    ap.add_argument("--sheet_only", action="store_true", help="only make the sheet, do not write the cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)

    rows = read_rows(args.splits_dir, tuple(args.splits.split(",")))
    if args.sheet:
        make_sheet(rows, args.sheet, args.sheet_n, args.seed)
        if args.sheet_only:
            return
    if args.limit:
        rows = rows[:args.limit]
    jobs = [(r["path"], cache_path(out_dir, r["label"], r["photo_id"]), args.quality) for r in rows]
    print(f"IP: {len(jobs):,} images -> {out_dir} ({args.workers} processes)", flush=True)

    t0, n_ok, n_skip, failed = time.time(), 0, 0, []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for i, (st, dst) in enumerate(pool.imap_unordered(_work, jobs, chunksize=64), 1):
            if st == "ok":
                n_ok += 1
            elif st == "skip":
                n_skip += 1
            else:
                failed.append((dst, st))
            if i % 10000 == 0 or i == len(jobs):
                rate = i / (time.time() - t0)
                print(f"  {i:,}/{len(jobs):,} | written {n_ok:,} skipped {n_skip:,} failed {len(failed)} | "
                      f"{rate:.0f} img/s, about {(len(jobs) - i) / rate / 60:.0f} min left", flush=True)

    if failed:
        log = os.path.join(out_dir, "failed.txt")
        with open(log, "w", encoding="utf-8") as f:
            f.writelines(f"{d}\t{s}\n" for d, s in failed)
        print(f"{len(failed)} images failed, see {log}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
