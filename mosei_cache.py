"""
Precompute VideoMAE pixel tensors [T,C,H,W] per clip and write labels CSV with paths.

MOSEI/MOSI layout (default):
  - label.csv: video_id, clip_id, text, label, mode, ...
  - Raw: data/MOSEI/Raw/<video_id>/<clip_id>-edited.mp4 OR <clip_id>.mp4
  - Caches: data/MOSEI/video_caches/<safe_video_id>__<clip_id>.pt
  - Output: data/MOSEI/labels_with_video_cache.csv (+ video_cache_file, audio_file)
"""

from __future__ import annotations

import argparse
import os
import re
import torch
import pandas as pd

from utils.video_data import VideoMAEClipLoader

INVALID_SEGMENTS_MOSEI = {
    ("3aIQUQgawaI", "12"),
    ("94ULum9MYX0", "2"),
    ("mRnEJOLkhp8", "24"),
    ("aE-X_QdDaqQ", "3"),
    ("94ULum9MYX0", "11"),
    ("mRnEJOLkhp8", "26"),
}


def _norm_clip_id(clip_id) -> str:
    s = str(clip_id).strip()
    if s.isdigit():
        return str(int(s))
    return s


def _norm_video_id(video_id) -> str:
    return str(video_id).strip()


def _safe_cache_basename(video_id: str, clip_id: str) -> str:
    vid = re.sub(r"[^\w\-.]", "_", video_id)
    return f"{vid}__{clip_id}.pt"


def resolve_clip_mp4(raw_root: str, video_id: str, clip_id: str) -> str | None:
    folder = os.path.join(raw_root, video_id)
    for name in (f"{clip_id}-edited.mp4", f"{clip_id}.mp4"):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label-csv", default="data/MOSEI/label.csv")
    p.add_argument("--raw-root", default="data/MOSEI/Raw")
    p.add_argument("--cache-root", default="data/MOSEI/video_caches")
    p.add_argument("--out-csv", default="data/MOSEI/labels_with_video_cache.csv")
    p.add_argument("--mode", default="all", choices=("all", "train", "valid", "test"))
    p.add_argument(
        "--no-skip-invalid-mosei",
        action="store_true",
        help="Also try to cache the segments Dataset_mosi marks invalid.",
    )
    args = p.parse_args()

    os.makedirs(args.cache_root, exist_ok=True)
    df = pd.read_csv(args.label_csv)
    required = {"video_id", "clip_id", "text", "label", "mode"}
    miss = required - set(df.columns)
    if miss:
        raise SystemExit(f"label.csv missing columns: {miss}")
    if args.mode != "all":
        df = df[df["mode"] == args.mode].reset_index(drop=True)

    loader = VideoMAEClipLoader()
    kept_rows: list[int] = []
    cache_paths: list[str] = []
    missing = failed_decode = skipped_invalid = 0
    skip_invalid = not args.no_skip_invalid_mosei

    for i, row in df.iterrows():
        vid = _norm_video_id(row["video_id"])
        cid = _norm_clip_id(row["clip_id"])
        if skip_invalid and (vid, cid) in INVALID_SEGMENTS_MOSEI:
            skipped_invalid += 1
            continue
        mp4 = resolve_clip_mp4(args.raw_root, vid, cid)
        if mp4 is None:
            missing += 1
            print(f"[skip missing] .../{vid}/{{{cid}-edited.mp4,{cid}.mp4}}")
            continue
        try:
            pixel_values = loader.load_pixel_values(mp4)
        except Exception as e:
            failed_decode += 1
            print(f"[skip decode error] {mp4} ({type(e).__name__}: {e})")
            continue
        base = _safe_cache_basename(vid, cid)
        torch.save(pixel_values, os.path.join(args.cache_root, base))
        kept_rows.append(i)
        cache_paths.append(base)

    out_df = df.iloc[kept_rows].copy()
    out_df["video_cache_file"] = cache_paths
    out_df["audio_file"] = [
        f"{_norm_video_id(out_df['video_id'].iloc[k])}/{_norm_clip_id(out_df['clip_id'].iloc[k])}.wav"
        for k in range(len(out_df))
    ]
    out_df.to_csv(args.out_csv, index=False)
    print(f"Saved {args.out_csv}")
    print(
        f"cached={len(out_df)}/{len(df)} missing={missing} "
        f"decode_errors={failed_decode} skipped_invalid={skipped_invalid}"
    )


if __name__ == "__main__":
    main()