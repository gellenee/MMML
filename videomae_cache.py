import os
import torch
import pandas as pd

from utils.video_data import VideoMAEClipLoader

RAW_ROOT = "data/VCE_CUSTOM/Raw"
CACHE_ROOT = "data/VCE/videomae_cache"
LABEL_CSV = "data/VCE_CUSTOM/label.csv"
OUT_CSV = "data/VCE/labels_with_video_cache.csv"


def main():
    os.makedirs(CACHE_ROOT, exist_ok=True)

    df = pd.read_csv(LABEL_CSV)
    loader = VideoMAEClipLoader()

    kept_rows = []
    cache_paths = []
    missing = 0
    failed_decode = 0

    for i, row in df.iterrows():
        vid_raw = str(row["video_id"])
        # If raw videos are zero-padded (e.g. 00042.mp4), pad numeric IDs to 5 digits.
        vid = vid_raw.zfill(5) if vid_raw.isdigit() else vid_raw
        mp4 = os.path.join(RAW_ROOT, vid + ".mp4")  # adjust if nested dirs

        if not os.path.isfile(mp4):
            missing += 1
            print(f"[skip missing] {mp4}")
            continue

        try:
            pixel_values = loader.load_pixel_values(mp4)  # [T,C,H,W]
        except Exception as e:
            failed_decode += 1
            print(f"[skip decode error] {mp4} ({type(e).__name__}: {e})")
            continue

        out_path = os.path.join(CACHE_ROOT, vid + ".pt")
        torch.save(pixel_values, out_path)

        # store relative path under CACHE_ROOT
        kept_rows.append(i)
        cache_paths.append(os.path.relpath(out_path, CACHE_ROOT))

    out_df = df.iloc[kept_rows].copy()
    out_df["video_cache_file"] = cache_paths
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved cache CSV to {OUT_CSV}")
    print(
        f"Done. cached={len(out_df)}/{len(df)} "
        f"(missing={missing}, decode_errors={failed_decode})."
    )


if __name__ == "__main__":
    main()