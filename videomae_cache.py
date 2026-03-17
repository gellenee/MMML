import os
import torch
import pandas as pd

from utils.video_data import VideoMAEClipLoader

RAW_ROOT = "data/VCE_CUSTOM/raw"
CACHE_ROOT = "data/VCE_CUSTOM/videomae_cache"
LABEL_CSV = "data/VCE_CUSTOM/labels.csv"
OUT_CSV = "data/VCE_CUSTOM/labels_with_video_cache.csv"


def main():
    os.makedirs(CACHE_ROOT, exist_ok=True)

    df = pd.read_csv(LABEL_CSV)
    loader = VideoMAEClipLoader()

    cache_paths = []

    for i, row in df.iterrows():
        vid = str(row["video_id"])
        mp4 = os.path.join(RAW_ROOT, vid + ".mp4")  # adjust if nested dirs

        if not os.path.isfile(mp4):
            raise FileNotFoundError(f"Video file not found: {mp4}")

        pixel_values = loader.load_pixel_values(mp4)  # [T,C,H,W]

        out_path = os.path.join(CACHE_ROOT, vid + ".pt")
        torch.save(pixel_values, out_path)

        # store relative path under CACHE_ROOT
        cache_paths.append(os.path.relpath(out_path, CACHE_ROOT))

    df["video_cache_file"] = cache_paths
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved cache CSV to {OUT_CSV}")


if __name__ == "__main__":
    main()