import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import VideoMAEImageProcessor

# preprocess video files to have 16 frames per video
# resizing to 224x224, normalizing pixel values
# rearranging the dimensions into the  (Time, Channels, Height, Width) required by VideoMAE encoder
from torchcodec.decoders import VideoDecoder

VIDEO_ROOT = "data/VCE_CUSTOM/videomae_cache" #reading from the cache folder
NUM_FRAMES = 16
SIZE = 224

class VideoMAEClipLoader:
    def __init__(self):
        self.processor = VideoMAEImageProcessor(
            size={"shortest_edge": SIZE},
            crop_size={"height": SIZE, "width": SIZE},
        )

    def _uniform_indices(self, n, k):
        if n <= 0:
            return np.zeros((k,), dtype=np.int64)
        if n >= k:
            return np.linspace(0, n - 1, k).astype(np.int64)
        # repeat last frame if too short
        base = np.arange(n, dtype=np.int64)
        pad = np.full((k - n,), base[-1], dtype=np.int64)
        return np.concatenate([base, pad], axis=0)

    def load_pixel_values(self, mp4_path: str) -> torch.Tensor:
        """
        Returns pixel_values shaped [T, C, H, W] for VideoMAE.
        """
        from torchcodec.decoders import VideoDecoder

        decoder = VideoDecoder(mp4_path)

        # number of frames available (VideoDecoder behaves like a sequence)
        n = len(decoder)

        if n == 0:
            frames_np = np.zeros((NUM_FRAMES, SIZE, SIZE, 3), dtype=np.uint8)
        else:
            idx = self._uniform_indices(n, NUM_FRAMES).tolist()
            frames = decoder.get_frames_at(indices=idx)  # Tensor, batch of frames

            # Convert to what VideoMAEImageProcessor expects:
            # Depending on torchcodec version, frames are commonly [T,H,W,C] uint8.
            # If you get [T,C,H,W], do frames = frames.permute(0,2,3,1).
            if frames.ndim == 4 and frames.shape[1] == 3:
                frames = frames.permute(0, 2, 3, 1).contiguous()

            frames_np = frames.cpu().numpy()  # [T,H,W,C], uint8

        processed = self.processor(list(frames_np), return_tensors="pt")
        return processed["pixel_values"].squeeze(0)  # [T,C,H,W]

# creates dataset to be used by the dataloader for video-text training
class Dataset_vce_custom_text_video(Dataset):
    """
    Uses precomputed VideoMAE clips from data/VCE_CUSTOM/videomae_cache.
    Expects CSV `data/VCE_CUSTOM/labels_with_video_cache.csv` to include at least:
      - mode: train/valid/test
      - text
      - label
      - video_cache_file: relative .pt path under VIDEO_ROOT (e.g. "123.pt")
    """
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
       
        text = str(row["text"])
        tok = self.tokenizer(
            text,
            max_length=96,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        #vid from cache
        rel = str(row["video_cache_file"])  # e.g. "123.pt"
        cache_path = os.path.join(VIDEO_ROOT, rel)
        pixel_values = torch.load(cache_path)  # [T,C,H,W]
        return {
            "text_tokens": torch.tensor(tok["input_ids"], dtype=torch.long),
            "text_masks": torch.tensor(tok["attention_mask"], dtype=torch.long),
            "video_pixel_values": pixel_values,  # [T,C,H,W]
            "targets": torch.tensor(float(row["label"]), dtype=torch.float),
        }
def collate_vce_text_video(batch):
    return {
        "text_tokens": torch.stack([b["text_tokens"] for b in batch], dim=0),
        "text_masks": torch.stack([b["text_masks"] for b in batch], dim=0),
        "video_pixel_values": torch.stack([b["video_pixel_values"] for b in batch], dim=0),  # [B,T,C,H,W]
        "targets": torch.stack([b["targets"] for b in batch], dim=0), #sentiment label
    }