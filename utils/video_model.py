import torch
from torch import nn
from transformers import VideoMAEModel

class VideoMAEEncoder(nn.Module):
    """
    Encodes a fixed number of frames into token features.
    Expects pixel_values shaped [B, T, C, H, W] (HF VideoMAE convention).
    Returns:
      - tokens: [B, S, 768]
      - pooled: [B, 768] (CLS token)
    """
    def __init__(self, pretrained_name: str = "MCG-NJU/videomae-base"):
        super().__init__()
        self.model = VideoMAEModel.from_pretrained(pretrained_name)

    def forward(self, pixel_values: torch.Tensor):
        out = self.model(pixel_values=pixel_values, return_dict=True)
        tokens = out.last_hidden_state                  # [B, S, 768]
        pooled = tokens[:, 0, :]                        # CLS
        return tokens, pooled