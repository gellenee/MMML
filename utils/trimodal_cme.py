from torch import nn
from utils.cross_attn_encoder import CMELayer, BertConfig

class TriModalCME(nn.Module):
    """
    Uses the existing 2-stream CMELayer to support:
      - TA: text<->audio
      - TV: text<->video
      - TAV: runs TA then TV (optionally AV)

    This avoids rewriting CMELayer.
    """
    def __init__(self, num_layers: int, hidden_size: int = 768, include_av: bool = False):
        super().__init__()
        cfg = BertConfig(num_hidden_layers=num_layers, hidden_size=hidden_size)
        self.layers = nn.ModuleList([CMELayer(cfg) for _ in range(num_layers)])
        self.include_av = include_av

    def forward(
        self,
        t_tokens, t_mask,
        a_tokens=None, a_mask=None,
        v_tokens=None, v_mask=None,
        modalities: str = "TV"  # "TA", "TV", "TAV"
    ):
        use_a = ("A" in modalities)
        use_v = ("V" in modalities)

        for layer in self.layers:
            if use_a and a_tokens is not None:
                t_tokens, a_tokens = layer(t_tokens, t_mask, a_tokens, a_mask)
            if use_v and v_tokens is not None:
                t_tokens, v_tokens = layer(t_tokens, t_mask, v_tokens, v_mask)
            if self.include_av and use_a and use_v and a_tokens is not None and v_tokens is not None:
                a_tokens, v_tokens = layer(a_tokens, a_mask, v_tokens, v_mask)

        return t_tokens, a_tokens, v_tokens