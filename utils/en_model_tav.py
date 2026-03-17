import torch
from torch import nn
from transformers import RobertaModel, Data2VecAudioModel
from utils.video_model import VideoMAEEncoder
from utils.trimodal_cme import TriModalCME

class rob_d2v_videomae_cme(nn.Module):
    """
    English tri-modal (Text/Audio/Video) model that can run CME over TA / TV / TAV.
    Returns dict keys consistent with existing trainers: 'T', 'A', 'M' plus optional 'V'.
    """
    def __init__(self, config):
        super().__init__()
        self.modalities = config.modalities  # e.g. "TA", "TV", "TAV"
        self.hidden_size = 768

        # Text encoder
        self.roberta = RobertaModel.from_pretrained("roberta-base")

        # Audio encoder (keep existing)
        self.data2vec = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")

        # Video encoder (new)
        self.videomae = VideoMAEEncoder(pretrained_name=getattr(config, "videomae_name", "MCG-NJU/videomae-base"))

        # Learnable CLS tokens for CME streams
        self.text_cls = nn.Embedding(1, self.hidden_size)
        self.audio_cls = nn.Embedding(1, self.hidden_size)
        self.video_cls = nn.Embedding(1, self.hidden_size)

        # CME wrapper
        self.cme = TriModalCME(num_layers=config.num_hidden_layers, hidden_size=self.hidden_size, include_av=getattr(config, "include_av", False))

        # Heads
        self.T_head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(self.hidden_size, 1))
        self.A_head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(self.hidden_size, 1))
        self.V_head = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(self.hidden_size, 1))

        # Fused M head
        # dimension depends on which modalities are active
        fused_in = self.hidden_size * (1 + ("A" in self.modalities) + ("V" in self.modalities))
        self.M_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(fused_in, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def _prepend_cls(self, x, mask, emb):
        # x: [B,S,D], mask: [B,S]
        idx = torch.zeros(1, dtype=torch.long, device=x.device)
        cls = emb(idx).expand(x.size(0), 1, x.size(2))
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones(x.size(0), 1, device=mask.device, dtype=mask.dtype)
        mask = torch.cat([cls_mask, mask], dim=1)
        return x, mask

    def forward(
        self,
        text_inputs, text_mask,
        audio_inputs=None, audio_mask=None,
        video_pixel_values=None, video_mask=None
    ):
        # --- Text ---
        t_out = self.roberta(text_inputs, text_mask, return_dict=True)
        t_tokens = t_out.last_hidden_state            # [B, St, 768]
        t_pooled = t_out["pooler_output"]             # [B, 768]
        t_pred = self.T_head(t_pooled)

        # --- Audio (optional) ---
        a_pred, a_tokens, a_mask_used, a_pooled = None, None, None, None
        if "A" in self.modalities:
            a_out = self.data2vec(audio_inputs, audio_mask, return_dict=True)
            a_tokens = a_out.last_hidden_state        # [B, Sa, 768]
            # simplest pooling: mean over masked frames
            m = audio_mask.unsqueeze(-1).float()      # [B, Sa, 1]
            a_pooled = (a_tokens * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
            a_pred = self.A_head(a_pooled)
            a_mask_used = audio_mask

        # --- Video (optional) ---
        v_pred, v_tokens, v_mask_used, v_pooled = None, None, None, None
        if "V" in self.modalities:
            v_tokens, v_pooled = self.videomae(video_pixel_values)   # tokens [B,Sv,768], pooled [B,768]
            v_pred = self.V_head(v_pooled)
            # If you don't have a natural mask, use ones
            if video_mask is None:
                v_mask_used = torch.ones(v_tokens.size(0), v_tokens.size(1), device=v_tokens.device, dtype=torch.long)
            else:
                v_mask_used = video_mask

        # --- CME (token-level fusion) ---
        t_tokens_cme, t_mask_cme = self._prepend_cls(t_tokens, text_mask, self.text_cls)

        if a_tokens is not None:
            a_tokens_cme, a_mask_cme = self._prepend_cls(a_tokens, a_mask_used, self.audio_cls)
        else:
            a_tokens_cme, a_mask_cme = None, None

        if v_tokens is not None:
            v_tokens_cme, v_mask_cme = self._prepend_cls(v_tokens, v_mask_used, self.video_cls)
        else:
            v_tokens_cme, v_mask_cme = None, None

        t_tokens_cme, a_tokens_cme, v_tokens_cme = self.cme(
            t_tokens_cme, t_mask_cme,
            a_tokens=a_tokens_cme, a_mask=a_mask_cme,
            v_tokens=v_tokens_cme, v_mask=v_mask_cme,
            modalities=self.modalities
        )

        # use CLS from CME outputs for fusion
        fused_parts = [t_tokens_cme[:, 0, :]]
        if a_tokens_cme is not None:
            fused_parts.append(a_tokens_cme[:, 0, :])
        if v_tokens_cme is not None:
            fused_parts.append(v_tokens_cme[:, 0, :])
        fused = torch.cat(fused_parts, dim=1)
        m_pred = self.M_head(fused)

        out = {"T": t_pred, "M": m_pred}
        if a_pred is not None:
            out["A"] = a_pred
        if v_pred is not None:
            out["V"] = v_pred
        return out