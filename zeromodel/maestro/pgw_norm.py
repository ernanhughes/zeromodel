# zeromodel/maestro/pgw_norm.py
import torch

def pgw_normalize(frame_hwC: torch.Tensor, group_slices, eps=1e-6):
    # frame_hwC: (H,W,C); normalize per patch later, but start global z if needed
    H, W, C = frame_hwC.shape
    out = torch.empty_like(frame_hwC)
    for s, e in group_slices:  # e exclusive
        g = frame_hwC[..., s:e]
        mu = g.mean(dim=(-3,-2,-1), keepdim=True)
        sd = g.std (dim=(-3,-2,-1), keepdim=True)
        out[..., s:e] = (g - mu) / (sd + eps)
    return out
