#  zeromodel/pipeline/maestro/online.py
from __future__ import annotations
import torch
from collections import deque

class OnlineMaestro:
    def __init__(self, encoder, residual_head, phase_head, L=8):
        self.buf = deque(maxlen=L)
        self.enc = encoder.eval()
        self.res = residual_head.eval()
        self.ph  = phase_head.eval()

    @torch.no_grad()
    def step(self, frame_hwC):  # np.float32 [H,W,C] in [0,1]
        import numpy as np
        self.buf.append(frame_hwC.astype(np.float32))
        X = np.stack(list(self.buf), axis=0)                  # (T,H,W,C)
        X = torch.from_numpy(X).unsqueeze(0).permute(0,1,4,2,3).contiguous() # (B=1,T,C,H,W) or adapt to tokenizer
        # → run tokenizer inside encode() if you prefer
        z = self.enc.encode(X)          # your wrapper to tokenize+encode
        res_map = self.res(z)           # (Hp,Wp) or (H,W) upsampled
        phase   = self.ph(z)            # logits
        return res_map, phase
