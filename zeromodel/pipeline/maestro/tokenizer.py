#  zeromodel/pipeline/maestro/tokenizer.py
from __future__ import annotations
import torch, math, torch.nn as nn

def sinusoid_pos(n, d):
    pos = torch.arange(n).unsqueeze(1)
    i   = torch.arange(d).unsqueeze(0)
    ang = pos / (10000 ** (2*(i//2)/d))
    pe = torch.zeros(n, d)
    pe[:,0::2] = torch.sin(ang[:,0::2])
    pe[:,1::2] = torch.cos(ang[:,1::2])
    return pe

class JointTokenizer(nn.Module):
    def __init__(self, P=16, C=8, Ce=256, H=128, W=128, T=16):
        super().__init__()
        self.P, self.C, self.Ce = P, C, Ce
        self.proj = nn.Linear(P*P*C, Ce)
        self.spos = sinusoid_pos((H//P)*(W//P), Ce)  # spatial
        self.tpos = sinusoid_pos(T, Ce)              # temporal
    def forward(self, x):  # x: (B,T,H,W,C)
        B,T,H,W,C = x.shape
        Hp, Wp = H//self.P, W//self.P
        # patchify
        x = x.view(B,T,Hp,self.P,Wp,self.P,C).permute(0,1,2,4,3,5,6).contiguous()
        x = x.view(B,T,Hp*Wp,self.P*self.P*C)                # (B,T,Np,PPC)
        tok = self.proj(x)                                   # (B,T,Np,Ce)
        # add pos (broadcast)
        spos = self.spos.to(tok.device)[None,None,:,:]       # (1,1,Np,Ce)
        tpos = self.tpos.to(tok.device)[None,:,None,:]       # (1,T,1,Ce)
        tok  = tok + spos + tpos
        return tok  # (B,T,Np,Ce)
