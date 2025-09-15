#  zeromodel/vpm/view.py
from __future__ import annotations
from .image import VPMImageReader


class VPMView:
    def __init__(self, reader: VPMImageReader):
        self.r = reader

    def top_left_tile(self, metric_idx: int = 0, size: int = 8):
        return self.r.get_virtual_view(
            metric_idx=metric_idx, x=0, y=0, width=size, height=size
        )

    def composite_top_left(self, weights: dict, size: int = 8):
        return self.r.get_virtual_view(
            weights=weights, x=0, y=0, width=size, height=size
        )

    def order(self, metric_idx=None, weights=None, top_k=None, descending=True):
        return self.r.virtual_order(
            metric_idx=metric_idx, weights=weights, top_k=top_k, descending=descending
        )
