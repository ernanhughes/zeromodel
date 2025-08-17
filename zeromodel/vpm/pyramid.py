# zeromodel/vpm/pyramid.py

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import png

from zeromodel.vpm.image import (
    META_MIN_COLS,
)  # same constant used by _check_header_width

from .image import (
    VPMImageReader,
    VPMImageWriter,
    _check_header_width,
    _round_u16,
    _u16_clip,
)
from .metadata import AggId, VPMMetadata


@dataclass
class VPMPyramidBuilder:
    """
    Build parent VPM tiles from a child tile using aggregation on the R channel
    and percentiles for G. The B channel stores an argmax-in-block hint when
    agg_id == MAX.

    If VPMImageWriter supports an optional `aux_channel` parameter, this class
    will use it. Otherwise, it will write the PNG directly (fallback path).
    """

    K: int = 8  # documents per parent column
    agg_id: int = int(AggId.MAX)  # parent aggregation
    compression: int = 6

    def build_parent(
        self,
        child: VPMImageReader,
        out_path: str,
        *,
        level: Optional[int] = None,
        metadata: Optional[VPMMetadata] = None,
        metric_names: Optional[List[str]] = None,
        doc_id_prefix: str = "d",
        use_writer_aux: bool = True,
    ) -> Tuple[int, int]:
        assert self.K >= 1, "Aggregation factor K must be >= 1"
        M = child.M
        D_phys = child.D

        # Prefer logical doc_count from child's VMETA if present
        D_eff = D_phys
        try:
            mb = child.read_metadata_bytes()
            if mb:
                md_child = VPMMetadata.from_bytes(mb)
                if md_child.doc_count:
                    D_eff = int(md_child.doc_count)
        except Exception:
            pass  # fall back to physical width

        _check_header_width(D_phys)  # validates the child file we read

        # Parent logical width is computed from logical child width
        P_logical = (D_eff + self.K - 1) // self.K

        # --- aggregate using ONLY the logical span ---
        R_child = child.image[child.h_meta :, :, 0].astype(np.uint16)  # (M, D_phys)

        R_parent = np.zeros((M, P_logical), dtype=np.uint16)
        B_parent = np.zeros((M, P_logical), dtype=np.uint16)

        for p in range(P_logical):
            lo = p * self.K
            hi = min(
                D_eff, lo + self.K
            )  # do not include padded zeros beyond logical width
            blk = R_child[:, lo:hi]

            if self.agg_id == int(AggId.MAX):
                vmax = blk.max(axis=1)
                R_parent[:, p] = vmax
                argm = blk.argmax(axis=1)
                if hi - lo > 1:
                    B_parent[:, p] = _round_u16((argm / (hi - lo - 1)) * 65535.0)
                else:
                    B_parent[:, p] = 0
            elif self.agg_id == int(AggId.MEAN):
                vmean = np.round(blk.mean(axis=1))
                R_parent[:, p] = _u16_clip(vmean)
                B_parent[:, p] = 0
            else:
                raise ValueError(f"Unsupported agg_id: {self.agg_id}")

        # G channel = percentiles over the logical width
        if P_logical == 1:
            G_parent = np.full((M, P_logical), 32767, dtype=np.uint16)
        else:
            ranks = np.argsort(np.argsort(R_parent, axis=1), axis=1)
            G_parent = _round_u16((ranks / (P_logical - 1)) * 65535.0)

        # ---- pad to satisfy header width, but keep returning logical (M, P_logical) ----
        out_P = max(P_logical, META_MIN_COLS)
        if out_P != P_logical:
            pad = out_P - P_logical
            R_parent = np.pad(R_parent, ((0, 0), (0, pad)), mode="constant")
            G_parent = np.pad(G_parent, ((0, 0), (0, pad)), mode="constant")
            B_parent = np.pad(B_parent, ((0, 0), (0, pad)), mode="constant")
        P_phys = out_P

        # ---------- metadata ----------
        if metadata is None:
            tile_payload = R_parent.tobytes() + G_parent.tobytes() + B_parent.tobytes()
            tile_id = VPMMetadata.make_tile_id(tile_payload)
            metric_names_eff = metric_names or [f"m{m}" for m in range(M)]
            metadata = VPMMetadata.for_tile(
                level=(child.level - 1 if level is None else level),
                metric_count=M,
                doc_count=P_logical,  # <<< logical count recorded in VMETA
                doc_block_size=child.doc_block_size * self.K,
                agg_id=self.agg_id,
                metric_weights={},
                metric_names=metric_names_eff,
                task_hash=0,
                tile_id=tile_id,
                parent_id=(getattr(child, "tile_id", b"\x00" * 16) or b"\x00" * 16),
            )
        else:
            # ensure the provided metadata has logical doc_count (not padded)
            metadata.doc_count = P_logical

        meta_bytes = metadata.to_bytes()

        # ---------- write PNG ----------
        if use_writer_aux and hasattr(VPMImageWriter, "write_with_channels"):
            writer = VPMImageWriter(
                score_matrix=(R_parent / 65535.0),
                store_minmax=False,
                compression=self.compression,
                level=metadata.level,
                doc_block_size=metadata.doc_block_size,
                agg_id=metadata.agg_id,
                metric_names=metric_names or [f"m{m}" for m in range(M)],
                # we need doc_ids length == physical width actually written
                doc_ids=[f"{doc_id_prefix}{p}" for p in range(P_phys)],
                metadata_bytes=meta_bytes,
            )
            writer.write_with_channels(out_path, R_parent, G_parent, B_parent)
        else:
            self._write_png_direct(
                out_path=out_path,
                level=metadata.level,
                doc_block_size=metadata.doc_block_size,
                agg_id=metadata.agg_id,
                R_parent=R_parent,
                G_parent=G_parent,
                B_parent=B_parent,
                compression=self.compression,
            )

        return M, P_logical

    def build_chain(
        self,
        start_reader: VPMImageReader,
        out_paths: List[str],
        *,
        level_start: Optional[int] = None,
        metric_names: Optional[List[str]] = None,
        doc_id_prefix: str = "d",
    ) -> List[Tuple[int, int]]:
        shapes: List[Tuple[int, int]] = []
        reader = start_reader
        cur_level = reader.level if level_start is None else level_start

        for i, out_path in enumerate(out_paths):
            # prefer child's logical doc_count if present
            child_D_eff = reader.D
            try:
                mb = reader.read_metadata_bytes()
                if mb:
                    md_child = VPMMetadata.from_bytes(mb)
                    if md_child.doc_count:
                        child_D_eff = int(md_child.doc_count)
            except Exception:
                pass

            md = VPMMetadata.for_tile(
                level=cur_level - 1,
                metric_count=reader.M,
                doc_count=(child_D_eff + self.K - 1) // self.K,  # logical
                doc_block_size=reader.doc_block_size * self.K,
                agg_id=self.agg_id,
                metric_weights={},
                metric_names=metric_names or [f"m{m}" for m in range(reader.M)],
                task_hash=0,
                tile_id=VPMMetadata.make_tile_id(f"pyr-{i}".encode()),
                parent_id=getattr(reader, "tile_id", b"\x00" * 16) or b"\x00" * 16,
            )

            shape = self.build_parent(
                reader,
                out_path,
                level=md.level,
                metadata=md,
                metric_names=metric_names,
                doc_id_prefix=doc_id_prefix,
            )
            shapes.append(shape)

            # advance
            reader = VPMImageReader(out_path)
            cur_level = reader.level

        return shapes

    @staticmethod
    def _write_png_direct(
        *,
        out_path: str,
        level: int,
        doc_block_size: int,
        agg_id: int,
        R_parent: np.ndarray,
        G_parent: np.ndarray,
        B_parent: np.ndarray,
        compression: int,
    ) -> None:
        M, P = R_parent.shape

        # ensure header can fit
        from zeromodel.vpm.image import META_MIN_COLS

        out_P = max(P, META_MIN_COLS)
        if out_P != P:
            pad = out_P - P
            R_parent = np.pad(R_parent, ((0, 0), (0, pad)), mode="constant")
            G_parent = np.pad(G_parent, ((0, 0), (0, pad)), mode="constant")
            B_parent = np.pad(B_parent, ((0, 0), (0, pad)), mode="constant")
            P = out_P

        DEFAULT_H_META_BASE = 2
        meta = np.zeros((DEFAULT_H_META_BASE, P, 3), dtype=np.uint16)

        # row 0: magic + core
        magic = [ord("V"), ord("P"), ord("M"), ord("1")]
        for i, v in enumerate(magic):
            meta[0, i, 0] = v
        meta[0, 4, 0] = 1  # version
        meta[0, 5, 0] = np.uint16(M)  # M
        meta[0, 6, 0] = np.uint16((P >> 16) & 0xFFFF)
        meta[0, 7, 0] = np.uint16(P & 0xFFFF)
        meta[0, 8, 0] = np.uint16(DEFAULT_H_META_BASE)
        meta[0, 9, 0] = np.uint16(level)
        meta[0, 10, 0] = np.uint16(min(doc_block_size, 0xFFFF))
        meta[0, 11, 0] = np.uint16(agg_id)

        # row 1: flags (no min/max here)
        meta[1, 0, 0] = 0

        full = np.vstack([meta, np.stack([R_parent, G_parent, B_parent], axis=-1)])
        rows = full.reshape(full.shape[0], -1)

        with open(out_path, "wb") as f:
            png.Writer(
                width=P,
                height=full.shape[0],
                bitdepth=16,
                greyscale=False,
                compression=compression,
                planes=3,
            ).write(f, rows.tolist())
