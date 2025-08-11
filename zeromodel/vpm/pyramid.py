import numpy as np
import png
from .image import VPMImageReader, VPMImageWriter, DEFAULT_H_META_BASE, _check_header_width, _round_u16, _u16_clip, AGG_MAX, AGG_MEAN, AGG_RAW

class VPMPyramid:
    @staticmethod
    def build_parent(child: VPMImageReader, out_path: str, K: int = 8, agg_id: int = AGG_MAX, compression: int = 6, level: int | None = None):
        assert K >= 1
        M, D = child.M, child.D; _check_header_width(D)
        P = (D + K - 1) // K

        block_R = child.image[child.h_meta:, :, 0].astype(np.uint16)  # (M,D)
        R_parent = np.zeros((M, P), dtype=np.uint16)
        B_parent = np.zeros((M, P), dtype=np.uint16)

        for p in range(P):
            lo, hi = p*K, min(D, p*K + K)
            blk = block_R[:, lo:hi]
            if agg_id == AGG_MAX:
                vmax = blk.max(axis=1); R_parent[:, p] = vmax
                argm = blk.argmax(axis=1)
                B_parent[:, p] = 0 if hi-lo <= 1 else _round_u16((argm / (hi-lo-1)) * 65535.0)
            elif agg_id == AGG_MEAN:
                R_parent[:, p] = _u16_clip(np.round(blk.mean(axis=1)))
            else:
                raise ValueError("agg_id not supported")

        if P == 1:
            G_parent = np.full((M, P), 32767, dtype=np.uint16)
        else:
            ranks = np.argsort(np.argsort(R_parent, axis=1), axis=1)
            G_parent = _round_u16((ranks / (P - 1)) * 65535.0)

        writer = VPMImageWriter(
            score_matrix=(R_parent / 65535.0),
            store_minmax=False, store_ids=False,
            compression=compression,
            level=child.level - 1 if level is None else level,
            doc_block_size=child.doc_block_size * K,
            agg_id=agg_id,
            metric_ids=[f"m{m}" for m in range(M)],
            doc_ids=[f"d{p}" for p in range(P)],
        )

        # Use writer to assemble meta rows
        norm, _, _ = writer._normalize()  # not used; we just want shapes
        h_meta = DEFAULT_H_META_BASE
        meta = np.zeros((h_meta, P, 3), dtype=np.uint16)

        # row0
        for i, v in enumerate([ord('V'), ord('P'), ord('M'), ord('1')]): meta[0, i, 0] = v
        meta[0, 4, 0]  = 1
        meta[0, 5, 0]  = np.uint16(M)
        meta[0, 6, 0]  = np.uint16((P >> 16) & 0xFFFF)
        meta[0, 7, 0]  = np.uint16(P & 0xFFFF)
        meta[0, 8, 0]  = np.uint16(h_meta)
        meta[0, 9, 0]  = np.uint16(writer.level)
        meta[0, 10, 0] = np.uint16(min(writer.doc_block_size, 0xFFFF))
        meta[0, 11, 0] = np.uint16(writer.agg_id)
        # row1 flags
        meta[1, 0, 0] = 0

        full = np.vstack([meta, np.stack([R_parent, G_parent, B_parent], axis=-1)])
        rows = full.reshape(full.shape[0], -1)
        with open(out_path, "wb") as f:
            png.Writer(width=P, height=full.shape[0], bitdepth=16, greyscale=False, planes=3, compression=compression).write(f, rows.tolist())
