"""Convert segmentation labels to short-range affinity targets."""

from __future__ import annotations

import numpy as np


_OFFSETS_SHORT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int32)

_OFFSETS_LONG = np.array(
    [
        [2, 0, 0], [0, 2, 0], [0, 0, 2],
        [4, 0, 0], [0, 4, 0], [0, 0, 4],
        [8, 0, 0], [0, 8, 0], [0, 0, 8],
    ],
    dtype=np.int32,
)


def seg_to_affinities(
    seg: np.ndarray,
    offsets: np.ndarray | None = None,
) -> np.ndarray:
    """
    Args:
        seg: (X, Y, Z) integer segmentation (0 = background/ignore).
        offsets: (N, 3) positive int offsets. Defaults to short-range ±1.

    Returns:
        affs: (N, X, Y, Z) float32 in [0, 1].
              1 = voxel and its +offset neighbour are in the same segment.
              0 = boundary or background.
    """
    if offsets is None:
        offsets = _OFFSETS_SHORT

    n = len(offsets)
    affs = np.zeros((n,) + seg.shape, dtype=np.float32)

    for i, (dx, dy, dz) in enumerate(offsets):
        sx = seg.shape[0] - dx
        sy = seg.shape[1] - dy
        sz = seg.shape[2] - dz
        # src: voxels at (x, y, z); nbr: voxels at (x+dx, y+dy, z+dz)
        src_sl = (slice(0, sx), slice(0, sy), slice(0, sz))
        nbr_sl = (slice(dx, None), slice(dy, None), slice(dz, None))

        s = seg[src_sl]
        nb = seg[nbr_sl]
        valid = (s > 0) & (nb > 0)
        affs[i][src_sl] = (s == nb).astype(np.float32) * valid

    return affs
