"""Convert segmentation labels to short-range affinity targets."""

from __future__ import annotations

import numpy as np


_OFFSETS_SHORT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int32)

# Long-range offsets at 2, 4, 8 voxels along each axis
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
        offsets: (N, 3) int array of neighbor offsets. Defaults to short-range.

    Returns:
        affs: (N, X, Y, Z) float32 in [0, 1]. 1 = same segment, 0 = boundary.
    """
    if offsets is None:
        offsets = _OFFSETS_SHORT

    n = len(offsets)
    affs = np.zeros((n,) + seg.shape, dtype=np.float32)

    for i, (dx, dy, dz) in enumerate(offsets):
        # Slice pairs: current vs shifted neighbour
        src = (
            slice(max(dx, 0), seg.shape[0] + min(dx, 0) if dx < 0 else None),
            slice(max(dy, 0), seg.shape[1] + min(dy, 0) if dy < 0 else None),
            slice(max(dz, 0), seg.shape[2] + min(dz, 0) if dz < 0 else None),
        )
        nbr = (
            slice(-dx if dx > 0 else 0, seg.shape[0] - dx if dx < 0 else None),
            slice(-dy if dy > 0 else 0, seg.shape[1] - dy if dy < 0 else None),
            slice(-dz if dz > 0 else 0, seg.shape[2] - dz if dz < 0 else None),
        )
        same = (seg[src] == seg[nbr]).astype(np.float32)
        # Zero out where either voxel is background (id=0)
        valid = (seg[src] > 0) & (seg[nbr] > 0)
        affs[i][src] = same * valid
    return affs
