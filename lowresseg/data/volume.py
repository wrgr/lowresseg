"""CloudVolume wrappers for MICrONS (and compatible) EM + segmentation data."""

from __future__ import annotations

import numpy as np
import cloudvolume as cv
from cloudvolume import CloudVolume


def open_volume(path: str, mip: int, cache_dir: str | None = None) -> CloudVolume:
    kwargs: dict = dict(
        mip=mip,
        bounded=True,
        fill_missing=True,
        progress=False,
    )
    if cache_dir:
        kwargs["cache"] = cache_dir
    return CloudVolume(path, **kwargs)


def fetch_chunk(
    vol: CloudVolume,
    bbox_start: tuple[int, int, int],
    bbox_end: tuple[int, int, int],
) -> np.ndarray:
    """Return (X, Y, Z) uint8/uint64 array from CloudVolume bbox (mip0 coords)."""
    slices = tuple(slice(s, e) for s, e in zip(bbox_start, bbox_end))
    data = vol[slices]
    # CloudVolume returns (X, Y, Z, C); squeeze channel dim
    if data.ndim == 4:
        data = data[..., 0]
    return np.asarray(data)


def voxel_size_nm(vol: CloudVolume) -> tuple[float, float, float]:
    res = vol.resolution  # nanometres
    return tuple(float(r) for r in res)
