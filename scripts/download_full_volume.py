"""
Download full MICrONS minnie65 EM + seg at 1µm to zarr.

EM:  ~1 GB uint8   (1664 x 1408 x 409)
Seg: ~5 GB uint64  (1504 x 1024 x 407)

Usage:
    python scripts/download_full_volume.py --output data/minnie65_1um.zarr
    python scripts/download_full_volume.py --em-only --output data/minnie65_em.zarr
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec
import cloudvolume as cv

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

EM_PATH  = "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/em"
SEG_PATH = "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg"
MIP = 7  # ~1024 nm isotropic


def download_volume(
    vol: cv.CloudVolume,
    out_arr: zarr.Array,
    chunk_size: int = 64,
    dtype=None,
) -> None:
    """
    Download a CloudVolume into a zarr array using native chunk-aligned fetches.
    Requests must align to the volume's internal chunk grid (64^3) to avoid
    CloudVolume's shard decoder receiving partial/misaligned byte ranges.
    """
    bounds = vol.bounds
    ox, oy, oz = bounds.minpt.tolist()
    sx, sy, sz = out_arr.shape

    x_steps = list(range(0, sx, chunk_size))
    y_steps = list(range(0, sy, chunk_size))
    z_steps = list(range(0, sz, chunk_size))
    total = len(x_steps) * len(y_steps) * len(z_steps)
    done = 0

    for xi in x_steps:
        xe = min(xi + chunk_size, sx)
        for yi in y_steps:
            ye = min(yi + chunk_size, sy)
            for zi in z_steps:
                ze = min(zi + chunk_size, sz)

                chunk = np.array(vol[ox+xi:ox+xe, oy+yi:oy+ye, oz+zi:oz+ze])
                if chunk.ndim == 4:
                    chunk = chunk[..., 0]
                if dtype is not None:
                    chunk = chunk.astype(dtype)

                out_arr[xi:xe, yi:ye, zi:ze] = chunk
                done += 1
                if done % max(1, total // 20) == 0 or done == total:
                    log.info("  %d/%d chunks  (%.1f%%)", done, total, 100*done/total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/minnie65_1um.zarr")
    parser.add_argument("--em-only", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)
    store = zarr.open_group(args.output, mode="w")
    blosc = BloscCodec(cname="zstd", clevel=3)

    # --- EM ---
    log.info("Opening EM volume...")
    em_vol = cv.CloudVolume(EM_PATH, mip=MIP, fill_missing=True, progress=False)
    em_res = em_vol.resolution.tolist()
    bounds = em_vol.bounds
    em_shape = (bounds.maxpt - bounds.minpt).tolist()
    log.info("EM shape: %s  resolution: %s nm", em_shape, em_res)

    em_arr = store.create_array(
        "em", shape=em_shape, chunks=(64, 64, 64),
        dtype=np.uint8, compressors=blosc,
    )
    em_arr.attrs.update({"resolution_nm": em_res, "axes": ["x","y","z"], "bounds_minpt": bounds.minpt.tolist()})

    t0 = time.time()
    cs = args.chunk_size
    log.info("Downloading EM (%d chunks of %d)...",
             np.prod([int(np.ceil(s/cs)) for s in em_shape]), cs)
    download_volume(em_vol, em_arr, chunk_size=cs, dtype=np.uint8)
    log.info("EM done in %.1f min", (time.time()-t0)/60)

    if args.em_only:
        log.info("Saved EM to %s", args.output)
        return

    # --- Seg ---
    log.info("Opening seg volume...")
    seg_vol = cv.CloudVolume(SEG_PATH, mip=MIP, fill_missing=True, progress=False)
    seg_bounds = seg_vol.bounds
    seg_shape = (seg_bounds.maxpt - seg_bounds.minpt).tolist()
    log.info("Seg shape: %s", seg_shape)

    seg_arr = store.create_array(
        "seg", shape=seg_shape, chunks=(64, 64, 64),
        dtype=np.uint64, compressors=blosc,
    )
    seg_arr.attrs.update({"resolution_nm": seg_vol.resolution.tolist(), "axes": ["x","y","z"],
                          "bounds_minpt": seg_bounds.minpt.tolist()})

    t0 = time.time()
    log.info("Downloading seg (%d GB uint64)...",
             int(np.prod(seg_shape) * 8 / 1e9))
    download_volume(seg_vol, seg_arr, chunk_size=cs, dtype=np.uint64)
    log.info("Seg done in %.1f min", (time.time()-t0)/60)

    store.attrs.update({"em_path": EM_PATH, "seg_path": SEG_PATH, "mip": MIP,
                        "voxel_size_nm": em_res})
    log.info("Full volume saved to %s", args.output)


if __name__ == "__main__":
    main()
