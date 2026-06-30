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
    tile_xy: int = 512,
    dtype=None,
) -> None:
    """
    Download a CloudVolume into a zarr array, tiling only in XY and fetching
    full Z slabs. CloudVolume handles its own internal shard chunking; we just
    need our request bbox to be large enough that it doesn't undercut the
    volume's native chunk size.
    """
    bounds = vol.bounds
    xs, ys, zs = bounds.minpt.tolist()
    shape = out_arr.shape  # (X, Y, Z) in zarr coords

    x_tiles = list(range(0, shape[0], tile_xy))
    y_tiles = list(range(0, shape[1], tile_xy))
    total = len(x_tiles) * len(y_tiles)
    done = 0

    for xi in x_tiles:
        xe_t = min(xi + tile_xy, shape[0])
        for yi in y_tiles:
            ye_t = min(yi + tile_xy, shape[1])

            cv_x0, cv_x1 = xs + xi, xs + xe_t
            cv_y0, cv_y1 = ys + yi, ys + ye_t
            cv_z0, cv_z1 = zs, zs + shape[2]

            slab = np.array(vol[cv_x0:cv_x1, cv_y0:cv_y1, cv_z0:cv_z1])
            if slab.ndim == 4:
                slab = slab[..., 0]
            if dtype is not None:
                slab = slab.astype(dtype)

            out_arr[xi:xe_t, yi:ye_t, :] = slab
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                log.info("  %d/%d slabs  (%.1f%%)", done, total, 100*done/total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/minnie65_1um.zarr")
    parser.add_argument("--em-only", action="store_true")
    parser.add_argument("--tile-size", type=int, default=256)
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
    log.info("Downloading EM (%d tiles of %d)...",
             np.prod([int(np.ceil(s/args.tile_size)) for s in em_shape]), args.tile_size)
    download_volume(em_vol, em_arr, tile_xy=args.tile_size, dtype=np.uint8)
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
    download_volume(seg_vol, seg_arr, tile_xy=args.tile_size, dtype=np.uint64)
    log.info("Seg done in %.1f min", (time.time()-t0)/60)

    store.attrs.update({"em_path": EM_PATH, "seg_path": SEG_PATH, "mip": MIP,
                        "voxel_size_nm": em_res})
    log.info("Full volume saved to %s", args.output)


if __name__ == "__main__":
    main()
