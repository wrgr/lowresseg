"""
Download full MICrONS minnie65 EM + seg at 1µm to zarr.

EM:  ~1 GB uint8   (1664 x 1408 x 409)
Seg: ~5 GB uint64  (1504 x 1024 x 407)

Uses tensorstore for reliable sharded neuroglancer precomputed access
(anonymous GCS; no credentials needed for public buckets).

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
import tensorstore as ts

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

EM_SPEC = {
    "driver": "neuroglancer_precomputed",
    "kvstore": {"driver": "gcs", "bucket": "iarpa_microns", "path": "minnie/minnie65/em"},
    "scale_index": 7,
}
SEG_SPEC = {
    "driver": "neuroglancer_precomputed",
    "kvstore": {"driver": "gcs", "bucket": "iarpa_microns", "path": "minnie/minnie65/seg"},
    "scale_index": 7,
}


def download_volume(
    dataset: ts.TensorStore,
    out_arr: zarr.Array,
    tile_xy: int = 512,
    dtype=None,
) -> None:
    """Download tensorstore dataset into zarr, tiling in XY."""
    ox, oy, oz = [int(v) for v in dataset.domain.origin[:3]]
    shape = out_arr.shape  # (X, Y, Z)
    sx, sy, sz = shape

    x_steps = list(range(0, sx, tile_xy))
    y_steps = list(range(0, sy, tile_xy))
    total = len(x_steps) * len(y_steps)
    done = 0

    for xi in x_steps:
        xe = min(xi + tile_xy, sx)
        for yi in y_steps:
            ye = min(yi + tile_xy, sy)

            slab = dataset[ox+xi:ox+xe, oy+yi:oy+ye, oz:oz+sz].read().result()
            if slab.ndim == 4:
                slab = slab[..., 0]
            slab = np.asarray(slab)
            if dtype is not None:
                slab = slab.astype(dtype)

            out_arr[xi:xe, yi:ye, :] = slab
            done += 1
            if done % max(1, total // 20) == 0 or done == total:
                log.info("  %d/%d slabs  (%.1f%%)", done, total, 100 * done / total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/minnie65_1um.zarr")
    parser.add_argument("--em-only", action="store_true")
    parser.add_argument("--tile-size", type=int, default=512)
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)
    store = zarr.open_group(args.output, mode="w")
    blosc = BloscCodec(cname="zstd", clevel=3)

    # --- EM ---
    log.info("Opening EM volume...")
    em_ds = ts.open(EM_SPEC).result()
    em_shape = list(em_ds.shape[:3])  # drop channel dim
    em_res = [1024, 1024, 1280]
    log.info("EM shape: %s  resolution: %s nm", em_shape, em_res)

    em_arr = store.create_array(
        "em", shape=em_shape, chunks=(64, 64, 64),
        dtype=np.uint8, compressors=blosc,
    )
    em_arr.attrs.update({"resolution_nm": em_res, "axes": ["x", "y", "z"]})

    t0 = time.time()
    ts_val = args.tile_size
    log.info("Downloading EM (%d XY slabs of %d)...",
             int(np.ceil(em_shape[0] / ts_val)) * int(np.ceil(em_shape[1] / ts_val)), ts_val)
    download_volume(em_ds, em_arr, tile_xy=ts_val, dtype=np.uint8)
    log.info("EM done in %.1f min", (time.time() - t0) / 60)

    if args.em_only:
        log.info("Saved EM to %s", args.output)
        return

    # --- Seg ---
    log.info("Opening seg volume...")
    seg_ds = ts.open(SEG_SPEC).result()
    seg_shape = list(seg_ds.shape[:3])
    log.info("Seg shape: %s", seg_shape)

    seg_arr = store.create_array(
        "seg", shape=seg_shape, chunks=(64, 64, 64),
        dtype=np.uint64, compressors=blosc,
    )
    seg_arr.attrs.update({"resolution_nm": [1024, 1024, 1280], "axes": ["x", "y", "z"]})

    t0 = time.time()
    log.info("Downloading seg (~%d GB uint64)...", int(np.prod(seg_shape) * 8 / 1e9))
    download_volume(seg_ds, seg_arr, tile_xy=ts_val, dtype=np.uint64)
    log.info("Seg done in %.1f min", (time.time() - t0) / 60)

    store.attrs.update({"mip": 7, "voxel_size_nm": em_res})
    log.info("Full volume saved to %s", args.output)


if __name__ == "__main__":
    main()
