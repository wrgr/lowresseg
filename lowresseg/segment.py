"""Agglomerate predicted affinities into a segmentation and save to zarr."""

from __future__ import annotations

import argparse
import logging

import numpy as np
import zarr
from zarr.codecs import BloscCodec

from .postprocess.agglomerate import affinities_to_segmentation

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agglomerate affinities → segmentation")
    parser.add_argument("affinities_zarr", help="Path to zarr with 'affinities' dataset")
    parser.add_argument("--output", default="segmentation.zarr")
    parser.add_argument(
        "--method",
        choices=["waterz", "cc"],
        default="waterz",
        help="Agglomeration method",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Merge threshold (waterz) or foreground threshold (cc)",
    )
    parser.add_argument("--merge-function", default="mean", help="waterz scoring function")
    args = parser.parse_args()

    log.info("Loading affinities from %s", args.affinities_zarr)
    store = zarr.open_group(args.affinities_zarr, mode="r")
    affs = store["affinities"][:]

    log.info("Agglomerating with method=%s threshold=%.3f", args.method, args.threshold)
    seg = affinities_to_segmentation(
        affs,
        threshold=args.threshold,
        method=args.method,
        merge_function=args.merge_function,
    )

    n_segments = int(seg.max())
    log.info("Segmentation shape: %s | %d segments", seg.shape, n_segments)

    out = zarr.open_group(args.output, mode="w")
    arr = out.create_array(
        "seg",
        data=seg,
        chunks=(64, 64, 64),
        compressors=BloscCodec(cname="zstd", clevel=3),
        dtype=np.uint64,
    )
    arr.attrs["voxel_size_um"] = 1.0
    log.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
