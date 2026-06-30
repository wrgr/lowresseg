"""
Download a small EM + seg chunk from MICrONS to zarr for offline dev/testing.

Usage:
    python scripts/download_microns_chunk.py --size 256 --output data/microns_sample.zarr
"""

import argparse
import logging
import numpy as np
import zarr
import cloudvolume as cv

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# MICrONS minnie65 public CloudVolume paths
EM_PATH = "precomputed://gs://iarpa_microns/minnie/minnie65/em"
SEG_PATH = "precomputed://gs://iarpa_microns/minnie/minnie65/seg/minnie65_8"

# Sensible default location in the volume (mip0 coords, ~centre of minnie65)
DEFAULT_START_MIP0 = [193536, 130048, 800]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256, help="Cube size in voxels at 1µm")
    parser.add_argument("--output", default="data/microns_sample.zarr")
    parser.add_argument(
        "--start",
        nargs=3,
        type=int,
        default=DEFAULT_START_MIP0,
        metavar=("X", "Y", "Z"),
        help="Bounding box start in mip0 voxels",
    )
    args = parser.parse_args()

    em_mip = 7   # ~1024nm
    seg_mip = 4  # adjust if needed

    em_scale = 2 ** em_mip
    seg_scale = 2 ** seg_mip

    em_start = [s // em_scale for s in args.start]
    em_end = [s + args.size for s in em_start]
    seg_start = [s // seg_scale for s in args.start]
    seg_end = [s + args.size for s in seg_start]

    log.info("Opening EM volume at mip %d...", em_mip)
    em_vol = cv.CloudVolume(EM_PATH, mip=em_mip, fill_missing=True, progress=True)
    log.info("EM resolution: %s nm", em_vol.resolution.tolist())

    log.info("Fetching EM chunk %s -> %s", em_start, em_end)
    em = np.array(em_vol[em_start[0]:em_end[0], em_start[1]:em_end[1], em_start[2]:em_end[2]])
    if em.ndim == 4:
        em = em[..., 0]
    log.info("EM shape: %s dtype: %s", em.shape, em.dtype)

    log.info("Opening seg volume at mip %d...", seg_mip)
    seg_vol = cv.CloudVolume(SEG_PATH, mip=seg_mip, fill_missing=True, progress=True)
    log.info("Seg resolution: %s nm", seg_vol.resolution.tolist())

    log.info("Fetching seg chunk %s -> %s", seg_start, seg_end)
    seg = np.array(seg_vol[seg_start[0]:seg_end[0], seg_start[1]:seg_end[1], seg_start[2]:seg_end[2]])
    if seg.ndim == 4:
        seg = seg[..., 0]

    # Resize seg to match EM if resolutions differ
    if seg.shape != em.shape:
        from skimage.transform import resize
        log.info("Resizing seg %s -> %s", seg.shape, em.shape)
        seg = resize(seg.astype(float), em.shape, order=0, anti_aliasing=False).astype(np.uint64)

    log.info("Saving to %s", args.output)
    store = zarr.open(args.output, mode="w")
    store.create_dataset("em", data=em, chunks=(64, 64, 64), compressor=zarr.Blosc(cname="zstd", clevel=3))
    store.create_dataset("seg", data=seg, chunks=(64, 64, 64), compressor=zarr.Blosc(cname="zstd", clevel=3), dtype=np.uint64)
    store.attrs["em_mip"] = em_mip
    store.attrs["seg_mip"] = seg_mip
    store.attrs["voxel_size_nm"] = em_vol.resolution.tolist()
    log.info("Done. EM: %s, Seg: %s", em.shape, seg.shape)


if __name__ == "__main__":
    main()
