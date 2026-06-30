"""
Download a small EM + seg chunk from MICrONS to zarr for offline dev/testing.

Usage:
    python scripts/download_microns_chunk.py --size 256 --output data/microns_sample.zarr
"""

import argparse
import logging
import numpy as np
import zarr
from zarr.codecs import BloscCodec
import cloudvolume as cv

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# MICrONS minnie65 - use https:// (no GCS credentials needed for public data)
EM_PATH  = "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/em"
SEG_PATH = "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg"

# Both EM and seg have a matching ~1µm mip key "1024x1024x1280"
# In CloudVolume mip index terms: EM mip7 = 1024nm, seg mip7 = 1024nm
EM_MIP  = 7
SEG_MIP = 7   # seg has the same resolution ladder

# Sensible default start (in 1µm mip-level voxels, within the volume bounds
# [108,108,463] -> [1772,1516,872])
DEFAULT_START = [300, 300, 10]   # voxels at 1µm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128, help="Cube side in voxels at 1µm")
    parser.add_argument("--output", default="data/microns_sample.zarr")
    parser.add_argument(
        "--start",
        nargs=3,
        type=int,
        default=DEFAULT_START,
        metavar=("X", "Y", "Z"),
        help="Bounding box start in 1µm voxels (mip7 coords)",
    )
    args = parser.parse_args()

    start = args.start
    end   = [s + args.size for s in start]

    log.info("Opening EM volume...")
    em_vol = cv.CloudVolume(EM_PATH, mip=EM_MIP, fill_missing=True, progress=True)
    log.info("EM resolution: %s nm | bounds: %s -> %s",
             em_vol.resolution.tolist(), em_vol.bounds.minpt.tolist(), em_vol.bounds.maxpt.tolist())

    log.info("Fetching EM chunk %s -> %s", start, end)
    em = np.array(em_vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]])
    if em.ndim == 4:
        em = em[..., 0]
    log.info("EM shape: %s  dtype: %s  range: [%d, %d]", em.shape, em.dtype, em.min(), em.max())

    log.info("Opening seg volume...")
    seg_vol = cv.CloudVolume(SEG_PATH, mip=SEG_MIP, fill_missing=True, progress=True)
    log.info("Seg resolution: %s nm", seg_vol.resolution.tolist())

    log.info("Fetching seg chunk %s -> %s", start, end)
    seg = np.array(seg_vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]])
    if seg.ndim == 4:
        seg = seg[..., 0]
    log.info("Seg shape: %s  unique IDs: %d", seg.shape, len(np.unique(seg)))

    if seg.shape != em.shape:
        from skimage.transform import resize
        log.info("Resizing seg %s -> %s", seg.shape, em.shape)
        seg = resize(seg.astype(float), em.shape, order=0, anti_aliasing=False).astype(np.uint64)

    import os; os.makedirs("data", exist_ok=True)
    log.info("Saving to %s", args.output)

    store = zarr.open_group(args.output, mode="w")
    store.create_array("em",  data=em,  chunks=(64,64,64), compressors=BloscCodec(cname="zstd", clevel=3))
    store.create_array("seg", data=seg.astype(np.uint64), chunks=(64,64,64), compressors=BloscCodec(cname="zstd", clevel=3))
    store.attrs.update({
        "em_path": EM_PATH, "seg_path": SEG_PATH,
        "em_mip": EM_MIP,   "seg_mip": SEG_MIP,
        "start_vox": start,  "end_vox": end,
        "voxel_size_nm": em_vol.resolution.tolist(),
    })
    log.info("Done.  EM: %s  Seg: %s  -> %s", em.shape, seg.shape, args.output)


if __name__ == "__main__":
    main()
