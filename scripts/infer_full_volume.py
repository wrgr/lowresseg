"""
Run affinity inference + agglomeration on the full 1µm EM volume.

Writes affinities and pred_seg arrays into the existing zarr store.

Usage:
    python scripts/infer_full_volume.py --checkpoint runs/microns_256/checkpoint_final.pt
    python scripts/infer_full_volume.py --checkpoint runs/microns_256/checkpoint_final.pt \\
        --zarr data/minnie65_1um.zarr --threshold 0.5 --patch 96 --overlap 16
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from zarr.codecs import BloscCodec

from lowresseg.models.unet import AffinityUNet
from lowresseg.postprocess.agglomerate import affinities_to_segmentation_fast
from lowresseg.postprocess.tiling import tile_predict

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def load_model(checkpoint_path: str, device: torch.device) -> AffinityUNet:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("model_cfg", {})
    model = AffinityUNet(
        channels=cfg.get("channels", (16, 32, 64, 128)),
        strides=cfg.get("strides", (2, 2, 2)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def normalize(em: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(em, 1), np.percentile(em, 99)
    return ((em.astype(np.float32) - lo) / (hi - lo + 1e-6)).clip(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/microns_256/checkpoint_final.pt")
    parser.add_argument("--zarr", default="data/minnie65_1um.zarr")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patch", type=int, default=96)
    parser.add_argument("--overlap", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading model from %s", args.checkpoint)
    model = load_model(args.checkpoint, device)

    store = zarr.open_group(args.zarr, mode="a")
    em_arr = store["em"]
    em = np.array(em_arr)
    res_nm = list(em_arr.attrs.get("resolution_nm", [1024, 1024, 1280]))
    log.info("EM shape: %s", em.shape)

    # Run inference in Z-slabs to stay within RAM budget.
    # Each slab is patch_z thick with overlap on both sides.
    p = args.patch
    ov = args.overlap
    stride_z = p - 2 * ov
    sx, sy, sz = em.shape

    blosc = BloscCodec(cname="zstd", clevel=3)
    if "affinities" in store:
        del store["affinities"]
    aff_arr = store.create_array(
        "affinities", shape=(3, sx, sy, sz), chunks=(3, 64, 64, 64),
        dtype=np.float32, compressors=blosc,
    )
    aff_arr.attrs.update({"resolution_nm": res_nm, "axes": ["c", "x", "y", "z"],
                          "offsets": [[1,0,0],[0,1,0],[0,0,1]]})

    log.info("Running inference in Z-slabs (patch=%d, overlap=%d, stride_z=%d)...",
             p, ov, stride_z)
    t0 = time.time()
    z_starts = list(range(0, max(sz - p, 0) + 1, stride_z))
    if not z_starts or z_starts[-1] + p < sz:
        z_starts.append(max(0, sz - p))

    for slab_idx, kz in enumerate(z_starts):
        kze = min(kz + p, sz)
        em_slab = em[:, :, kz:kze].astype(np.float32)

        # normalize per slab
        lo, hi = np.percentile(em_slab, 1), np.percentile(em_slab, 99)
        em_slab = ((em_slab - lo) / (hi - lo + 1e-6)).clip(0, 1)

        affs_slab = tile_predict(
            em_slab, model,
            patch_size=(p, p, min(p, kze - kz)),
            overlap=(ov, ov, min(ov, (kze - kz) // 4)),
            device=device,
            normalize=False,
        )  # (3, sx, sy, slab_z)

        # Crop to non-overlapping interior (except first/last slab)
        z_lo = 0 if kz == 0 else ov
        z_hi = affs_slab.shape[3] if (kz + p >= sz) else affs_slab.shape[3] - ov
        dest_z0 = kz + z_lo
        dest_z1 = kz + z_hi
        aff_arr[:, :, :, dest_z0:dest_z1] = affs_slab[:, :, :, z_lo:z_hi]

        log.info("  slab %d/%d  z=%d-%d  written z=%d-%d",
                 slab_idx + 1, len(z_starts), kz, kze, dest_z0, dest_z1)

    log.info("Inference done in %.1f min", (time.time() - t0) / 60)

    # Agglomerate: threshold mean affinity → binary mask → connected components.
    # Full MST is infeasible at this scale (~1B voxels, ~3B edges).
    # CC on thresholded affinities is RAM-efficient and fast enough for a first pass.
    log.info("Agglomerating via threshold+CC (threshold=%.2f)...", args.threshold)
    t0 = time.time()
    from scipy import ndimage as ndi

    # Load one channel at a time to compute mean affinity (stays in ~4 GB per channel)
    log.info("  computing mean affinity across channels...")
    mean_aff = np.zeros((sx, sy, sz), dtype=np.float32)
    for c in range(3):
        mean_aff += np.array(aff_arr[c])
    mean_aff /= 3.0

    log.info("  mean aff=%.3f, thresholding...", mean_aff.mean())
    binary = mean_aff >= args.threshold
    del mean_aff

    log.info("  running connected components...")
    seg, n_segs = ndi.label(binary)
    seg = seg.astype(np.uint64)
    del binary
    log.info("Agglomeration done in %.1f min — %d segments", (time.time() - t0) / 60, n_segs)
    log.info("Predicted seg: shape=%s  unique IDs=%d", seg.shape, n_segs)

    if "pred_seg" in store:
        del store["pred_seg"]
    seg_arr = store.create_array(
        "pred_seg", shape=seg.shape, chunks=(64, 64, 64),
        dtype=np.uint64, compressors=blosc,
    )
    seg_arr[:] = seg
    seg_arr.attrs.update({"resolution_nm": res_nm, "axes": ["x", "y", "z"],
                          "threshold": args.threshold})
    log.info("Saved pred_seg to zarr")
    log.info("Done. Run: python scripts/view_neuroglancer.py --tunnel --seg --pred-seg")


if __name__ == "__main__":
    main()
