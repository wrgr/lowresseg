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

    log.info("Running affinity inference (patch=%d, overlap=%d)...", args.patch, args.overlap)
    t0 = time.time()
    affs = tile_predict(
        em, model,
        patch_size=(args.patch, args.patch, args.patch),
        overlap=(args.overlap, args.overlap, args.overlap),
        device=device,
        normalize=False,  # already normalized
    )
    # affs shape: (3, X, Y, Z)
    log.info("Inference done in %.1f min", (time.time() - t0) / 60)
    log.info("Affinities: shape=%s  mean=%.3f", affs.shape, affs.mean())

    blosc = BloscCodec(cname="zstd", clevel=3)
    if "affinities" in store:
        del store["affinities"]
    aff_arr = store.create_array(
        "affinities", shape=affs.shape, chunks=(3, 64, 64, 64),
        dtype=np.float32, compressors=blosc,
    )
    aff_arr[:] = affs
    aff_arr.attrs.update({"resolution_nm": res_nm, "axes": ["c", "x", "y", "z"],
                          "offsets": [[1,0,0],[0,1,0],[0,0,1]]})
    log.info("Saved affinities to zarr")

    log.info("Agglomerating (threshold=%.2f)...", args.threshold)
    t0 = time.time()
    seg = affinities_to_segmentation_fast(affs, threshold=args.threshold)
    log.info("Agglomeration done in %.1f min", (time.time() - t0) / 60)
    n_segs = len(np.unique(seg))
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
