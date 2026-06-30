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
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from zarr.codecs import BloscCodec

from lowresseg.models.unet import AffinityUNet
from lowresseg.postprocess.agglomerate import affinities_to_segmentation_fast
from lowresseg.postprocess.tiling import tile_predict, _hanning_window

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
    parser.add_argument("--min-size", type=int, default=100,
                        help="Remove segments smaller than this many voxels (default 100)")
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

    # Tile in all 3 dims, writing each XYZ chunk directly to zarr.
    # Peak RAM per chunk: em_chunk (p^3 float32 ~3.5MB) + aff_chunk (3*p^3 ~10MB) — trivial.
    stride = p - 2 * ov

    def make_starts(dim_size):
        starts = list(range(0, max(dim_size - p, 0) + 1, stride))
        if not starts or starts[-1] + p < dim_size:
            starts.append(max(0, dim_size - p))
        return starts

    starts_x = make_starts(sx)
    starts_y = make_starts(sy)
    starts_z = make_starts(sz)
    total = len(starts_x) * len(starts_y) * len(starts_z)
    log.info("Running inference: %dx%dx%d=%d patches of size %d, overlap %d...",
             len(starts_x), len(starts_y), len(starts_z), total, p, ov)

    # Accumulate in a small rolling buffer per Z-slab to handle overlap blending.
    # We flush one Z-stride worth of Z-planes once we've seen all patches covering them.
    w = _hanning_window((p, p, p))

    # Use a full-XY, one-Z-slab accumulator — but only p voxels deep at a time
    # Resume checkpoint: tracks how many Z-planes have been safely flushed to zarr.
    ckpt_path = Path(args.zarr) / "infer_checkpoint.json"
    resume_z0 = 0
    if ckpt_path.exists():
        try:
            resume_z0 = json.loads(ckpt_path.read_text()).get("flushed_z", 0)
            log.info("Resuming from flushed_z=%d (skipping patches entirely within z<%d)",
                     resume_z0, resume_z0)
        except Exception:
            pass

    aff_buf  = np.zeros((3, sx, sy, p), dtype=np.float32)
    wgt_buf  = np.zeros((sx, sy, p),    dtype=np.float32)
    buf_z0   = resume_z0

    t0 = time.time()
    done = 0

    # Global percentile for normalization (cheap: just sample the EM)
    lo, hi = np.percentile(em[::4, ::4, ::4], [1, 99])

    def save_checkpoint() -> None:
        ckpt_path.write_text(json.dumps({"flushed_z": buf_z0}))

    for iz, kz in enumerate(starts_z):
        kze = min(kz + p, sz)
        pz  = kze - kz

        # Skip Z-slabs that are entirely within already-flushed planes.
        if kze <= resume_z0:
            done += len(starts_x) * len(starts_y)
            continue

        # Flush and roll the buffer when we move to a new Z-stride
        flush_up_to = kz + (0 if iz == 0 else ov)
        if flush_up_to > buf_z0:
            n_flush = flush_up_to - buf_z0
            safe = np.maximum(wgt_buf[:, :, :n_flush], 1e-6)
            aff_arr[:, :, :, buf_z0:flush_up_to] = (
                aff_buf[:, :, :, :n_flush] / safe[None]).clip(0, 1)
            keep = p - n_flush
            aff_buf[:, :, :, :keep] = aff_buf[:, :, :, n_flush:n_flush + keep]
            aff_buf[:, :, :, keep:] = 0
            wgt_buf[:, :, :keep]    = wgt_buf[:, :, n_flush:n_flush + keep]
            wgt_buf[:, :, keep:]    = 0
            buf_z0 = flush_up_to
            save_checkpoint()

        for ix, kx in enumerate(starts_x):
            kxe = min(kx + p, sx)
            px_ = kxe - kx

            for iy, ky in enumerate(starts_y):
                kye = min(ky + p, sy)
                py_ = kye - ky

                tile = ((em[kx:kxe, ky:kye, kz:kze].astype(np.float32) - lo)
                        / (hi - lo + 1e-6)).clip(0, 1)
                pad = [(0, p - px_), (0, p - py_), (0, p - pz)]
                tile_p = np.pad(tile, pad, mode="reflect")

                t_in = torch.from_numpy(tile_p[None, None].astype(np.float32))
                with torch.no_grad():
                    pred = model(t_in.to(device)).cpu().numpy()[0]  # (3, p, p, p)

                bz = kz - buf_z0
                # When resuming, kz may be below buf_z0 — skip already-flushed planes.
                z_skip = max(0, -bz)
                bz = max(0, bz)
                pz_w = pz - z_skip
                if pz_w <= 0:
                    done += 1
                    continue
                wi = w[:px_, :py_, z_skip:z_skip + pz_w]
                aff_buf[:, kx:kxe, ky:kye, bz:bz + pz_w] += pred[:, :px_, :py_, z_skip:z_skip + pz_w] * wi
                wgt_buf[   kx:kxe, ky:kye, bz:bz + pz_w] += wi

                done += 1
                if done % max(1, total // 40) == 0 or done == total:
                    log.info("  %d/%d patches (%.0f%%)", done, total, 100 * done / total)

    # Final flush
    safe = np.maximum(wgt_buf[:, :, :sz - buf_z0], 1e-6)
    aff_arr[:, :, :, buf_z0:sz] = (
        aff_buf[:, :, :, :sz - buf_z0] / safe[None]).clip(0, 1)
    ckpt_path.unlink(missing_ok=True)  # clean up on successful completion

    log.info("Inference done in %.1f min", (time.time() - t0) / 60)

    # Free EM and inference buffers before agglomeration to reclaim ~1.5 GB RAM.
    del em, aff_buf, wgt_buf

    # Agglomerate: threshold mean affinity → binary mask → connected components.
    log.info("Agglomerating via threshold+CC (threshold=%.2f)...", args.threshold)
    t0 = time.time()
    from scipy import ndimage as ndi

    # Load affinities in Z-slabs to stay within RAM budget.
    # Peak per slab: one float32 slab (~3.8 GB / (sz/slab_z)) — well under 2 GB per slab.
    log.info("  thresholding affinities in Z-slabs...")
    slab_z = 64  # ~64 * 1664 * 1408 * 4 * 3 ≈ 1.6 GB per slab across 3 channels
    binary = np.zeros((sx, sy, sz), dtype=bool)
    vote = np.zeros((sx, sy, sz), dtype=np.uint8)  # counts channels >= threshold

    for z0 in range(0, sz, slab_z):
        z1 = min(z0 + slab_z, sz)
        for c in range(3):
            slab = np.array(aff_arr[c, :, :, z0:z1], dtype=np.float32)
            vote[:, :, z0:z1] += (slab >= args.threshold).astype(np.uint8)
            del slab
        log.info("  slabs z=%d-%d done", z0, z1)

    # Majority vote: at least 2 of 3 channels above threshold
    binary = vote >= 2
    del vote

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

    # Size-filtered segmentation
    log.info("Size-filtering segments (min_size=%d voxels)...", args.min_size)
    seg_ids, counts = np.unique(seg, return_counts=True)
    small_mask_ids = seg_ids[(counts < args.min_size) & (seg_ids != 0)]
    seg_filtered = seg.copy()
    if len(small_mask_ids):
        remove_mask = np.isin(seg, small_mask_ids)
        seg_filtered[remove_mask] = 0
    n_kept = int(((counts >= args.min_size) & (seg_ids != 0)).sum())
    log.info("Kept %d / %d segments (removed %d small)", n_kept, n_segs, len(small_mask_ids))

    if "pred_seg_filtered" in store:
        del store["pred_seg_filtered"]
    filt_arr = store.create_array(
        "pred_seg_filtered", shape=seg_filtered.shape, chunks=(64, 64, 64),
        dtype=np.uint64, compressors=blosc,
    )
    filt_arr[:] = seg_filtered
    filt_arr.attrs.update({"resolution_nm": res_nm, "axes": ["x", "y", "z"],
                           "threshold": args.threshold, "min_size": args.min_size})
    log.info("Saved pred_seg_filtered to zarr")
    log.info("Done. Run: python scripts/view_neuroglancer.py --tunnel --seg --pred-seg")


if __name__ == "__main__":
    main()
