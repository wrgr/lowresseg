"""Evaluate segmentation against ground truth (Rand, VI, skeleton metrics)."""

from __future__ import annotations

import argparse
import logging

import numpy as np
import zarr
from scipy.sparse import coo_matrix

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def rand_index(seg: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Compute Rand split/merge errors."""
    seg = seg.ravel()
    gt = gt.ravel()

    # Only evaluate on foreground
    mask = gt > 0
    seg, gt = seg[mask], gt[mask]

    # Build contingency table
    n = len(seg)
    seg_ids = np.unique(seg, return_inverse=True)[1]
    gt_ids = np.unique(gt, return_inverse=True)[1]
    contingency = coo_matrix(
        (np.ones(n, dtype=np.float64), (seg_ids, gt_ids))
    ).toarray()

    p_ij = contingency / n
    a_i = p_ij.sum(axis=1)
    b_j = p_ij.sum(axis=0)

    rand_split = 1.0 - (p_ij ** 2).sum() / (a_i ** 2).sum()
    rand_merge = 1.0 - (p_ij ** 2).sum() / (b_j ** 2).sum()

    return {"rand_split": float(rand_split), "rand_merge": float(rand_merge)}


def variation_of_information(seg: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Compute Variation of Information (split and merge components)."""
    seg = seg.ravel()
    gt = gt.ravel()

    mask = gt > 0
    seg, gt = seg[mask], gt[mask]

    n = len(seg)
    seg_ids = np.unique(seg, return_inverse=True)[1]
    gt_ids = np.unique(gt, return_inverse=True)[1]
    contingency = coo_matrix(
        (np.ones(n, dtype=np.float64), (seg_ids, gt_ids))
    ).toarray()

    p_ij = contingency / n
    a_i = p_ij.sum(axis=1, keepdims=True) + 1e-10
    b_j = p_ij.sum(axis=0, keepdims=True) + 1e-10

    h_seg_given_gt = -np.sum(p_ij * np.log2(p_ij / b_j + 1e-10))
    h_gt_given_seg = -np.sum(p_ij * np.log2(p_ij / a_i + 1e-10))

    return {"vi_split": float(h_seg_given_gt), "vi_merge": float(h_gt_given_seg)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation quality")
    parser.add_argument("seg_zarr", help="Predicted segmentation zarr")
    parser.add_argument("gt_zarr", help="Ground truth segmentation zarr")
    parser.add_argument("--seg-dataset", default="seg")
    parser.add_argument("--gt-dataset", default="seg")
    args = parser.parse_args()

    log.info("Loading predicted segmentation from %s", args.seg_zarr)
    seg = zarr.open(args.seg_zarr, mode="r")[args.seg_dataset][:]

    log.info("Loading ground truth from %s", args.gt_zarr)
    gt = zarr.open(args.gt_zarr, mode="r")[args.gt_dataset][:]

    if seg.shape != gt.shape:
        raise ValueError(f"Shape mismatch: seg={seg.shape} vs gt={gt.shape}")

    rand = rand_index(seg, gt)
    vi = variation_of_information(seg, gt)

    results = {**rand, **vi}
    log.info("Results:")
    for k, v in results.items():
        log.info("  %s: %.4f", k, v)


if __name__ == "__main__":
    main()
