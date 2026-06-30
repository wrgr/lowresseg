"""Affinity-based agglomeration: mutex watershed → waterz."""

from __future__ import annotations

import numpy as np


def affinities_to_segmentation(
    affs: np.ndarray,
    threshold: float = 0.5,
    method: str = "waterz",
    merge_function: str = "mean",
    aff_threshold_seeds: float = 0.9,
    aff_threshold_connected: float = 0.3,
) -> np.ndarray:
    """
    Convert predicted affinities to a dense segmentation.

    Args:
        affs: (3, X, Y, Z) float32 short-range affinities.
        threshold: Waterz merge threshold (ignored for cc method).
        method: "waterz" | "cc" (connected components on thresholded affs).
        merge_function: waterz merge function ("mean" | "max" | "min" | "rmax").
        aff_threshold_seeds: minimum mean affinity to seed (waterz).
        aff_threshold_connected: waterz stop threshold.

    Returns:
        seg: (X, Y, Z) uint64 segmentation.
    """
    if method == "cc":
        return _cc_segmentation(affs, threshold)
    elif method == "waterz":
        return _waterz_segmentation(
            affs, merge_function, aff_threshold_seeds, aff_threshold_connected
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def _cc_segmentation(affs: np.ndarray, threshold: float) -> np.ndarray:
    """Simple connected components on voxels where all affinities > threshold."""
    import cc3d

    foreground = (affs.min(axis=0) > threshold).astype(np.uint8)
    seg = cc3d.connected_components(foreground, connectivity=26)
    return seg.astype(np.uint64)


def _waterz_segmentation(
    affs: np.ndarray,
    merge_function: str,
    aff_threshold_seeds: float,
    aff_threshold_connected: float,
) -> np.ndarray:
    import waterz

    # waterz expects (C, Z, Y, X) with C=3 for xyz affinities
    # Our affs are (C, X, Y, Z) so transpose
    affs_zyx = affs.transpose(0, 3, 2, 1).copy()

    # Generate segments at a range of thresholds, take the one we want
    # waterz yields (seg, metrics) for each threshold
    thresholds = [aff_threshold_seeds, aff_threshold_connected]
    generator = waterz.agglomerate(
        affs_zyx,
        thresholds,
        scoring_function=merge_function,
    )
    seg = None
    for seg, _ in generator:
        pass  # last threshold = most agglomerated

    if seg is None:
        raise RuntimeError("waterz returned no segmentation")

    # Transpose back to (X, Y, Z)
    return seg.transpose(2, 1, 0).astype(np.uint64)
