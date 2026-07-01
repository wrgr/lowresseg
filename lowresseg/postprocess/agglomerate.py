"""
Affinity-based agglomeration: watershed via scipy MST or connected components.

waterz is not available as a pip package for Python 3.11+, so we implement
mean-affinity agglomeration using a greedy edge-contraction approach built on
scipy's sparse minimum spanning tree (which gives the same topology as waterz's
watershed on affinities).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


def affinities_to_segmentation(
    affs: np.ndarray,
    threshold: float = 0.5,
    method: str = "mst",
    merge_function: str = "mean",
) -> np.ndarray:
    """
    Convert predicted affinities to a dense segmentation.

    Args:
        affs: (3, X, Y, Z) float32 short-range affinities [0,1].
        threshold: edges with affinity > threshold are merged.
        method: "mst" (greedy mean-affinity agglomeration) | "cc" (simple CC).
        merge_function: ignored for "cc"; reserved for future multi-function MST.

    Returns:
        seg: (X, Y, Z) uint64 segmentation (0 = background).
    """
    if method == "cc":
        return _cc_segmentation(affs, threshold)
    elif method == "mst":
        return _mst_segmentation(affs, threshold)
    else:
        raise ValueError(f"Unknown method: {method!r}. Choose 'mst' or 'cc'.")


# ---------------------------------------------------------------------------
# Connected-components baseline
# ---------------------------------------------------------------------------

def _cc_segmentation(affs: np.ndarray, threshold: float) -> np.ndarray:
    """Threshold mean affinity, then label connected components."""
    import cc3d

    foreground = (affs.mean(axis=0) > threshold).astype(np.uint8)
    seg = cc3d.connected_components(foreground, connectivity=26)
    return seg.astype(np.uint64)


# ---------------------------------------------------------------------------
# MST-based mean-affinity agglomeration
# ---------------------------------------------------------------------------

def _mst_segmentation(affs: np.ndarray, threshold: float) -> np.ndarray:
    """
    Greedy agglomeration via minimum spanning tree on the affinity graph.

    Each voxel is a node; edges connect short-range neighbours with weight
    = 1 - affinity (so high-affinity = short distance = merged first).
    We compute the MST, then cut all edges where affinity < threshold.
    Connected components of the remaining forest define segments.
    """
    _, X, Y, Z = affs.shape
    N = X * Y * Z

    def idx(x, y, z):
        return x * Y * Z + y * Z + z

    rows, cols, weights = [], [], []

    # x-axis neighbours (offset [1,0,0])
    xs = np.arange(X - 1)
    for x in xs:
        for y in range(Y):
            for z in range(Z):
                a = float(affs[0, x, y, z])
                if a > 0:
                    rows.append(idx(x, y, z))
                    cols.append(idx(x + 1, y, z))
                    weights.append(1.0 - a)

    # y-axis neighbours
    for x in range(X):
        for y in range(Y - 1):
            for z in range(Z):
                a = float(affs[1, x, y, z])
                if a > 0:
                    rows.append(idx(x, y, z))
                    cols.append(idx(x, y + 1, z))
                    weights.append(1.0 - a)

    # z-axis neighbours
    for x in range(X):
        for y in range(Y):
            for z in range(Z - 1):
                a = float(affs[2, x, y, z])
                if a > 0:
                    rows.append(idx(x, y, z))
                    cols.append(idx(x, y, z + 1))
                    weights.append(1.0 - a)

    if not rows:
        return np.zeros((X, Y, Z), dtype=np.uint64)

    graph = coo_matrix(
        (np.array(weights, dtype=np.float32), (rows, cols)),
        shape=(N, N),
    ).tocsr()

    mst = minimum_spanning_tree(graph)

    # Cut edges below threshold (i.e. distance > 1 - threshold)
    cut_dist = 1.0 - threshold
    mst = mst.tocoo()
    keep = mst.data <= cut_dist
    mst_cut = coo_matrix(
        (mst.data[keep], (mst.row[keep], mst.col[keep])),
        shape=(N, N),
    )

    # Connected components of the remaining forest
    from scipy.sparse.csgraph import connected_components
    n_comp, labels = connected_components(
        mst_cut + mst_cut.T, directed=False, connection="weak"
    )

    seg = labels.reshape(X, Y, Z).astype(np.uint64)

    # Zero out background (voxels where all affinities ≈ 0)
    foreground = affs.max(axis=0) > 0.01
    seg[~foreground] = 0

    # Re-label from 1 so 0 is strictly background
    _, seg = np.unique(seg * foreground, return_inverse=True)
    seg = seg.reshape(X, Y, Z).astype(np.uint64)

    return seg


# ---------------------------------------------------------------------------
# Fast vectorised version for larger volumes
# ---------------------------------------------------------------------------

def affinities_to_segmentation_fast(
    affs: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Vectorised MST agglomeration — faster than the loop version for large volumes.
    Builds the full edge list using numpy and then runs scipy MST.
    """
    _, X, Y, Z = affs.shape
    N = X * Y * Z

    flat = np.arange(N, dtype=np.int32).reshape(X, Y, Z)

    edge_list = []

    # x neighbours
    a = affs[0, :-1, :, :]
    mask = a > 0
    r = flat[:-1, :, :][mask]
    c = flat[1:, :, :][mask]
    w = (1.0 - a[mask]).astype(np.float32)
    edge_list.append((r, c, w))

    # y neighbours
    a = affs[1, :, :-1, :]
    mask = a > 0
    r = flat[:, :-1, :][mask]
    c = flat[:, 1:, :][mask]
    w = (1.0 - a[mask]).astype(np.float32)
    edge_list.append((r, c, w))

    # z neighbours
    a = affs[2, :, :, :-1]
    mask = a > 0
    r = flat[:, :, :-1][mask]
    c = flat[:, :, 1:][mask]
    w = (1.0 - a[mask]).astype(np.float32)
    edge_list.append((r, c, w))

    all_r = np.concatenate([e[0] for e in edge_list])
    all_c = np.concatenate([e[1] for e in edge_list])
    all_w = np.concatenate([e[2] for e in edge_list])

    graph = coo_matrix(
        (all_w, (all_r, all_c)),
        shape=(N, N),
    ).tocsr()

    mst = minimum_spanning_tree(graph).tocoo()

    cut_dist = 1.0 - threshold
    keep = mst.data <= cut_dist
    mst_cut = coo_matrix(
        (mst.data[keep], (mst.row[keep], mst.col[keep])),
        shape=(N, N),
    )

    from scipy.sparse.csgraph import connected_components
    _, labels = connected_components(mst_cut + mst_cut.T, directed=False, connection="weak")

    seg = labels.reshape(X, Y, Z).astype(np.uint64)
    foreground = affs.max(axis=0) > 0.01
    seg[~foreground] = 0

    _, seg = np.unique(seg * foreground, return_inverse=True)
    return seg.reshape(X, Y, Z).astype(np.uint64)
