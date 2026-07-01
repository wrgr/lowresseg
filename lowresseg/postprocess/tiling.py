"""Tile-and-stitch whole-volume inference with overlap blending."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def tile_predict(
    em: np.ndarray,
    model: Callable[[torch.Tensor], torch.Tensor],
    patch_size: tuple[int, int, int],
    overlap: tuple[int, int, int],
    device: str | torch.device = "cuda",
    batch_size: int = 1,
    normalize: bool = True,
    clip_percentile: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """
    Run model over full EM volume using overlapping tiles.

    Args:
        em: (X, Y, Z) float32 or uint8 EM volume.
        model: callable (B, 1, X, Y, Z) -> (B, C, X, Y, Z).
        patch_size: (px, py, pz) tile size.
        overlap: (ox, oy, oz) overlap on each side.
        device: torch device.
        normalize: normalise each tile independently.

    Returns:
        affs: (C, X, Y, Z) float32 predicted affinities.
    """
    em = em.astype(np.float32)
    if normalize:
        lo, hi = np.percentile(em, clip_percentile)
        em = np.clip(em, lo, hi)
        em = (em - lo) / (hi - lo + 1e-6)

    px, py, pz = patch_size
    ox, oy, oz = overlap
    sx, sy, sz = em.shape

    # Stride = patch - 2*overlap
    stride_x = max(px - 2 * ox, 1)
    stride_y = max(py - 2 * oy, 1)
    stride_z = max(pz - 2 * oz, 1)

    # Build tile start coordinates, ensuring we cover the full volume
    starts_x = list(range(0, max(sx - px, 0) + 1, stride_x))
    starts_y = list(range(0, max(sy - py, 0) + 1, stride_y))
    starts_z = list(range(0, max(sz - pz, 0) + 1, stride_z))
    if starts_x[-1] + px < sx:
        starts_x.append(sx - px)
    if starts_y[-1] + py < sy:
        starts_y.append(sy - py)
    if starts_z[-1] + pz < sz:
        starts_z.append(sz - pz)

    # Peek at output channels
    test_in = torch.zeros(1, 1, px, py, pz, device=device)
    with torch.no_grad():
        test_out = model(test_in)
    n_channels = test_out.shape[1]

    aff_sum = np.zeros((n_channels, sx, sy, sz), dtype=np.float32)
    weight_sum = np.zeros((sx, sy, sz), dtype=np.float32)

    # Hanning window for smooth blending
    w = _hanning_window(patch_size)

    device = torch.device(device)
    batch_inputs: list[tuple] = []

    def _flush(batch: list[tuple]) -> None:
        if not batch:
            return
        xs = torch.cat([b[0] for b in batch], dim=0).to(device)
        with torch.no_grad():
            preds = model(xs).cpu().numpy()
        for pred, (_, i, j, k) in zip(preds, batch):
            ie = min(i + px, sx)
            je = min(j + py, sy)
            ke = min(k + pz, sz)
            aff_sum[:, i:ie, j:je, k:ke] += pred[:, :ie-i, :je-j, :ke-k] * w[:ie-i, :je-j, :ke-k]
            weight_sum[i:ie, j:je, k:ke] += w[:ie-i, :je-j, :ke-k]

    for i in starts_x:
        for j in starts_y:
            for k in starts_z:
                ie, je, ke = min(i + px, sx), min(j + py, sy), min(k + pz, sz)
                tile = em[i:ie, j:je, k:ke]
                # Pad if tile is smaller than patch_size
                pad = [(0, px - tile.shape[0]), (0, py - tile.shape[1]), (0, pz - tile.shape[2])]
                tile = np.pad(tile, pad, mode="reflect")
                t = torch.from_numpy(tile[None, None].astype(np.float32))  # (1, 1, px, py, pz)
                batch_inputs.append((t, i, j, k))
                if len(batch_inputs) >= batch_size:
                    _flush(batch_inputs)
                    batch_inputs = []

    _flush(batch_inputs)

    weight_sum = np.maximum(weight_sum, 1e-6)
    return aff_sum / weight_sum[None]


def _hanning_window(size: tuple[int, int, int]) -> np.ndarray:
    wx = np.hanning(size[0]).astype(np.float32)
    wy = np.hanning(size[1]).astype(np.float32)
    wz = np.hanning(size[2]).astype(np.float32)
    return wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
