"""PyTorch Dataset for patch-based training on CloudVolume EM + segmentation."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import monai.transforms as mt

from .volume import open_volume, fetch_chunk, voxel_size_nm
from .affinity import seg_to_affinities


class AffinityDataset(Dataset):
    """Randomly samples patches from EM + segmentation CloudVolume pairs."""

    def __init__(
        self,
        em_path: str,
        seg_path: str,
        em_mip: int,
        seg_mip: int,
        bbox_start: tuple[int, int, int],
        bbox_end: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        num_samples: int = 1000,
        augment: bool = True,
        cache_dir: str | None = None,
        clip_percentile: tuple[float, float] = (1.0, 99.0),
    ) -> None:
        self.em_vol = open_volume(em_path, em_mip, cache_dir)
        self.seg_vol = open_volume(seg_path, seg_mip, cache_dir)

        # Convert bbox from mip0 to each volume's mip level
        em_scale = 2 ** em_mip
        seg_scale = 2 ** seg_mip
        self.em_start = tuple(s // em_scale for s in bbox_start)
        self.em_end = tuple(e // em_scale for e in bbox_end)
        self.seg_start = tuple(s // seg_scale for s in bbox_start)
        self.seg_end = tuple(e // seg_scale for e in bbox_end)

        self.patch_size = np.array(patch_size, dtype=np.int32)
        self.num_samples = num_samples
        self.clip_lo, self.clip_hi = clip_percentile
        self.augment = augment
        self._build_transforms()

    def _build_transforms(self) -> None:
        xforms: list[Any] = []
        if self.augment:
            xforms += [
                mt.RandFlipd(keys=["em"], prob=0.5, spatial_axis=0),
                mt.RandFlipd(keys=["em"], prob=0.5, spatial_axis=1),
                mt.RandFlipd(keys=["em"], prob=0.5, spatial_axis=2),
                mt.RandRotate90d(keys=["em"], prob=0.5, max_k=3, spatial_axes=(0, 1)),
                mt.RandScaleIntensityd(keys=["em"], prob=0.8, factors=0.2),
                mt.RandShiftIntensityd(keys=["em"], prob=0.8, offsets=0.1),
                mt.RandGaussianNoised(keys=["em"], prob=0.3, mean=0.0, std=0.02),
                mt.RandGaussianSmoothd(
                    keys=["em"],
                    prob=0.2,
                    sigma_x=(0.5, 1.5),
                    sigma_y=(0.5, 1.5),
                    sigma_z=(0.5, 1.5),
                ),
            ]
        self.transforms = mt.Compose(xforms) if xforms else None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, _idx: int) -> dict[str, torch.Tensor]:
        em_patch, seg_patch = self._random_patch()

        # Normalise EM to [0, 1]
        lo = np.percentile(em_patch, self.clip_lo)
        hi = np.percentile(em_patch, self.clip_hi)
        em_patch = np.clip(em_patch, lo, hi).astype(np.float32)
        em_patch = (em_patch - lo) / (hi - lo + 1e-6)

        # Compute affinity targets from segmentation patch
        affs = seg_to_affinities(seg_patch)  # (3, X, Y, Z)

        # Build weight mask: foreground voxels only
        weights = (seg_patch > 0).astype(np.float32)

        em_tensor = torch.from_numpy(em_patch[None])  # (1, X, Y, Z)
        aff_tensor = torch.from_numpy(affs)           # (3, X, Y, Z)
        wt_tensor = torch.from_numpy(weights[None])   # (1, X, Y, Z)

        sample: dict[str, Any] = {"em": em_tensor}
        if self.transforms is not None:
            sample = self.transforms(sample)
        em_tensor = sample["em"]

        return {"em": em_tensor, "affinities": aff_tensor, "weights": wt_tensor}

    def _random_patch(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample a random aligned patch from EM and seg volumes."""
        em_size = np.array(self.em_end) - np.array(self.em_start)
        max_start = em_size - self.patch_size
        # Ensure there's room
        max_start = np.maximum(max_start, 0)

        offset = np.array([random.randint(0, int(m)) for m in max_start])
        em_s = np.array(self.em_start) + offset
        em_e = em_s + self.patch_size

        em_patch = fetch_chunk(self.em_vol, tuple(em_s), tuple(em_e))

        # Map EM patch coords to seg coords
        seg_scale_rel = (2 ** self.em_vol.mip) / (2 ** self.seg_vol.mip)
        seg_s = (em_s * seg_scale_rel).astype(int)
        seg_e = (em_e * seg_scale_rel).astype(int)
        seg_patch = fetch_chunk(self.seg_vol, tuple(seg_s), tuple(seg_e))

        # Resize seg to match em patch shape if needed
        if seg_patch.shape != em_patch.shape:
            from skimage.transform import resize
            seg_patch = resize(
                seg_patch.astype(float),
                em_patch.shape,
                order=0,
                anti_aliasing=False,
            ).astype(np.uint64)

        return em_patch, seg_patch


def build_dataloader(cfg: Any, split: str = "train") -> DataLoader:
    ds = AffinityDataset(
        em_path=cfg.data.em_path,
        seg_path=cfg.data.seg_path,
        em_mip=cfg.data.em_mip,
        seg_mip=cfg.data.seg_mip,
        bbox_start=cfg.data.bbox_start,
        bbox_end=cfg.data.bbox_end,
        patch_size=cfg.data.patch_size,
        num_samples=cfg.train.batch_size * 1000 if split == "train" else 200,
        augment=(split == "train"),
        cache_dir=cfg.data.cache_dir,
        clip_percentile=cfg.data.clip_percentile,
    )
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=cfg.train.num_workers > 0,
    )
