"""
Offline training from a local zarr (downloaded by download_microns_chunk.py).
Useful when CloudVolume access is unavailable.

Usage:
    python scripts/train_from_zarr.py data/microns_sample.zarr
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from lowresseg.data.affinity import seg_to_affinities
from lowresseg.models.unet import AffinityUNet
from lowresseg.models.loss import BalancedAffinityLoss

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ZarrAffinityDataset(Dataset):
    def __init__(self, zarr_path: str, patch_size: int = 64, n_samples: int = 500) -> None:
        import zarr

        store = zarr.open_group(zarr_path, mode="r")
        self.em = np.array(store["em"]).astype(np.float32)
        self.seg = np.array(store["seg"]).astype(np.uint64)
        self.patch_size = patch_size
        self.n_samples = n_samples
        log.info("Loaded em=%s seg=%s", self.em.shape, self.seg.shape)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, _idx: int) -> dict:
        import random

        p = self.patch_size
        sx, sy, sz = self.em.shape
        xi = random.randint(0, max(sx - p, 0))
        yi = random.randint(0, max(sy - p, 0))
        zi = random.randint(0, max(sz - p, 0))

        em_p = self.em[xi:xi+p, yi:yi+p, zi:zi+p].astype(np.float32)
        seg_p = self.seg[xi:xi+p, yi:yi+p, zi:zi+p]

        lo, hi = np.percentile(em_p, [1, 99])
        em_p = np.clip(em_p, lo, hi)
        em_p = ((em_p - lo) / (hi - lo + 1e-6)).astype(np.float32)

        affs = seg_to_affinities(seg_p)
        weights = (seg_p > 0).astype(np.float32)

        return {
            "em": torch.from_numpy(em_p[None]),
            "affinities": torch.from_numpy(affs),
            "weights": torch.from_numpy(weights[None]),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_path")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output", default="runs/zarr_train")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    ds = ZarrAffinityDataset(args.zarr_path, patch_size=args.patch_size, n_samples=args.steps * args.batch_size)
    n_val = max(len(ds) // 10, 10)
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    model = AffinityUNet(channels=(16, 32, 64, 128), strides=(2, 2, 2)).to(device)
    log.info("Parameters: {:,}".format(sum(p.numel() for p in model.parameters())))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = BalancedAffinityLoss()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    step = 0
    for batch in train_loader:
        model.train()
        em = batch["em"].to(device)
        affs = batch["affinities"].to(device)
        weights = batch["weights"].to(device)

        opt.zero_grad()
        pred = model(em)
        loss = criterion(pred, affs, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        step += 1
        if step % 100 == 0:
            log.info("Step %d | loss=%.4f", step, loss.item())

    torch.save({"model": model.state_dict(), "step": step}, f"{args.output}/checkpoint_final.pt")
    log.info("Saved to %s/checkpoint_final.pt", args.output)


if __name__ == "__main__":
    main()
