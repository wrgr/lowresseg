"""Whole-volume affinity prediction entry point."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import zarr
import hydra
from omegaconf import DictConfig

from .data.volume import open_volume, fetch_chunk
from .models.unet import build_model
from .postprocess.tiling import tile_predict

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs/train", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    output_zarr = cfg.get("output_zarr", "affinities.zarr")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not checkpoint:
        raise ValueError("Provide checkpoint=path/to/checkpoint.pt on the command line")

    model = build_model(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info("Loaded checkpoint from step %d", ckpt["step"])

    log.info("Fetching EM volume...")
    em_vol = open_volume(cfg.data.em_path, cfg.data.em_mip, cfg.data.cache_dir)
    em_scale = 2 ** cfg.data.em_mip
    start = tuple(s // em_scale for s in cfg.data.bbox_start)
    end = tuple(e // em_scale for e in cfg.data.bbox_end)
    em = fetch_chunk(em_vol, start, end).astype(np.float32)
    log.info("EM shape: %s", em.shape)

    log.info("Running tiled inference...")
    affs = tile_predict(
        em,
        model=model,
        patch_size=tuple(cfg.data.patch_size),
        overlap=tuple(cfg.data.tile_overlap),
        device=device,
        batch_size=1,
    )
    log.info("Affinity output shape: %s", affs.shape)

    log.info("Saving affinities to %s", output_zarr)
    store = zarr.open(output_zarr, mode="w")
    store.create_dataset(
        "affinities",
        data=affs,
        chunks=(3, 64, 64, 64),
        compressor=zarr.Blosc(cname="zstd", clevel=3),
    )
    store["affinities"].attrs["voxel_size_um"] = 1.0
    store["affinities"].attrs["axes"] = ["c", "x", "y", "z"]
    log.info("Done.")


if __name__ == "__main__":
    main()
