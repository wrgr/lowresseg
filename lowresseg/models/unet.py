"""3D UNet predicting short-range affinities, wrapping MONAI's UNet."""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import UNet


class AffinityUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 3,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        strides: tuple[int, ...] = (2, 2, 2, 2),
        num_res_units: int = 2,
        dropout: float = 0.1,
        norm: str = "BATCH",
        act: str = "LEAKYRELU",
    ) -> None:
        super().__init__()
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout,
            norm=norm,
            act=act,
        )
        # Sigmoid to squash affinities to [0, 1]
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.unet(x))


def build_model(cfg: object) -> AffinityUNet:
    m = cfg.model
    return AffinityUNet(
        spatial_dims=m.spatial_dims,
        in_channels=m.in_channels,
        out_channels=m.out_channels,
        channels=tuple(m.channels),
        strides=tuple(m.strides),
        num_res_units=m.num_res_units,
        dropout=m.dropout,
        norm=m.norm,
        act=m.act,
    )
