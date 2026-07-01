"""Balanced binary cross-entropy loss for affinity prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedAffinityLoss(nn.Module):
    """BCE with per-batch class balancing and optional foreground mask."""

    def __init__(self, pos_weight_clamp: float = 10.0) -> None:
        super().__init__()
        self.pos_weight_clamp = pos_weight_clamp

    def forward(
        self,
        pred: torch.Tensor,    # (B, C, X, Y, Z) in [0,1]
        target: torch.Tensor,  # (B, C, X, Y, Z) in {0,1}
        mask: torch.Tensor | None = None,  # (B, 1, X, Y, Z) foreground
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.expand_as(pred)

        losses = []
        for c in range(pred.shape[1]):
            p = pred[:, c]
            t = target[:, c]
            m = mask[:, c] if mask is not None else torch.ones_like(t)

            # Only compute loss where mask is valid
            valid = m > 0
            if valid.sum() == 0:
                continue

            p_v = p[valid]
            t_v = t[valid]

            # Balanced pos_weight: ratio of negatives to positives
            n_pos = t_v.sum().clamp(min=1)
            n_neg = (1 - t_v).sum().clamp(min=1)
            pos_w = (n_neg / n_pos).clamp(max=self.pos_weight_clamp)

            loss = F.binary_cross_entropy(p_v, t_v, reduction="none")
            weights = torch.where(t_v > 0.5, pos_w * torch.ones_like(t_v), torch.ones_like(t_v))
            losses.append((loss * weights).mean())

        return torch.stack(losses).mean() if losses else pred.sum() * 0.0
