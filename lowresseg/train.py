"""Training entry point."""

from __future__ import annotations

import os
import logging
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from .data.dataset import build_dataloader
from .models.unet import build_model
from .models.loss import BalancedAffinityLoss

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs/train", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "tb")

    model = build_model(cfg).to(device)
    log.info("Model parameters: {:,}".format(sum(p.numel() for p in model.parameters())))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.max_steps,
        eta_min=cfg.scheduler.eta_min,
    )
    criterion = BalancedAffinityLoss()

    start_step = 0
    if cfg.train.resume:
        ckpt = torch.load(cfg.train.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        log.info("Resumed from step %d", start_step)

    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val")

    step = start_step
    model.train()
    loader_iter = iter(train_loader)

    with tqdm(total=cfg.train.max_steps, initial=start_step, desc="Training") as pbar:
        while step < cfg.train.max_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            em = batch["em"].to(device)
            affs = batch["affinities"].to(device)
            weights = batch["weights"].to(device)

            optimizer.zero_grad()
            pred = model(em)
            loss = criterion(pred, affs, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)

            if step % cfg.train.val_interval == 0:
                val_loss = _validate(model, val_loader, criterion, device)
                writer.add_scalar("val/loss", val_loss, step)
                log.info("Step %d | val_loss=%.4f", step, val_loss)
                model.train()

            if step % cfg.train.checkpoint_interval == 0:
                ckpt_path = out_dir / f"checkpoint_{step:07d}.pt"
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "cfg": OmegaConf.to_container(cfg),
                    },
                    ckpt_path,
                )
                log.info("Saved checkpoint: %s", ckpt_path)

    writer.close()
    log.info("Training complete.")


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: BalancedAffinityLoss,
    device: torch.device,
) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        em = batch["em"].to(device)
        affs = batch["affinities"].to(device)
        weights = batch["weights"].to(device)
        pred = model(em)
        loss = criterion(pred, affs, weights)
        total += loss.item()
        count += 1
    return total / max(count, 1)


if __name__ == "__main__":
    main()
