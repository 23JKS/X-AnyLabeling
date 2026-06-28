# -*- coding: utf-8 -*-
"""Training script for lightweight track centerline model.

Usage:
    python train.py                        # uses config.yaml by default
    python train.py --config my_cfg.yaml   # alternative config
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root and the original project to sys.path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, r"D:\desk\data\log\project")

from model import LightTrackSegModel
from datasets.track_dataset import TrackDataset
from models.losses import TotalLoss
from utils.cvat_parser import parse_cvat_xml
from utils.metrics import compute_semantic_metrics


# ================================================================
#  Helpers
# ================================================================

def _load_config(path: str) -> dict:
    if not os.path.isabs(path):
        path = os.path.join(str(HERE), path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_summary(annotations: dict) -> None:
    total_tracks = sum(len(v) for v in annotations.values())
    print(f"Images: {len(annotations)}, tracks: {total_tracks}")
    for name, polys in annotations.items():
        print(f"   {name}: {len(polys)} tracks")


def _save_checkpoint(model, optimizer, epoch, val_dice, cfg, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": val_dice,
            "config": cfg,
        },
        path,
    )
    print(f"  Checkpoint saved: {path}")


def _convert_checkpoint_for_inference(model, cfg, src_path, dst_path):
    """Save a lightweight checkpoint for use with X-AnyLabeling."""
    state = torch.load(src_path, map_location="cpu")
    torch.save(
        {
            "model_state_dict": state["model_state_dict"],
            "config": {
                "model": {
                    "encoder": cfg["model"]["encoder"],
                    "in_channels": cfg["model"]["in_channels"],
                    "embedding_dim": cfg["model"]["embedding_dim"],
                    "decoder_channels": cfg["model"]["decoder_channels"],
                },
                "data": {"patch_size": cfg["data"]["patch_size"]},
                "postprocess": cfg["postprocess"],
            },
        },
        dst_path,
    )
    size_mb = os.path.getsize(dst_path) / 1024 / 1024
    print(f"  Inference checkpoint saved: {dst_path} ({size_mb:.1f} MB)")


# ================================================================
#  Training loop
# ================================================================

def train(config_path: str = "config.yaml") -> None:
    cfg = _load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Parse annotations ---
    root_dir = cfg["data"]["root_dir"]
    xml_path = os.path.join(root_dir, cfg["data"]["annotation_file"])
    annotations = parse_cvat_xml(xml_path)
    _print_summary(annotations)

    # --- Train / val split ---
    image_names = sorted(annotations.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(image_names)
    split_idx = max(1, int(len(image_names) * cfg["data"]["train_split"]))
    train_names = image_names[:split_idx]
    val_names = image_names[split_idx:]
    print(f"\nTrain: {train_names}")
    print(f"Val:   {val_names}")

    # --- Datasets ---
    patch_size = cfg["data"]["patch_size"]
    cw = cfg["data"]["centerline_width"]
    train_ds = TrackDataset(root_dir, annotations, train_names, patch_size, cw, is_train=True)
    val_ds = TrackDataset(root_dir, annotations, val_names, patch_size, cw, is_train=False)
    print(f"\nTrain patches: {len(train_ds)}, Val images: {len(val_ds)}")

    use_pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=use_pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=use_pin)

    # --- Model ---
    model = LightTrackSegModel(
        encoder_name=cfg["model"]["encoder"],
        encoder_weights=cfg["model"]["encoder_weights"],
        in_channels=cfg["model"]["in_channels"],
        embedding_dim=cfg["model"]["embedding_dim"],
        decoder_channels=cfg["model"]["decoder_channels"],
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {total_params/1e6:.1f} M")

    # --- Loss, optimizer, scheduler ---
    criterion = TotalLoss(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    scaler = GradScaler(enabled=cfg["training"]["mixed_precision"])

    # --- Logging ---
    log_dir = Path(cfg["logging"]["log_dir"])
    chkpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    chkpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    # --- State ---
    best_val_dice = -1.0
    patience_left = cfg["training"]["early_stopping_patience"]
    threshold = cfg["postprocess"]["semantic_threshold"]

    # ============================================================
    #  Training loop
    # ============================================================
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        train_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['training']['epochs']}")
        for batch_idx, (images, centerline_gt, instance_gt) in enumerate(pbar):
            images = images.to(device)
            centerline_gt = centerline_gt.to(device)
            instance_gt = instance_gt.to(device)

            optimizer.zero_grad()
            with autocast(enabled=cfg["training"]["mixed_precision"]):
                semantic, embedding = model(images)
                losses = criterion(semantic, embedding, centerline_gt, instance_gt)
                loss = losses["total"]

            scaler.scale(loss).backward()
            if cfg["training"]["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if batch_idx % cfg["logging"]["log_interval"] == 0:
                step = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar("Train/Loss", loss.item(), step)

        avg_train_loss = train_loss_sum / len(train_loader)
        writer.add_scalar("Train/Epoch_Loss", avg_train_loss, epoch)
        print(f"Epoch {epoch} train loss: {avg_train_loss:.4f}")
        scheduler.step()

        # --- Validation ---
        if epoch % cfg["logging"]["val_interval"] == 0:
            model.eval()
            val_dice_sum = 0.0
            with torch.no_grad():
                for images, centerline_gt, _ in val_loader:
                    images = images.to(device)
                    centerline_gt = centerline_gt.to(device)
                    semantic, _ = model(images)
                    sem_prob = torch.sigmoid(semantic)
                    metrics = compute_semantic_metrics(sem_prob, centerline_gt, threshold)
                    val_dice_sum += metrics["dice"]

            val_dice = val_dice_sum / len(val_loader)
            writer.add_scalar("Val/Dice", val_dice, epoch)
            print(f"Epoch {epoch} val dice: {val_dice:.4f}")

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                patience_left = cfg["training"]["early_stopping_patience"]
                _save_checkpoint(model, optimizer, epoch, val_dice, cfg,
                    str(chkpt_dir / "best_model.pth"))
                _convert_checkpoint_for_inference(model, cfg,
                    str(chkpt_dir / "best_model.pth"),
                    str(chkpt_dir / "best_model_inference.pth"))
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"Early stopping at epoch {epoch}")
                    break

        if epoch % cfg["logging"]["save_interval"] == 0:
            _save_checkpoint(model, optimizer, epoch, val_dice, cfg,
                str(chkpt_dir / f"checkpoint_epoch_{epoch}.pth"))

    writer.close()
    print(f"\nTraining complete. Best val dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    train(args.config)
