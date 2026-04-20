"""
train.py
Supervised fine-tuning (SFT) pipeline for deepfake detection.

Strategy:
- Start from EfficientNet-B0 pretrained on ImageNet
- Phase 1 (epochs 1-3):  freeze backbone, train only the new classifier head
                          → fast convergence without destroying pretrained features
- Phase 2 (epochs 4+):   unfreeze all layers with a lower LR for the backbone
                          → full fine-tuning with differential learning rates
- BCEWithLogitsLoss for numerical stability (combines sigmoid + BCE in one op)
- CosineAnnealingLR scheduler → smooth LR decay, avoids oscillating around minima
- Best checkpoint saved based on validation AUC-ROC (more meaningful than accuracy
  on potentially imbalanced data)
"""

import os
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from data_loader import get_dataloaders
from detectors.model_based import build_efficientnet


def train_one_epoch(model, loader, optimizer, criterion, device) -> dict:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for tensors, labels, _ in tqdm(loader, desc="  train", leave=False):
        tensors = tensors.to(device)
        labels  = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(tensors).squeeze(1)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total   += len(labels)

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for tensors, labels, _ in tqdm(loader, desc="  val  ", leave=False):
        tensors = tensors.to(device)
        labels  = labels.float().to(device)

        logits = model(tensors).squeeze(1)
        loss   = criterion(logits, labels)
        probs  = torch.sigmoid(logits)

        total_loss += loss.item() * len(labels)
        preds = (probs >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total   += len(labels)

        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return {
        "loss":     total_loss / total,
        "accuracy": correct / total,
        "auc_roc":  auc,
    }


def freeze_backbone(model: nn.Module):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def get_optimizer(model: nn.Module, head_lr: float, backbone_lr: float):
    """
    Differential learning rates: backbone gets 10x lower LR than head.
    This is standard practice in transfer learning — the backbone already has
    good features; we don't want to clobber them with large gradient updates.
    """
    backbone_params = [p for n, p in model.named_parameters()
                       if "classifier" not in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if "classifier" in n and p.requires_grad]
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ], weight_decay=1e-4)


def train(args):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  Deepfake Detection — Supervised Fine-Tuning")
    print(f"  Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    print(f"{'='*60}\n")

    train_loader, val_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = build_efficientnet(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_auc = 0.0
    history  = []
    warmup_epochs = min(3, args.epochs // 3)  # head-only warmup

    print(f"Phase 1: Warmup ({warmup_epochs} epochs) — training classifier head only\n")

    for epoch in range(1, args.epochs + 1):
        # ── Phase transitions ──────────────────────────────────────────────────
        if epoch == 1:
            freeze_backbone(model)
            optimizer = get_optimizer(model, head_lr=args.lr, backbone_lr=0.0)
            scheduler = CosineAnnealingLR(optimizer, T_max=warmup_epochs, eta_min=1e-6)

        if epoch == warmup_epochs + 1:
            print(f"\nPhase 2: Full fine-tuning — unfreezing backbone\n")
            unfreeze_all(model)
            optimizer = get_optimizer(model,
                                      head_lr=args.lr,
                                      backbone_lr=args.lr / 10)
            scheduler = CosineAnnealingLR(optimizer,
                                          T_max=args.epochs - warmup_epochs,
                                          eta_min=1e-7)

        # ── Train + validate ───────────────────────────────────────────────────
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics   = validate(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch":        epoch,
            "train_loss":   round(train_metrics["loss"],     4),
            "train_acc":    round(train_metrics["accuracy"], 4),
            "val_loss":     round(val_metrics["loss"],       4),
            "val_acc":      round(val_metrics["accuracy"],   4),
            "val_auc":      round(val_metrics["auc_roc"],    4),
            "lr":           round(optimizer.param_groups[-1]["lr"], 7),
        }
        history.append(row)

        print(f"Epoch {epoch:02d}/{args.epochs:02d}  "
              f"train_loss={row['train_loss']:.4f}  train_acc={row['train_acc']:.4f}  "
              f"val_loss={row['val_loss']:.4f}  val_acc={row['val_acc']:.4f}  "
              f"val_auc={row['val_auc']:.4f}  lr={row['lr']:.2e}")

        # ── Save best checkpoint ───────────────────────────────────────────────
        if val_metrics["auc_roc"] > best_auc:
            best_auc = val_metrics["auc_roc"]
            ckpt_path = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_auc":          best_auc,
                "val_acc":          val_metrics["accuracy"],
                "args":             vars(args),
            }, ckpt_path)
            print(f"  ✓ New best AUC={best_auc:.4f} — checkpoint saved to {ckpt_path}")

    # ── Save training history ──────────────────────────────────────────────────
    history_path = checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Training complete. Best val AUC: {best_auc:.4f}")
    print(f"✓ Checkpoint: {checkpoint_dir}/best_model.pth")
    print(f"✓ History:    {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B0 on CelebDF")
    parser.add_argument("--data_dir",       default="./data",          help="Dir with real/ and fake/ subdirs")
    parser.add_argument("--checkpoint_dir", default="./checkpoints",   help="Where to save model checkpoints")
    parser.add_argument("--epochs",         type=int,   default=10,    help="Total training epochs")
    parser.add_argument("--batch_size",     type=int,   default=32,    help="Batch size")
    parser.add_argument("--lr",             type=float, default=1e-4,  help="Head learning rate")
    parser.add_argument("--num_workers",    type=int,   default=4,     help="DataLoader workers")
    args = parser.parse_args()
    train(args)