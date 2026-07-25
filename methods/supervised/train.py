"""
Training loops for the supervised two-stage method (SHIPPED).

Usage:
    python -m methods.supervised.train
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (
    CHECKPOINTS_DIR,
    DESPECKLED_DIR,
    TABLES_DIR,
    build_patch_dataset,
    load_hom38_folds,
    make_stage1_train_val_split,
    normalize_patch,
    parse_ground_truths_table,
)
from common.config import GT_DIR
from .datasets import LabeledPatchDataset, ResNetPatchDataset
from .model import PatchClassifier, build_resnet_grader

STAGE1_CKPT_DIR = os.path.join(CHECKPOINTS_DIR, "supervised", "stage1_detector")
STAGE2_CKPT_DIR = os.path.join(CHECKPOINTS_DIR, "supervised", "stage2_grader")
RESNET_CKPT_DIR = os.path.join(CHECKPOINTS_DIR, "supervised", "stage2_resnet")


def train_stage1_detector_fold(
    fold, id_to_basename, fold_idx,
    batch_size=64, max_epochs=200, patience=10, lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    train_ids, val_ids = make_stage1_train_val_split(fold["train_ids"])
    X_train, y_train, _ = build_patch_dataset(train_ids, id_to_basename, DESPECKLED_DIR)
    X_val, y_val, _ = build_patch_dataset(val_ids, id_to_basename, DESPECKLED_DIR)

    y_train_bin = (y_train != 0).astype(np.float32)
    y_val_bin = (y_val != 0).astype(np.float32)

    train_loader = DataLoader(LabeledPatchDataset(X_train, y_train_bin), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(LabeledPatchDataset(X_val, y_val_bin), batch_size=batch_size, shuffle=False)

    model = PatchClassifier(num_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss, epochs_without_improvement, best_state = float("inf"), 0, None
    for epoch in range(max_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                val_losses.append(criterion(model(x), y).item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        print(f"[Stage1] Fold {fold_idx} | Epoch {epoch+1}: val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[Stage1] Fold {fold_idx}: early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


def train_stage2_grader_fold(
    fold, id_to_basename, fold_idx,
    batch_size=64, max_epochs=200, patience=20,  # B4/B6 needed a few more epochs to recover past their spike
    lr=1e-4, warmup_epochs=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    train_ids, val_ids = make_stage1_train_val_split(fold["train_ids"])
    X_train, y_train, _ = build_patch_dataset(train_ids, id_to_basename, DESPECKLED_DIR)
    X_val, y_val, _ = build_patch_dataset(val_ids, id_to_basename, DESPECKLED_DIR)

    pos_train_mask, pos_val_mask = y_train != 0, y_val != 0
    X_train, y_train = X_train[pos_train_mask], y_train[pos_train_mask] - 1
    X_val, y_val = X_val[pos_val_mask], y_val[pos_val_mask] - 1

    # Inverse-frequency class weighting, computed per fold from that fold's
    # own training distribution — imbalance severity varies by fold (Cat1 is
    # especially thin in some), so a single global weighting would be wrong.
    class_counts = np.bincount(y_train, minlength=4)
    class_weights = torch.tensor(1.0 / np.sqrt(np.maximum(class_counts, 1)), dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum() * 4
    print(f"[Stage2] Fold {fold_idx} class counts: {class_counts}, weights: {class_weights.cpu().numpy().round(3)}")

    train_loader = DataLoader(LabeledPatchDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(LabeledPatchDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = PatchClassifier(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss, epochs_without_improvement, best_state = float("inf"), 0, None
    for epoch in range(max_epochs):
        # Linear LR warm-up over the first few epochs — a training-loss spike
        # shows up consistently across every fold in the first ~5-9 epochs,
        # right when the model is least stable; ramping up from a lower LR
        # gives training a gentler on-ramp instead of hitting full LR before
        # the model has any real footing.
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for g in optimizer.param_groups:
                g["lr"] = warmup_lr

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_losses.append(criterion(model(x), y).item())
        val_loss = np.mean(val_losses)
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)

        print(f"[Stage2] Fold {fold_idx} | Epoch {epoch+1}: val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[Stage2] Fold {fold_idx}: early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


def train_resnet_grader_fold(
    fold, id_to_basename, fold_idx,
    batch_size=32, max_epochs=100, patience=20,
    lr=1e-4, warmup_epochs=5,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    train_ids, val_ids = make_stage1_train_val_split(fold["train_ids"])
    X_train, y_train, _ = build_patch_dataset(train_ids, id_to_basename, DESPECKLED_DIR)
    X_val, y_val, _ = build_patch_dataset(val_ids, id_to_basename, DESPECKLED_DIR)

    pos_train_mask, pos_val_mask = y_train != 0, y_val != 0
    X_train, y_train = X_train[pos_train_mask], y_train[pos_train_mask] - 1
    X_val, y_val = X_val[pos_val_mask], y_val[pos_val_mask] - 1

    class_counts = np.bincount(y_train, minlength=4)
    class_weights = torch.tensor(1.0 / np.sqrt(np.maximum(class_counts, 1)), dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum() * 4

    train_loader = DataLoader(ResNetPatchDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ResNetPatchDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = build_resnet_grader(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss, epochs_without_improvement, best_state = float("inf"), 0, None
    for epoch in range(max_epochs):
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for g in optimizer.param_groups:
                g["lr"] = warmup_lr

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_losses.append(criterion(model(x), y).item())
        val_loss = np.mean(val_losses)
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)

        print(f"[ResNet] Fold {fold_idx} | Epoch {epoch+1}: val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[ResNet] Fold {fold_idx}: early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


def main(train_resnet=True):
    os.makedirs(STAGE1_CKPT_DIR, exist_ok=True)
    os.makedirs(STAGE2_CKPT_DIR, exist_ok=True)
    if train_resnet:
        os.makedirs(RESNET_CKPT_DIR, exist_ok=True)

    id_to_basename = parse_ground_truths_table(os.path.join(TABLES_DIR, "Images_GroundTruths_Table"))
    hom38_folds = load_hom38_folds(TABLES_DIR)

    for fold in hom38_folds:
        fold_idx = fold["subexp"]
        print(f"\n{'='*20} Fold {fold_idx} {'='*20}")

        stage1_model, _ = train_stage1_detector_fold(fold, id_to_basename, fold_idx)
        torch.save(stage1_model.state_dict(), os.path.join(STAGE1_CKPT_DIR, f"stage1_fold{fold_idx}.pt"))

        stage2_model, _ = train_stage2_grader_fold(fold, id_to_basename, fold_idx)
        torch.save(stage2_model.state_dict(), os.path.join(STAGE2_CKPT_DIR, f"stage2_fold{fold_idx}.pt"))

        if train_resnet:
            resnet_model, _ = train_resnet_grader_fold(fold, id_to_basename, fold_idx)
            torch.save(resnet_model.state_dict(), os.path.join(RESNET_CKPT_DIR, f"resnet_fold{fold_idx}.pt"))

    print("\nAll folds trained. Run `python -m methods.supervised.evaluate` for cross-fold metrics.")


if __name__ == "__main__":
    main()
