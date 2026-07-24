"""
Training loop for the autoencoder method.

STATUS: investigated and ruled out (see README.md in this folder). Kept
runnable for reproducibility of the negative result — not part of the
shipped pipeline (methods/supervised is shipped).

NOTE ON A FIXED BUG: the original notebook's training loop called an
undefined `Stage1Autoencoder(latent_dim=64)`. The only architecture
actually defined anywhere in that notebook is
`Stage1AutoencoderSpatialLatent(latent_channels=...)`, which is what this
file uses. If you have a from-scratch run that produced different
numbers, it was working from a different, unrecorded class definition —
treat any numbers reproduced from this file as coming from the spatial-
latent architecture specifically.

Usage:
    python -m methods.autoencoder.train
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import (
    CHECKPOINTS_DIR,
    DESPECKLED_DIR,
    TABLES_DIR,
    build_stage1_dataset,
    load_hom38_folds,
    make_stage1_train_val_split,
    parse_ground_truths_table,
)
from .datasets import PatchDataset
from .model import HybridReconstructionLoss, Stage1AutoencoderSpatialLatent

STAGE1_CKPT_DIR = os.path.join(CHECKPOINTS_DIR, "autoencoder", "stage1")


def train_stage1_fold(
    fold, id_to_basename, fold_idx,
    latent_channels=16, alpha=0.2,
    batch_size=64, max_epochs=200, patience=10,
    lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu",
):
    # --- Image-level train/val split (never split by patch — see common.folds) ---
    stage1_train_ids, val_ids = make_stage1_train_val_split(fold["train_ids"])
    X_train, _ = build_stage1_dataset(stage1_train_ids, id_to_basename, DESPECKLED_DIR)
    X_val, _ = build_stage1_dataset(val_ids, id_to_basename, DESPECKLED_DIR)

    train_loader = DataLoader(PatchDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PatchDataset(X_val), batch_size=batch_size, shuffle=False)

    model = Stage1AutoencoderSpatialLatent(latent_channels=latent_channels).to(device)
    criterion = HybridReconstructionLoss(alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, _ = model(batch)
                loss = criterion(recon, batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        print(f"Fold {fold_idx} | Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Fold {fold_idx}: early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    return model, best_val_loss


def main():
    os.makedirs(STAGE1_CKPT_DIR, exist_ok=True)
    id_to_basename = parse_ground_truths_table(os.path.join(TABLES_DIR, "Images_GroundTruths_Table"))
    hom38_folds = load_hom38_folds(TABLES_DIR)

    fold_results = {}
    for fold_idx, fold in enumerate(hom38_folds):
        print(f"\n=== Training Stage 1 autoencoder — Fold {fold['subexp']} ===")
        model, best_val_loss = train_stage1_fold(fold, id_to_basename, fold_idx)

        ckpt_path = os.path.join(STAGE1_CKPT_DIR, f"stage1_fold{fold['subexp']}.pt")
        torch.save(model.state_dict(), ckpt_path)
        fold_results[fold["subexp"]] = best_val_loss
        print(f"Fold {fold['subexp']} done. Best val_loss={best_val_loss:.4f}. Saved to {ckpt_path}")

    print("\nAll folds trained.")
    for k, v in fold_results.items():
        print(f"  Fold {k}: best val_loss = {v:.4f}")


if __name__ == "__main__":
    main()
