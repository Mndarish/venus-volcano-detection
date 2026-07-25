"""
Training loop for the Deep SVDD method.

STATUS: investigated and ruled out (see README.md in this folder). Kept
runnable for reproducibility of the negative result — not part of the
shipped pipeline (methods/supervised is shipped).

Deep SVDD (Ruff et al., 2018) sidesteps pixel-reconstruction anomaly
scoring entirely: there is no decoder at inference, just an encoder that
maps background patches toward a single fixed center `c`. Anomaly score =
distance from `c`.

Two things matter for correctness here, not just performance:
- No bias terms anywhere in the encoder (including BatchNorm's affine
  params) — otherwise the network can trivially minimize the loss by
  mapping every input to `c` regardless of content (hypersphere collapse).
- `c` is fixed, not learned. It's computed once from a forward pass over
  training data, then frozen, with near-zero coordinates nudged away from
  zero (a coordinate of exactly zero gives the network a free way to
  "cheat" that dimension).

A plain Deep SVDD from random init collapses (both bg and volcano
distances converge toward ~0) since nothing in the objective rewards
staying input-sensitive. The fix: pretrain the encoder as part of a
standard autoencoder first, then transplant only the encoder weights into
the bias-free SVDD network before fine-tuning.

Usage:
    python -m methods.deep_svdd.train
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
    build_stage1_dataset,
    load_hom38_folds,
    make_stage1_train_val_split,
    parse_ground_truths_table,
)
from .datasets import PatchDataset
from .model import DeepSVDDEncoder, PretrainAutoencoder, transplant_encoder_weights

SVDD_CKPT_DIR = os.path.join(CHECKPOINTS_DIR, "deep_svdd", "stage1")


def initialize_center(model, data_loader, embedding_dim, device, eps=0.1):
    """Fixed center c = mean embedding over an initial forward pass through
    the training data. Coordinates too close to zero are nudged away from
    zero (either direction) — a center coordinate of exactly zero lets the
    network trivially satisfy that dimension by always outputting zero
    there, a free degenerate shortcut."""
    model.eval()
    n_samples = 0
    c = torch.zeros(embedding_dim, device=device)
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            z = model(batch)
            c += z.sum(dim=0)
            n_samples += z.shape[0]
    c /= n_samples

    c[(c.abs() < eps) & (c >= 0)] = eps
    c[(c.abs() < eps) & (c < 0)] = -eps
    return c


def pretrain_encoder(
    fold, id_to_basename, embedding_dim=32,
    batch_size=64, max_epochs=100, patience=10, lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    stage1_train_ids, val_ids = make_stage1_train_val_split(fold["train_ids"])
    X_train, _ = build_stage1_dataset(stage1_train_ids, id_to_basename, DESPECKLED_DIR)
    X_val, _ = build_stage1_dataset(val_ids, id_to_basename, DESPECKLED_DIR)

    train_loader = DataLoader(PatchDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PatchDataset(X_val), batch_size=batch_size, shuffle=False)

    ae = PretrainAutoencoder(embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss, epochs_without_improvement, best_state = float("inf"), 0, None
    for epoch in range(max_epochs):
        ae.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = ae(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

        ae.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, _ = ae(batch)
                val_losses.append(criterion(recon, batch).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in ae.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    ae.load_state_dict(best_state)
    print(f"  Pretrain AE best val_loss: {best_val_loss:.5f}")
    return ae


def train_deep_svdd_fold(
    fold, id_to_basename, fold_idx,
    embedding_dim=32, weight_decay=1e-5,
    batch_size=64, max_epochs=200, patience=10,
    lr=1e-4,  # smaller lr for fine-tuning, since we start from a good init
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"  Pretraining encoder (AE) for fold {fold_idx}...")
    pretrained_ae = pretrain_encoder(fold, id_to_basename, embedding_dim=embedding_dim, device=device)

    stage1_train_ids, val_ids = make_stage1_train_val_split(fold["train_ids"])
    X_train, _ = build_stage1_dataset(stage1_train_ids, id_to_basename, DESPECKLED_DIR)
    X_val, _ = build_stage1_dataset(val_ids, id_to_basename, DESPECKLED_DIR)

    train_loader = DataLoader(PatchDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PatchDataset(X_val), batch_size=batch_size, shuffle=False)
    center_loader = DataLoader(PatchDataset(X_train), batch_size=batch_size, shuffle=False)

    model = DeepSVDDEncoder(embedding_dim=embedding_dim).to(device)
    model = transplant_encoder_weights(pretrained_ae, model)

    # weight_decay doubles as collapse prevention -- pulls weights toward
    # zero, discouraging drift back toward the degenerate
    # all-inputs-map-to-c solution
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    c = initialize_center(model, center_loader, embedding_dim, device)

    best_val_loss, epochs_without_improvement, best_state = float("inf"), 0, None
    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = model(batch)
            loss = ((z - c) ** 2).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                z = model(batch)
                val_losses.append(((z - c) ** 2).sum(dim=1).mean().item())

        train_loss, val_loss = np.mean(train_losses), np.mean(val_losses)
        scheduler.step(val_loss)
        print(f"Fold {fold_idx} | Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Fold {fold_idx}: early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.6f})")
                break

    model.load_state_dict(best_state)
    return model, c, best_val_loss


def main():
    os.makedirs(SVDD_CKPT_DIR, exist_ok=True)
    id_to_basename = parse_ground_truths_table(os.path.join(TABLES_DIR, "Images_GroundTruths_Table"))
    hom38_folds = load_hom38_folds(TABLES_DIR)

    fold_results, fold_centers = {}, {}
    for fold_idx, fold in enumerate(hom38_folds):
        print(f"\n=== Training Deep SVDD — Fold {fold['subexp']} ===")
        model, c, best_val_loss = train_deep_svdd_fold(fold, id_to_basename, fold_idx)

        ckpt_path = os.path.join(SVDD_CKPT_DIR, f"svdd_fold{fold['subexp']}.pt")
        torch.save({"model_state": model.state_dict(), "center": c.cpu()}, ckpt_path)
        fold_results[fold["subexp"]] = best_val_loss
        fold_centers[fold["subexp"]] = c
        print(f"Fold {fold['subexp']} done. Best val_loss={best_val_loss:.4f}. Saved to {ckpt_path}")

    print("\nAll folds trained.")
    for k, v in fold_results.items():
        print(f"  Fold {k}: best val_loss = {v:.4f}")


if __name__ == "__main__":
    main()
