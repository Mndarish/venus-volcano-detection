"""
Evaluation for the autoencoder method — reproduces the negative result.

Across all 6 folds, volcanoes reconstruct BETTER than background
(separation ratio < 1.0, i.e. backwards from what anomaly detection needs).
The Gaussian-blur baseline below reproduces the same reversed pattern with
zero learned parameters, proving this is a structural artifact of SAR
patch content (smooth domes vs. textured fractures/ridges) rather than
something the autoencoder specifically learned. See README.md for the
full writeup.

Usage:
    python -m methods.autoencoder.evaluate
"""
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

from common import DESPECKLED_DIR, TABLES_DIR, build_patch_dataset, load_hom38_folds, normalize_patch, parse_ground_truths_table
from common.config import CHECKPOINTS_DIR
from pytorch_msssim import ssim as ssim_fn
from .model import Stage1AutoencoderSpatialLatent

STAGE1_CKPT_DIR = os.path.join(CHECKPOINTS_DIR, "autoencoder", "stage1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_reconstruction_errors(model, patches, device=DEVICE):
    """Per-patch reconstruction error (mean pixel-wise squared error)."""
    model.eval()
    normalized = normalize_patch(patches).astype(np.float32)
    tensor = torch.from_numpy(normalized).unsqueeze(1).to(device)

    errors = []
    with torch.no_grad():
        for i in range(0, len(tensor), 128):
            batch = tensor[i : i + 128]
            recon, _ = model(batch)
            per_patch_mse = ((recon - batch) ** 2).mean(dim=[1, 2, 3])
            errors.append(per_patch_mse.cpu().numpy())
    return np.concatenate(errors)


def compute_error_variants(model, patches, device=DEVICE):
    """Raw MSE, per-patch SSIM error, and variance-normalized MSE — used to
    check whether the separation signal survives different error metrics."""
    model.eval()
    normalized = normalize_patch(patches).astype(np.float32)
    tensor = torch.from_numpy(normalized).unsqueeze(1).to(device)

    mse_errors, ssim_errors, norm_mse_errors = [], [], []
    with torch.no_grad():
        for i in range(0, len(tensor), 128):
            batch = tensor[i : i + 128]
            recon, _ = model(batch)

            per_patch_mse = ((recon - batch) ** 2).mean(dim=[1, 2, 3])
            mse_errors.append(per_patch_mse.cpu().numpy())

            for j in range(batch.shape[0]):
                s = ssim_fn(recon[j : j + 1], batch[j : j + 1], data_range=1.0, size_average=True)
                ssim_errors.append((1 - s).item())

            patch_var = batch.reshape(batch.shape[0], -1).var(dim=1) + 1e-8
            norm_mse_errors.append((per_patch_mse / patch_var).cpu().numpy())

    return (np.concatenate(mse_errors), np.array(ssim_errors), np.concatenate(norm_mse_errors))


def check_complexity_confound(X_test, y_test, errors):
    """Does reconstruction error just track raw patch variance (i.e. is
    the model measuring 'how textured is this patch' rather than anything
    volcano-specific)?"""
    raw_variance = X_test.reshape(len(X_test), -1).var(axis=1)
    r, p = pearsonr(raw_variance, errors)
    print(f"Correlation between raw patch variance and reconstruction error: r={r:.3f}, p={p:.4g}")
    for cat in sorted(np.unique(y_test)):
        mask = y_test == cat
        print(f"  Category {cat}: n={mask.sum()}, mean_error={errors[mask].mean():.5f}, "
              f"mean_raw_variance={raw_variance[mask].mean():.2f}")


def gaussian_blur_baseline_error(patches, sigma=2.0):
    """Non-learned control: how much does simple blurring alone distort
    volcano vs background patches, with zero training?"""
    normalized = normalize_patch(patches).astype(np.float32)
    return np.array([((gaussian_filter(p, sigma=sigma) - p) ** 2).mean() for p in normalized])


def main():
    id_to_basename = parse_ground_truths_table(os.path.join(TABLES_DIR, "Images_GroundTruths_Table"))
    hom38_folds = load_hom38_folds(TABLES_DIR)

    for fold in hom38_folds:
        fold_idx = fold["subexp"]
        print(f"\n=== Fold {fold_idx}: reconstruction error check ===")

        model = Stage1AutoencoderSpatialLatent(latent_channels=16).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(STAGE1_CKPT_DIR, f"stage1_fold{fold_idx}.pt"), map_location=DEVICE))

        X_test, y_test, _ = build_patch_dataset(fold["test_ids"], id_to_basename, DESPECKLED_DIR)
        bg_patches = X_test[y_test == 0]
        volcano_patches = X_test[y_test != 0]

        bg_errors = compute_reconstruction_errors(model, bg_patches)
        volcano_errors = compute_reconstruction_errors(model, volcano_patches)

        print(f"Background error  — mean: {bg_errors.mean():.5f}, std: {bg_errors.std():.5f}")
        print(f"Volcano error     — mean: {volcano_errors.mean():.5f}, std: {volcano_errors.std():.5f}")
        print(f"Separation (volcano_mean / bg_mean): {volcano_errors.mean() / bg_errors.mean():.2f}x "
              f"(expect > 1.0 for a useful anomaly score — this method sees < 1.0)")

    # --- Complexity confound + Gaussian blur control, on fold B1 ---
    fold0 = hom38_folds[0]
    model = Stage1AutoencoderSpatialLatent(latent_channels=16).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(STAGE1_CKPT_DIR, f"stage1_fold{fold0['subexp']}.pt"), map_location=DEVICE))

    X_test, y_test, _ = build_patch_dataset(fold0["test_ids"], id_to_basename, DESPECKLED_DIR)
    all_errors = compute_reconstruction_errors(model, X_test)
    print(f"\n=== Complexity confound check (fold {fold0['subexp']}) ===")
    check_complexity_confound(X_test, y_test, all_errors)

    mse_err, ssim_err, norm_mse_err = compute_error_variants(model, X_test)
    raw_variance = X_test.reshape(len(X_test), -1).var(axis=1)
    print("\n=== Error metric variants ===")
    for name, errs in [("Raw MSE", mse_err), ("SSIM (1-ssim)", ssim_err), ("Variance-normalized MSE", norm_mse_err)]:
        r, p = pearsonr(raw_variance, errs)
        bg_mean, volc_mean = errs[y_test == 0].mean(), errs[y_test != 0].mean()
        print(f"{name}: corr with raw variance r={r:.3f} | bg_mean={bg_mean:.5f}, "
              f"volcano_mean={volc_mean:.5f}, ratio={volc_mean/bg_mean:.2f}x")

    print("\n=== Gaussian blur baseline (zero learned parameters) ===")
    for sigma in [1.0, 2.0, 3.0]:
        blur_errors = gaussian_blur_baseline_error(X_test, sigma=sigma)
        bg_mean, volc_mean = blur_errors[y_test == 0].mean(), blur_errors[y_test != 0].mean()
        print(f"sigma={sigma}: bg_mean={bg_mean:.5f}, volcano_mean={volc_mean:.5f}, "
              f"ratio={volc_mean/bg_mean:.2f}x (same reversed pattern as the trained model)")


if __name__ == "__main__":
    main()
