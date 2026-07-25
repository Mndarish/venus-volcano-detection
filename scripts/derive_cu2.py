"""
Derivation of Cu2 for the Lee filter — kept for reproducibility, not
needed for normal use (common.despeckle.lee_filter already has the
locked result baked in as CU2_FINAL = 0.01086).

Cu2 is estimated empirically from flat, volcano-free patches rather than
assumed, since whole-image and percentile estimates were both biased
(too high / overcorrected respectively). Final value = median across 24
flat patches (top-3 flattest per image) from 8 sample images spanning the
ID range.

Usage:
    python scripts/derive_cu2.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import IMAGES_DIR, load_raw_image
from common.despeckle import get_volcano_mask_exclusions
from common.config import GT_DIR


def patch_overlaps_volcano(px, py, patch_size, volcano_circles, margin=10):
    patch_cx, patch_cy = px + patch_size / 2, py + patch_size / 2
    half_diag = (patch_size / 2) * np.sqrt(2)
    return any(
        np.hypot(patch_cx - vx, patch_cy - vy) < (vr + margin + half_diag)
        for vx, vy, vr in volcano_circles
    )


def find_flat_patches(image, basename, patch_size=64, stride=32, top_k=3, gt_dir=GT_DIR, min_mean=5.0):
    """Grid-search patches, exclude volcano regions, rank by lowest variance."""
    volcano_circles = get_volcano_mask_exclusions(basename, gt_dir)
    h, w = image.shape
    candidates = []
    for py in range(0, h - patch_size, stride):
        for px in range(0, w - patch_size, stride):
            if patch_overlaps_volcano(px, py, patch_size, volcano_circles):
                continue
            patch = image[py : py + patch_size, px : px + patch_size].astype(np.float64)
            variance, mean_val = np.var(patch), np.mean(patch)
            if mean_val < min_mean or variance == 0:
                continue
            candidates.append((variance, px, py, patch))
    candidates.sort(key=lambda c: c[0])
    return candidates[:top_k]


def compute_cu2(patch):
    patch = patch.astype(np.float64)
    return np.var(patch) / (np.mean(patch) ** 2 + 1e-8)


def main():
    sample_ids = [1, 20, 40, 60, 80, 100, 120, 134]
    all_cu2_estimates = []
    for img_id in sample_ids:
        basename = f"img{img_id}"
        sdt_path = os.path.join(IMAGES_DIR, f"{basename}.sdt")
        if not os.path.exists(sdt_path):
            continue
        img = load_raw_image(sdt_path)
        flat_patches = find_flat_patches(img, basename, patch_size=64, top_k=3)
        all_cu2_estimates.extend(compute_cu2(patch) for _, _, _, patch in flat_patches)

    print(f"Cu2 across {len(all_cu2_estimates)} flat patches from {len(sample_ids)} images:")
    print(
        f"  median={np.median(all_cu2_estimates):.5f}  mean={np.mean(all_cu2_estimates):.5f}  "
        f"std={np.std(all_cu2_estimates):.5f}"
    )
    print("\nLocked production value: CU2_FINAL = 0.01086 (see common/config.py)")


if __name__ == "__main__":
    main()
