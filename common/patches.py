"""
Patch extraction and per-fold dataset construction.

Positive patches use an adaptive crop scaled to each volcano's radius
(crop_size = clip(r * 4, 32, 200)), then resized to a fixed network input
size. Boundary-clipping is handled via reflect-padding (see
extract_positive_patches) rather than dropping edge volcanoes — the
padding fix recovers all 1,520 labeled volcanoes; a fixed-crop, no-pad
version would silently drop ~a few percent, biased toward larger radii.
"""
import os

import numpy as np
from skimage.transform import resize

from .config import (
    CROP_MULTIPLIER,
    GT_DIR,
    MAX_CROP,
    MIN_CROP,
    NETWORK_INPUT_SIZE,
    STAGE1_NEGATIVES_PER_IMAGE,
)
from .data_loading import load_lxyr_labels
from .despeckle import get_volcano_mask_exclusions


def normalize_patch(patch):
    """Scale despeckled patch to [0,1] using the known 8-bit ceiling (255),
    not the empirical max of any particular sample — background patches
    systematically miss the brightest pixels (volcanoes), so normalizing
    against the true physical ceiling avoids clipping brighter unseen
    pixels at inference time."""
    return np.clip(patch, 0, 255) / 255.0


def patch_overlaps_volcano(px, py, patch_size, volcano_circles, margin=10):
    patch_cx, patch_cy = px + patch_size / 2, py + patch_size / 2
    half_diag = (patch_size / 2) * np.sqrt(2)
    return any(
        np.hypot(patch_cx - vx, patch_cy - vy) < (vr + margin + half_diag)
        for vx, vy, vr in volcano_circles
    )


def extract_positive_patches(despeckled_img, basename, gt_dir=GT_DIR):
    """Returns list of (patch, category, x, y, r) for each labeled volcano.
    Pads the image so no volcano is dropped due to boundary clipping."""
    data = load_lxyr_labels(basename, gt_dir)
    h, w = despeckled_img.shape

    pad = MAX_CROP // 2  # enough padding for the largest possible crop
    padded_img = np.pad(despeckled_img, pad, mode="reflect")

    patches = []
    for label, x, y, r in data:
        crop_size = int(np.clip(r * CROP_MULTIPLIER, MIN_CROP, MAX_CROP))
        half = crop_size // 2
        cx, cy = int(round(x)) + pad, int(round(y)) + pad
        x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
        crop = padded_img[y0:y1, x0:x1]
        patch = resize(
            crop, (NETWORK_INPUT_SIZE, NETWORK_INPUT_SIZE), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        patches.append((patch, int(label), x, y, r))
    return patches


def extract_negative_patches(
    despeckled_img,
    basename,
    gt_dir=GT_DIR,
    patch_size=NETWORK_INPUT_SIZE,
    num_negatives=10,
    stride=32,
    margin=10,
    seed=None,
):
    """Grid-sample candidate patches, exclude any overlapping a volcano + margin, randomly select."""
    volcano_circles = get_volcano_mask_exclusions(basename, gt_dir)
    h, w = despeckled_img.shape
    candidates = [
        (px, py)
        for py in range(0, h - patch_size, stride)
        for px in range(0, w - patch_size, stride)
        if not patch_overlaps_volcano(px, py, patch_size, volcano_circles, margin=margin)
    ]

    rng = np.random.default_rng(seed)
    if len(candidates) > num_negatives:
        idx = rng.choice(len(candidates), size=num_negatives, replace=False)
        chosen = [candidates[i] for i in idx]
    else:
        chosen = candidates

    patches = []
    for px, py in chosen:
        patch = despeckled_img[py : py + patch_size, px : px + patch_size]
        patches.append((patch, 0, px + patch_size // 2, py + patch_size // 2, None))  # category 0 = background
    return patches


def build_patch_dataset(image_ids, id_to_basename, despeckled_dir, seed=42):
    """Stage 2 / supervised dataset: 1:1 positive:negative ratio per image."""
    X, y, meta = [], [], []
    for img_id in image_ids:
        basename = id_to_basename.get(img_id)
        npy_path = os.path.join(despeckled_dir, f"{basename}.npy") if basename else None
        if not npy_path or not os.path.exists(npy_path):
            print(f"Missing despeckled file for basename={basename} (ID {img_id})")
            continue
        img = np.load(npy_path)
        pos_patches = extract_positive_patches(img, basename)

        num_negatives = len(pos_patches)
        neg_patches = extract_negative_patches(img, basename, num_negatives=num_negatives, seed=seed + img_id)

        for patch, cat, x, y_, r in pos_patches + neg_patches:
            X.append(patch)
            y.append(cat)
            meta.append({"img_id": img_id, "basename": basename, "x": x, "y": y_, "r": r})
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), meta


def build_stage1_dataset(image_ids, id_to_basename, despeckled_dir, seed=42):
    """Background-only patches for the unsupervised Stage 1 methods
    (autoencoder, deep_svdd). Decoupled from positive counts — sized for
    max terrain diversity, not label balance. Respects the same fold
    boundaries as the supervised dataset (pass fold['train_ids'] /
    fold['test_ids'])."""
    X, meta = [], []
    for img_id in image_ids:
        basename = id_to_basename.get(img_id)
        npy_path = os.path.join(despeckled_dir, f"{basename}.npy") if basename else None
        if not npy_path or not os.path.exists(npy_path):
            print(f"Missing despeckled file for basename={basename} (ID {img_id})")
            continue
        img = np.load(npy_path)
        neg_patches = extract_negative_patches(
            img, basename, num_negatives=STAGE1_NEGATIVES_PER_IMAGE, seed=seed + img_id
        )
        for patch, cat, x, y_, r in neg_patches:
            X.append(patch)
            meta.append({"img_id": img_id, "basename": basename, "x": x, "y": y_})
    return np.array(X, dtype=np.float32), meta
