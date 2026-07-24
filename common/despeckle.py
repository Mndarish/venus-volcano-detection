"""
Lee filter despeckling.

Multiplicative noise model: I_observed = I_true * N. Cu2 is the pure-noise
baseline. It's estimated empirically from flat, volcano-free patches rather
than assumed, since whole-image and percentile estimates were both biased
(too high / overcorrected respectively). See scripts/derive_cu2.py for the
derivation; this module uses the locked result (CU2_FINAL).
"""
import glob
import os

import numpy as np
from scipy.ndimage import uniform_filter
from tqdm import tqdm

from .config import CU2_FINAL, WINDOW_SIZE_FINAL
from .data_loading import load_lxyr_labels, load_raw_image


def lee_filter(image, window_size=WINDOW_SIZE_FINAL, Cu2=CU2_FINAL):
    img = image.astype(np.float64)
    local_mean = uniform_filter(img, size=window_size)
    local_sq_mean = uniform_filter(img**2, size=window_size)
    local_variance = np.maximum(local_sq_mean - local_mean**2, 0)

    W = np.clip(local_variance / (local_variance + Cu2 * local_mean**2 + 1e-8), 0, 1)
    denoised = local_mean + W * (img - local_mean)
    return denoised, W


def get_volcano_mask_exclusions(basename, gt_dir):
    """(x, y, r) circles to avoid when hunting for flat/background patches."""
    data = load_lxyr_labels(basename, gt_dir)
    return [(x, y, r) for _, x, y, r in data]


def despeckle_all_tiles(images_dir, despeckled_dir):
    """Batch despeckle every .sdt tile in images_dir, caching results as
    .npy in despeckled_dir. Skips tiles already processed."""
    os.makedirs(despeckled_dir, exist_ok=True)
    sdt_files = sorted(glob.glob(os.path.join(images_dir, "*.sdt")))

    failed, skipped, processed = [], [], 0
    for sdt_path in tqdm(sdt_files, desc="Despeckling"):
        basename = os.path.splitext(os.path.basename(sdt_path))[0]
        out_path = os.path.join(despeckled_dir, f"{basename}.npy")

        if os.path.exists(out_path):
            skipped.append(basename)
            continue
        try:
            raw_image = load_raw_image(sdt_path)
        except ValueError as e:
            failed.append((basename, str(e)))
            continue

        denoised, _ = lee_filter(raw_image)  # CU2_FINAL / WINDOW_SIZE_FINAL baked in as defaults
        np.save(out_path, denoised.astype(np.float32))
        processed += 1

    print(f"Processed: {processed} | Skipped (cached): {len(skipped)} | Failed: {len(failed)}")
    if failed:
        for basename, err in failed:
            print(f"  {basename}: {err}")
    return {"processed": processed, "skipped": skipped, "failed": failed}


def load_despeckled_batch(image_ids, id_to_basename, despeckled_dir):
    images = {}
    for img_id in image_ids:
        basename = id_to_basename.get(img_id)
        npy_path = os.path.join(despeckled_dir, f"{basename}.npy") if basename else None
        if npy_path and os.path.exists(npy_path):
            images[img_id] = np.load(npy_path)
    return images
