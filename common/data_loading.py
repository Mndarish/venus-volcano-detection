"""
Raw SAR tile loading and ground-truth label parsing.

`.sdt` files are raw uint8, 1024x1024, no header.
`.lxyr` files hold [label, x, y, r] rows per labeled volcano (Cat1-4).
"""
import os
import numpy as np

from .config import GT_DIR


def load_raw_image(sdt_path, image_size=1024):
    """Load a raw .sdt SAR tile as a (image_size, image_size) uint8 array."""
    with open(sdt_path, "rb") as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
    expected_bytes = image_size * image_size
    if raw_data.size != expected_bytes:
        raise ValueError(f"Size mismatch: got {raw_data.size}, expected {expected_bytes}")
    return raw_data.reshape((image_size, image_size))


def load_lxyr_labels(basename, gt_dir=GT_DIR):
    """Returns (N, 4) array of [label, x, y, r], or empty (0, 4) if unlabeled."""
    path = os.path.join(gt_dir, f"{basename}.lxyr")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return np.empty((0, 4))
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    return data


def parse_ground_truths_table(path):
    """Maps ImageID -> 'img{ID}', matching on-disk filenames.

    Note: Images_GroundTruths_Table embeds original JPL `ff*`-style source
    paths that do NOT match the on-disk filenames. The fix is to map
    ImageID -> "img{ID}" directly rather than trusting the embedded path.
    """
    id_to_basename = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 3 or not parts[0].isdigit():
                continue
            img_id = int(parts[0])
            id_to_basename[img_id] = f"img{img_id}"
    return id_to_basename
