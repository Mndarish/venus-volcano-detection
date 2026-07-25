"""
Evaluation for the Deep SVDD method — reproduces the negative result.

Once collapse was resolved via AE-pretrain-then-transplant, separation
still ran backwards across all 6 folds (background sits closer to the
volcano cluster than typical background does, not farther). Root cause:
background is not a coherent "normal" class here — it's a diverse catch-
all (fractures, ridges, tessera, plains) — while volcanoes are visually
homogeneous. Any clustering approach will naturally treat the homogeneous
class as tight and the diverse catch-all as spread out, regardless of
which one it trained on: the anomaly-detection premise (normal = coherent,
anomaly = rare deviation) is inverted in this dataset. See README.md.

Usage:
    python -m methods.deep_svdd.evaluate
"""
import os

import numpy as np
import torch

from common import DESPECKLED_DIR, TABLES_DIR, build_patch_dataset, load_hom38_folds, normalize_patch, parse_ground_truths_table
from common.config import CHECKPOINTS_DIR
from .model import DeepSVDDEncoder

SVDD_CKPT_DIR = os.path.join(CHECKPOINTS_DIR, "deep_svdd", "stage1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_svdd_distances(model, c, patches, device=DEVICE):
    model.eval()
    normalized = normalize_patch(patches).astype(np.float32)
    tensor = torch.from_numpy(normalized).unsqueeze(1).to(device)

    distances = []
    with torch.no_grad():
        for i in range(0, len(tensor), 128):
            batch = tensor[i : i + 128]
            z = model(batch)
            d = ((z - c) ** 2).sum(dim=1)
            distances.append(d.cpu().numpy())
    return np.concatenate(distances)


def main():
    id_to_basename = parse_ground_truths_table(os.path.join(TABLES_DIR, "Images_GroundTruths_Table"))
    hom38_folds = load_hom38_folds(TABLES_DIR)

    for fold in hom38_folds:
        fold_idx = fold["subexp"]
        print(f"\n=== Fold {fold_idx}: Deep SVDD distance check ===")

        ckpt = torch.load(os.path.join(SVDD_CKPT_DIR, f"svdd_fold{fold_idx}.pt"), map_location=DEVICE)
        model = DeepSVDDEncoder(embedding_dim=32).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        c = ckpt["center"].to(DEVICE)

        X_test, y_test, _ = build_patch_dataset(fold["test_ids"], id_to_basename, DESPECKLED_DIR)
        bg_patches = X_test[y_test == 0]
        volcano_patches = X_test[y_test != 0]

        bg_dist = compute_svdd_distances(model, c, bg_patches)
        volcano_dist = compute_svdd_distances(model, c, volcano_patches)

        print(f"Background distance — mean: {bg_dist.mean():.8f}, std: {bg_dist.std():.8f}")
        print(f"Volcano distance    — mean: {volcano_dist.mean():.8f}, std: {volcano_dist.std():.8f}")
        print(f"Separation (volcano_mean / bg_mean): {volcano_dist.mean() / bg_dist.mean():.2f}x "
              f"(expect > 1.0 for a useful anomaly score — this method sees < 1.0 in most folds)")


if __name__ == "__main__":
    main()
