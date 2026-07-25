"""
Run this after cloning the repo (and after data/ is in place) to verify
the whole pipeline end to end: data directories exist, HOM38 folds parse
and pass overlap/coverage checks, despeckling recovers all 134 tiles, and
patch extraction recovers all 1,520 labeled volcanoes with a clean 1:1
positive:negative ratio in every fold.

Usage:
    python scripts/verify_setup.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import (
    BASE_PATH,
    DESPECKLED_DIR,
    GT_DIR,
    IMAGES_DIR,
    TABLES_DIR,
    build_patch_dataset,
    despeckle_all_tiles,
    load_hom38_folds,
    parse_ground_truths_table,
)


def main():
    for d in [IMAGES_DIR, GT_DIR, TABLES_DIR]:
        assert os.path.exists(d), f"Missing expected directory: {d}"
    print(f"All data directories found under {BASE_PATH}")

    id_to_basename = parse_ground_truths_table(os.path.join(TABLES_DIR, "Images_GroundTruths_Table"))
    print(f"Parsed {len(id_to_basename)} image ID -> basename mappings")

    hom38_folds = load_hom38_folds(TABLES_DIR)
    print(f"Parsed {len(hom38_folds)} HOM38 folds — all pass overlap/coverage checks")
    for fold in hom38_folds:
        print(f"  {fold['subexp']}: train={len(fold['train_ids'])}, test={len(fold['test_ids'])}")

    result = despeckle_all_tiles(IMAGES_DIR, DESPECKLED_DIR)
    if result["failed"]:
        raise RuntimeError(f"{len(result['failed'])} tiles failed to despeckle")

    total_positives = 0
    fold_totals = []
    for fold in hom38_folds:
        X_train, y_train, _ = build_patch_dataset(fold["train_ids"], id_to_basename, DESPECKLED_DIR)
        X_test, y_test, _ = build_patch_dataset(fold["test_ids"], id_to_basename, DESPECKLED_DIR)

        n_train_pos, n_train_neg = int((y_train != 0).sum()), int((y_train == 0).sum())
        n_test_pos, n_test_neg = int((y_test != 0).sum()), int((y_test == 0).sum())
        assert n_train_pos == n_train_neg, f"Fold {fold['subexp']} train not 1:1"
        assert n_test_pos == n_test_neg, f"Fold {fold['subexp']} test not 1:1"

        fold_totals.append(n_train_pos + n_test_pos)
        print(f"Fold {fold['subexp']}: train={X_train.shape}, test={X_test.shape}")

    assert len(set(fold_totals)) == 1, f"Fold volcano totals disagree: {fold_totals}"
    total_positives = fold_totals[0]
    assert total_positives == 1520, f"Expected 1520 labeled volcanoes, got {total_positives}"
    print(f"\nVerified: all {len(hom38_folds)} folds see the same {total_positives} total volcanoes.")
    print("Setup OK.")


if __name__ == "__main__":
    main()
