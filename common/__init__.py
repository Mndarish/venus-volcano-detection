"""
Shared data pipeline: raw .sdt loading -> Lee filter despeckling ->
HOM38 fold definitions -> positive/negative patch extraction ->
per-fold dataset build.

Every method (methods/supervised, methods/autoencoder, methods/deep_svdd)
imports from here rather than duplicating this logic.
"""
from .config import (
    BASE_PATH,
    CAT_NAMES,
    CHECKPOINTS_DIR,
    CU2_FINAL,
    DESPECKLED_DIR,
    GT_DIR,
    IMAGES_DIR,
    NETWORK_INPUT_SIZE,
    STAGE1_NEGATIVES_PER_IMAGE,
    TABLES_DIR,
    WINDOW_SIZE_FINAL,
)
from .data_loading import load_lxyr_labels, load_raw_image, parse_ground_truths_table
from .despeckle import despeckle_all_tiles, get_volcano_mask_exclusions, lee_filter, load_despeckled_batch
from .folds import load_hom38_folds, make_stage1_train_val_split, parse_experiments_table
from .patches import (
    build_patch_dataset,
    build_stage1_dataset,
    extract_negative_patches,
    extract_positive_patches,
    normalize_patch,
    patch_overlaps_volcano,
)

__all__ = [
    "BASE_PATH", "IMAGES_DIR", "GT_DIR", "TABLES_DIR", "DESPECKLED_DIR", "CHECKPOINTS_DIR",
    "CU2_FINAL", "WINDOW_SIZE_FINAL", "NETWORK_INPUT_SIZE", "STAGE1_NEGATIVES_PER_IMAGE", "CAT_NAMES",
    "load_raw_image", "load_lxyr_labels", "parse_ground_truths_table",
    "lee_filter", "despeckle_all_tiles", "load_despeckled_batch", "get_volcano_mask_exclusions",
    "parse_experiments_table", "load_hom38_folds", "make_stage1_train_val_split",
    "normalize_patch", "patch_overlaps_volcano", "extract_positive_patches",
    "extract_negative_patches", "build_patch_dataset", "build_stage1_dataset",
]
