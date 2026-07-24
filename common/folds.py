"""
HOM38 6-fold cross-validation scheme (B1-B6) over image IDs 5-42, parsed
from Experiments_Images_Table.

Known structural limitation (not a bug): Cat1 test counts are dangerously
thin in some folds. This is a property of the HOM38 split itself. Report
cross-fold mean +/- std rather than trusting any single fold's per-class
metrics — see each method's evaluate.py.
"""
import re

from .config import HOM38_ID_RANGE


def parse_matlab_range_list(range_str):
    """Parses '5,6,13:42' -> [5, 6, 13, 14, ..., 42]"""
    result = []
    for part in range_str.split(","):
        part = part.strip()
        if ":" in part:
            start, end = part.split(":")
            result.extend(range(int(start), int(end) + 1))
        elif part:
            result.append(int(part))
    return result


def parse_experiments_table(path, target_experiment="HOM38"):
    """Returns list of dicts: {subexp: 'B1', train_ids: [...], test_ids: [...]}"""
    folds = []
    in_target_section = False
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("%") and target_experiment in line:
                in_target_section = True
                continue
            if (
                line.startswith("%")
                and in_target_section
                and re.match(r"%\s*HOM\d+|%\s*[A-Z]+\d*", line)
                and target_experiment not in line
            ):
                break
            if not in_target_section or not line or line.startswith("%"):
                continue
            match = re.match(r"(\w+);\s*TRN\s*=\s*\[([^\]]*)\];\s*TST\s*=\s*\[([^\]]*)\];", line)
            if match:
                subexp, trn_str, tst_str = match.groups()
                folds.append(
                    {
                        "subexp": subexp,
                        "train_ids": parse_matlab_range_list(trn_str),
                        "test_ids": parse_matlab_range_list(tst_str),
                    }
                )
    return folds


def load_hom38_folds(tables_dir):
    """Parses HOM38 folds and runs the standard sanity checks
    (no train/test overlap, full ID coverage per fold)."""
    import os

    hom38_folds = parse_experiments_table(os.path.join(tables_dir, "Experiments_Images_Table"), "HOM38")

    all_hom38_ids = set(HOM38_ID_RANGE)
    for fold in hom38_folds:
        overlap = set(fold["train_ids"]) & set(fold["test_ids"])
        assert not overlap, f"Fold {fold['subexp']} has train/test overlap: {overlap}"
        fold_ids = set(fold["train_ids"]) | set(fold["test_ids"])
        assert not (all_hom38_ids - fold_ids), f"Fold {fold['subexp']} missing IDs"

    return hom38_folds


def make_stage1_train_val_split(train_ids, val_fraction=0.15, seed=42):
    """Split a fold's training image IDs into train/val at the IMAGE level,
    so no patch from the same image appears in both (patches from the same
    image share terrain/speckle and would otherwise leak information)."""
    import numpy as np

    rng = np.random.RandomState(seed)
    ids = list(train_ids)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    val_ids = ids[:n_val]
    stage1_train_ids = ids[n_val:]
    return stage1_train_ids, val_ids
