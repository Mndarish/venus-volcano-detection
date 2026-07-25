"""
Evaluation for the supervised two-stage method (SHIPPED).

Per Phase 7's flagged HOM38 constraint: Cat1 test counts are dangerously
thin in some folds (a structural fact of the fold scheme, not a bug), so
single-fold per-class metrics aren't trustworthy on their own. Everything
here reports cross-fold mean +/- std.

Usage:
    python -m methods.supervised.evaluate
"""
import os

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from common import CAT_NAMES, DESPECKLED_DIR, TABLES_DIR, build_patch_dataset, load_hom38_folds, normalize_patch, parse_ground_truths_table
from common.config import CHECKPOINTS_DIR
from .model import PatchClassifier, build_resnet_grader

STAGE1_CKPT_DIR = os.path.join(CHECKPOINTS_DIR, "supervised", "stage1_detector")
STAGE2_CKPT_DIR = os.path.join(CHECKPOINTS_DIR, "supervised", "stage2_grader")
RESNET_CKPT_DIR = os.path.join(CHECKPOINTS_DIR, "supervised", "stage2_resnet")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_stage1(fold, id_to_basename, device=DEVICE):
    fold_idx = fold["subexp"]
    model = PatchClassifier(num_classes=1).to(device)
    model.load_state_dict(torch.load(os.path.join(STAGE1_CKPT_DIR, f"stage1_fold{fold_idx}.pt"), map_location=device))
    model.eval()

    X_test, y_test, _ = build_patch_dataset(fold["test_ids"], id_to_basename, DESPECKLED_DIR)
    y_test_bin = (y_test != 0).astype(int)
    with torch.no_grad():
        tensor = torch.from_numpy(normalize_patch(X_test).astype(np.float32)).unsqueeze(1).to(device)
        preds = (torch.sigmoid(model(tensor)) > 0.5).cpu().numpy().flatten().astype(int)

    acc = accuracy_score(y_test_bin, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_bin, preds, average="binary", zero_division=0)
    return {"acc": acc, "precision": prec, "recall": rec, "f1": f1}, (X_test, y_test, preds)


def evaluate_stage2(fold, id_to_basename, device=DEVICE):
    fold_idx = fold["subexp"]
    model = PatchClassifier(num_classes=4).to(device)
    model.load_state_dict(torch.load(os.path.join(STAGE2_CKPT_DIR, f"stage2_fold{fold_idx}.pt"), map_location=device))
    model.eval()

    X_test, y_test, _ = build_patch_dataset(fold["test_ids"], id_to_basename, DESPECKLED_DIR)
    pos_mask = y_test != 0
    X_test_pos, y_test_pos = X_test[pos_mask], y_test[pos_mask] - 1

    with torch.no_grad():
        tensor = torch.from_numpy(normalize_patch(X_test_pos).astype(np.float32)).unsqueeze(1).to(device)
        preds = model(tensor).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test_pos, preds)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_test_pos, preds, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    return {
        "acc": acc, "precision": prec, "recall": rec, "f1": f1, "support": support,
        "confusion_matrix": confusion_matrix(y_test_pos, preds),
    }


def cross_fold_report(id_to_basename, hom38_folds):
    stage1_accs, stage1_f1s, stage2_accs = [], [], []
    per_cat_precision = {c: [] for c in CAT_NAMES}
    per_cat_recall = {c: [] for c in CAT_NAMES}
    per_cat_f1 = {c: [] for c in CAT_NAMES}

    for fold in hom38_folds:
        s1, _ = evaluate_stage1(fold, id_to_basename)
        s2 = evaluate_stage2(fold, id_to_basename)

        stage1_accs.append(s1["acc"])
        stage1_f1s.append(s1["f1"])
        stage2_accs.append(s2["acc"])
        for i, cat in enumerate(CAT_NAMES):
            per_cat_precision[cat].append(s2["precision"][i])
            per_cat_recall[cat].append(s2["recall"][i])
            per_cat_f1[cat].append(s2["f1"][i])

        print(f"Fold {fold['subexp']} — Stage1 acc={s1['acc']:.3f} f1={s1['f1']:.3f} | "
              f"Stage2 acc={s2['acc']:.3f} | support: {dict(zip(CAT_NAMES, s2['support']))}")

    print("\nStage 1 (binary detection) across 6 folds:")
    print(f"  Accuracy: {np.mean(stage1_accs):.3f} +/- {np.std(stage1_accs):.3f}")
    print(f"  F1:       {np.mean(stage1_f1s):.3f} +/- {np.std(stage1_f1s):.3f}")

    print("\nStage 2 (Cat1-4 grading) across 6 folds:")
    print(f"  Accuracy: {np.mean(stage2_accs):.3f} +/- {np.std(stage2_accs):.3f}")

    print("\nCross-fold per-category summary (mean +/- std):")
    for cat in CAT_NAMES:
        p_mean, p_std = np.mean(per_cat_precision[cat]), np.std(per_cat_precision[cat])
        r_mean, r_std = np.mean(per_cat_recall[cat]), np.std(per_cat_recall[cat])
        f_mean, f_std = np.mean(per_cat_f1[cat]), np.std(per_cat_f1[cat])
        print(f"  {cat}: precision={p_mean:.3f}+/-{p_std:.3f}, recall={r_mean:.3f}+/-{r_std:.3f}, "
              f"F1={f_mean:.3f}+/-{f_std:.3f}")

    return {"per_cat_f1": per_cat_f1, "stage2_accs": stage2_accs}


def end_to_end_report(id_to_basename, hom38_folds, device=DEVICE):
    """Chains Stage 1 -> Stage 2: a volcano only 'counts' end-to-end if
    Stage 1 detects it AND Stage 2 grades it correctly. Also reports
    Stage 1 miss rate and what category Stage 2 assigns to false alarms."""
    end_to_end_accs, stage1_miss_rates, false_alarm_category_counts = [], [], []

    for fold in hom38_folds:
        fold_idx = fold["subexp"]
        stage1_model = PatchClassifier(num_classes=1).to(device)
        stage1_model.load_state_dict(torch.load(os.path.join(STAGE1_CKPT_DIR, f"stage1_fold{fold_idx}.pt"), map_location=device))
        stage1_model.eval()

        stage2_model = PatchClassifier(num_classes=4).to(device)
        stage2_model.load_state_dict(torch.load(os.path.join(STAGE2_CKPT_DIR, f"stage2_fold{fold_idx}.pt"), map_location=device))
        stage2_model.eval()

        X_test, y_test, _ = build_patch_dataset(fold["test_ids"], id_to_basename, DESPECKLED_DIR)
        tensor_all = torch.from_numpy(normalize_patch(X_test).astype(np.float32)).unsqueeze(1).to(device)

        with torch.no_grad():
            stage1_preds = (torch.sigmoid(stage1_model(tensor_all)) > 0.5).cpu().numpy().flatten().astype(int)

        true_volcano_mask = y_test != 0
        n_true_volcanoes = true_volcano_mask.sum()
        detected_volcano_mask = true_volcano_mask & (stage1_preds == 1)
        missed_volcano_mask = true_volcano_mask & (stage1_preds == 0)

        n_correct_end_to_end = 0
        if detected_volcano_mask.sum() > 0:
            X_detected = X_test[detected_volcano_mask]
            y_detected_true = y_test[detected_volcano_mask] - 1
            with torch.no_grad():
                tensor_detected = torch.from_numpy(normalize_patch(X_detected).astype(np.float32)).unsqueeze(1).to(device)
                stage2_preds = stage2_model(tensor_detected).argmax(dim=1).cpu().numpy()
            n_correct_end_to_end = int((stage2_preds == y_detected_true).sum())

        end_to_end_acc = n_correct_end_to_end / n_true_volcanoes
        miss_rate = missed_volcano_mask.sum() / n_true_volcanoes
        end_to_end_accs.append(end_to_end_acc)
        stage1_miss_rates.append(miss_rate)

        false_positive_mask = (y_test == 0) & (stage1_preds == 1)
        n_false_positives = false_positive_mask.sum()
        if n_false_positives > 0:
            X_fp = X_test[false_positive_mask]
            with torch.no_grad():
                tensor_fp = torch.from_numpy(normalize_patch(X_fp).astype(np.float32)).unsqueeze(1).to(device)
                fp_cat_preds = stage2_model(tensor_fp).argmax(dim=1).cpu().numpy()
            false_alarm_category_counts.append(np.bincount(fp_cat_preds, minlength=4))
        else:
            false_alarm_category_counts.append(np.zeros(4, dtype=int))

        print(f"Fold {fold_idx}: end-to-end accuracy={end_to_end_acc:.3f}, "
              f"Stage1 miss rate={miss_rate:.3f}, false alarms={n_false_positives}")

    print(f"\nEnd-to-end accuracy across 6 folds: {np.mean(end_to_end_accs):.3f} +/- {np.std(end_to_end_accs):.3f}")
    print(f"Stage 1 miss rate across 6 folds:  {np.mean(stage1_miss_rates):.3f} +/- {np.std(stage1_miss_rates):.3f}")

    total_fp_by_cat = np.sum(false_alarm_category_counts, axis=0)
    print(f"\nFalse alarms by assigned category (summed across all folds): "
          f"{dict(zip(CAT_NAMES, total_fp_by_cat))}")


def compare_resnet(id_to_basename, hom38_folds, per_cat_f1_cnn, device=DEVICE):
    """Compares the from-scratch CNN Stage 2 grader against the ResNet18
    transfer-learning variant, with a Wilcoxon signed-rank test per
    category (n=6 folds — low power, interpret cautiously)."""
    resnet_per_cat_f1 = {c: [] for c in CAT_NAMES}
    resnet_overall_accs = []

    for fold in hom38_folds:
        fold_idx = fold["subexp"]
        resnet_model = build_resnet_grader(num_classes=4).to(device)
        resnet_model.load_state_dict(torch.load(os.path.join(RESNET_CKPT_DIR, f"resnet_fold{fold_idx}.pt"), map_location=device))
        resnet_model.eval()

        X_test, y_test, _ = build_patch_dataset(fold["test_ids"], id_to_basename, DESPECKLED_DIR)
        pos_mask = y_test != 0
        X_test_pos, y_test_pos = X_test[pos_mask], y_test[pos_mask] - 1

        from .datasets import ResNetPatchDataset
        from torch.utils.data import DataLoader

        test_loader = DataLoader(ResNetPatchDataset(X_test_pos, y_test_pos), batch_size=32, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                all_preds.append(resnet_model(x).argmax(dim=1).cpu().numpy())
        preds = np.concatenate(all_preds)

        acc = accuracy_score(y_test_pos, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test_pos, preds, labels=[0, 1, 2, 3], average=None, zero_division=0)
        resnet_overall_accs.append(acc)
        for i, cat in enumerate(CAT_NAMES):
            resnet_per_cat_f1[cat].append(f1[i])

    print(f"\n{'Category':<8} {'CNN F1 (scratch)':<20} {'ResNet18 F1 (transfer)':<25}")
    for cat in CAT_NAMES:
        cnn_f1 = f"{np.mean(per_cat_f1_cnn[cat]):.3f} +/- {np.std(per_cat_f1_cnn[cat]):.3f}"
        resnet_f1 = f"{np.mean(resnet_per_cat_f1[cat]):.3f} +/- {np.std(resnet_per_cat_f1[cat]):.3f}"
        print(f"{cat:<8} {cnn_f1:<20} {resnet_f1:<25}")

    for cat in CAT_NAMES:
        cnn_scores, resnet_scores = per_cat_f1_cnn[cat], resnet_per_cat_f1[cat]
        if len(set(cnn_scores)) > 1 or len(set(resnet_scores)) > 1:
            stat, p = wilcoxon(cnn_scores, resnet_scores)
            print(f"{cat}: Wilcoxon p={p:.3f} (n=6 folds — low power, interpret cautiously)")
        else:
            print(f"{cat}: scores identical or degenerate, skipping test")


def main():
    id_to_basename = parse_ground_truths_table(os.path.join(TABLES_DIR, "Images_GroundTruths_Table"))
    hom38_folds = load_hom38_folds(TABLES_DIR)

    report = cross_fold_report(id_to_basename, hom38_folds)
    print("\n" + "=" * 50)
    end_to_end_report(id_to_basename, hom38_folds)

    if os.path.isdir(RESNET_CKPT_DIR) and os.listdir(RESNET_CKPT_DIR):
        print("\n" + "=" * 50)
        compare_resnet(id_to_basename, hom38_folds, report["per_cat_f1"])


if __name__ == "__main__":
    main()
