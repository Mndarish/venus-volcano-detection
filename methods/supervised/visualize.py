"""Visual inspection helpers — show misclassified patches and false alarms
rather than just counting them. Not needed for training/evaluation; use
interactively (e.g. from run_colab.ipynb) when you want to eyeball errors."""
import numpy as np
import torch
import matplotlib.pyplot as plt

from common import normalize_patch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def plot_misclassified_patches(model, X, y_true, title, n_show=12, device=DEVICE):
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(normalize_patch(X).astype(np.float32)).unsqueeze(1).to(device)
        preds = model(tensor).argmax(dim=1).cpu().numpy()

    wrong_idx = np.where(preds != y_true)[0]
    if len(wrong_idx) == 0:
        print(f"{title}: no misclassifications to show.")
        return

    n_show = min(n_show, len(wrong_idx))
    chosen = np.random.RandomState(42).choice(wrong_idx, size=n_show, replace=False)

    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()

    for i, idx in enumerate(chosen):
        axes[i].imshow(X[idx], cmap="gray")
        axes[i].set_title(f"True: Cat{y_true[idx]+1}, Pred: Cat{preds[idx]+1}", fontsize=10)
        axes[i].axis("off")
    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_false_alarms(X_fp, pred_categories, title, n_show=8):
    if len(X_fp) == 0:
        print(f"{title}: no false alarms to show.")
        return
    n_show = min(n_show, len(X_fp))
    chosen = np.random.RandomState(42).choice(len(X_fp), size=n_show, replace=False)

    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()

    for i, idx in enumerate(chosen):
        axes[i].imshow(X_fp[idx], cmap="gray")
        axes[i].set_title(f"Predicted: Cat{pred_categories[idx]+1}", fontsize=10)
        axes[i].axis("off")
    for i in range(n_show, len(axes)):
        axes[i].axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()
