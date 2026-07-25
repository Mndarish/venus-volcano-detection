"""Dataset wrapper for the Deep SVDD method (and its AE-pretraining stage)."""
import torch
from torch.utils.data import Dataset

from common import normalize_patch


class PatchDataset(Dataset):
    """Unlabeled background patches."""

    def __init__(self, patches):
        self.patches = normalize_patch(patches).astype("float32")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return torch.from_numpy(self.patches[idx]).unsqueeze(0)
