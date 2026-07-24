"""Dataset wrapper for the autoencoder method."""
import torch
from torch.utils.data import Dataset

from common import normalize_patch


class PatchDataset(Dataset):
    """Unlabeled background patches for Stage 1 autoencoder training."""

    def __init__(self, patches):
        # patches: (N, 64, 64) raw despeckled float32 array
        self.patches = normalize_patch(patches).astype("float32")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        return torch.from_numpy(patch).unsqueeze(0)  # add channel dim -> (1, 64, 64)
