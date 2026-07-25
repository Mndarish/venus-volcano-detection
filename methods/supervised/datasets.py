"""Dataset wrappers for the supervised two-stage method."""
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from common import normalize_patch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class LabeledPatchDataset(Dataset):
    """Patches + integer labels for the from-scratch CNN Stage 1/2 models."""

    def __init__(self, patches, labels):
        self.patches = normalize_patch(patches).astype("float32")
        self.labels = labels

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.patches[idx]).unsqueeze(0)
        y = torch.tensor(self.labels[idx])
        return x, y


class ResNetPatchDataset(Dataset):
    """Replicates the single despeckled channel to 3, resizes to 224x224,
    and applies ImageNet normalization so pretrained filters see input in
    the distribution they were trained on."""

    def __init__(self, patches, labels):
        self.patches = normalize_patch(patches).astype("float32")
        self.labels = labels
        self.resize = T.Resize((224, 224), antialias=True)
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.patches[idx]).unsqueeze(0)  # (1, 64, 64)
        x = x.repeat(3, 1, 1)  # (3, 64, 64)
        x = self.resize(x)  # (3, 224, 224)
        x = self.normalize(x)
        y = torch.tensor(self.labels[idx])
        return x, y
