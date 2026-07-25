"""
Model architectures for the supervised two-stage method (SHIPPED).

Stage 1: binary volcano-vs-background detector.
Stage 2: Cat1-4 grader (positives only).
Both stages share the same 4-stage strided-conv backbone (PatchClassifier),
differing only in the final layer's output size. Bias terms and standard
BatchNorm are fine here — the Deep SVDD collapse failure mode (see
methods/deep_svdd) was specific to the unsupervised distance-to-center
objective and doesn't apply once there's a real label and decision
boundary to learn.

A ResNet18 transfer-learning variant is also included for Stage 2, as a
comparison point against the from-scratch CNN grader.
"""
import torch.nn as nn
import torchvision.models as models


class PatchClassifier(nn.Module):
    """Shared backbone for both Stage 1 (binary) and Stage 2 (4-way)."""

    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),  # 64->32
            nn.Conv2d(16, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),  # 32->16
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),  # 16->8
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),  # 8->4
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.enc(x)
        h = self.gap(h).flatten(1)
        return self.fc(h)


def build_resnet_grader(num_classes=4):
    """ResNet18 (ImageNet pretrained), fine-tuned for Cat1-4 grading.

    Freezes everything except layer4 and the final fc — only the
    highest-level, most task-specific features get fine-tuned; early conv
    layers (general edges/textures) stay as pretrained.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for name, param in model.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
