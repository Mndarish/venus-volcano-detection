"""
Model architecture for the autoencoder method.

STATUS: investigated and ruled out — see README.md in this folder for the
full negative result. Kept in the repo because the reasoning behind ruling
it out is itself part of the project's methodology, not because this is
the shipped detector. For the shipped method, see methods/supervised.
"""
import torch
import torch.nn as nn
from pytorch_msssim import ssim


class Stage1AutoencoderSpatialLatent(nn.Module):
    """Same encoder/decoder skeleton used throughout the project, but the
    bottleneck keeps a small spatial extent (4x4xlatent_channels) instead
    of collapsing to a pure vector via GAP — full GAP discards ALL
    positional info, which is why early reconstructions were generic
    centered blobs regardless of where structure actually was in the
    input."""

    def __init__(self, latent_channels=16, in_channels=1):
        super().__init__()
        self.latent_channels = latent_channels

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),  # 64->32
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 32->16
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16->8
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 8->4
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

        # No GAP, no Linear — just a 1x1 conv to compress channels,
        # keeping the 4x4 spatial grid intact.
        self.to_latent = nn.Conv2d(128, latent_channels, kernel_size=1)
        self.from_latent = nn.Conv2d(latent_channels, 128, kernel_size=1)

        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 4->8
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 8->16
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 16->32
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 32->64
            nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        z = self.to_latent(h)  # (B, latent_channels, 4, 4) — spatial kept
        return z

    def decode(self, z):
        h = self.from_latent(z)  # (B, 128, 4, 4)
        return self.dec(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class HybridReconstructionLoss(nn.Module):
    """alpha * MSE + (1 - alpha) * (1 - SSIM). Motivated by MSE's
    overreaction to SAR speckle noise and SSIM training instability when
    used alone."""

    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, recon, target):
        mse_loss = self.mse(recon, target)
        # ssim() expects (B, C, H, W) in [0,1] — matches our Sigmoid output + /255 normalization
        ssim_val = ssim(recon, target, data_range=1.0, size_average=True)
        ssim_loss = 1 - ssim_val
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss
