"""
Model architectures for the Deep SVDD method.

STATUS: investigated and ruled out — see README.md in this folder for the
full negative result. For the shipped method, see methods/supervised.
"""
import torch.nn as nn


class DeepSVDDEncoder(nn.Module):
    """No bias anywhere -- including BatchNorm's affine params, which
    otherwise reintroduce a bias-like shift/scale that lets the network
    trivially collapse every input toward the fixed center c (a known
    Deep SVDD pitfall: BatchNorm's default affine=True gives the network
    exactly the loophole the no-bias design was meant to close)."""

    def __init__(self, embedding_dim=32, in_channels=1):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 64->32
            nn.BatchNorm2d(16, affine=False), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 32->16
            nn.BatchNorm2d(32, affine=False), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 16->8
            nn.BatchNorm2d(64, affine=False), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 8->4
            nn.BatchNorm2d(128, affine=False), nn.LeakyReLU(0.1, inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, embedding_dim, bias=False)

    def forward(self, x):
        h = self.enc(x)
        h = self.gap(h).flatten(1)  # (B, 128)
        z = self.fc(h)  # (B, embedding_dim)
        return z


class PretrainAutoencoder(nn.Module):
    """Standard AE (bias allowed -- this stage isn't subject to SVDD
    collapse, since reconstruction has no degenerate all-zero shortcut).
    Encoder here must match DeepSVDDEncoder's conv layer shapes exactly,
    so weights transfer."""

    def __init__(self, embedding_dim=32, in_channels=1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_enc = nn.Linear(128, embedding_dim)

        self.fc_dec = nn.Linear(embedding_dim, 128 * 4 * 4)
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, in_channels, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.enc(x)
        h = self.gap(h).flatten(1)
        z = self.fc_enc(h)
        d = self.fc_dec(z).view(-1, 128, 4, 4)
        return self.dec(d), z


def transplant_encoder_weights(pretrained_ae, svdd_model):
    """Copy conv-layer weights from the pretrained AE's encoder into the
    bias-free SVDD encoder (BatchNorm running stats included, affine
    skipped since svdd_model's BatchNorm has affine=False)."""
    ae_state = pretrained_ae.enc.state_dict()
    svdd_state = svdd_model.enc.state_dict()
    for key in svdd_state:
        if key in ae_state and svdd_state[key].shape == ae_state[key].shape:
            svdd_state[key] = ae_state[key]
    svdd_model.enc.load_state_dict(svdd_state)

    # final Linear: AE's fc_enc has bias, SVDD's fc doesn't -- transfer weight only
    svdd_model.fc.weight.data = pretrained_ae.fc_enc.weight.data.clone()
    return svdd_model
