# Autoencoder (Reconstruction Error) — **investigated and ruled out**

**For the shipped detector, use `methods/supervised/` instead.** This
folder is kept because the reasoning behind ruling this method out is
part of the project's methodology — a negative result worth documenting,
not hiding.

## What was tried

A convolutional autoencoder (`Stage1AutoencoderSpatialLatent`: strided-conv
encoder, a small 4×4×`latent_channels` spatial bottleneck — not a
GAP-collapsed vector, since full GAP discards positional info and produced
generic centered-blob reconstructions) was trained on background-only
patches per fold, using a hybrid `alpha·MSE + (1-alpha)·(1-SSIM)`
reconstruction loss (motivated by MSE's overreaction to SAR speckle noise
and SSIM's training instability when used alone). Anomaly score = per-pixel
reconstruction error.

## Why it failed

Across all 6 HOM38 folds, volcanoes **reconstructed better than
background** — separation ratios (volcano_error / bg_error) of 0.53×–0.96×,
i.e. backwards from what an anomaly score needs (should be > 1.0).

A controlled comparison against an **untrained Gaussian blur baseline**
(zero learned parameters, see `evaluate.py`'s `gaussian_blur_baseline_error`)
reproduced the same reversed pattern (ratios 0.44×–0.56×). This proves the
effect is a **structural artifact**, not something the autoencoder
specifically learned from training data: smooth content (volcano domes) is
inherently easier for any low-pass system to reconstruct than textured
content (background fractures/ridges/tessera). No amount of retraining or
architecture tweaking fixes this — the anomaly-detection framing itself is
mismatched to this dataset's content structure.

## Files

- `model.py` — `Stage1AutoencoderSpatialLatent`, `HybridReconstructionLoss`
- `datasets.py` — `PatchDataset`
- `train.py` — training loop, all 6 folds
- `evaluate.py` — reconstruction error separation check, complexity
  confound check (does error just track raw patch variance?), and the
  Gaussian blur control that confirmed the structural-artifact diagnosis
- `run_colab.ipynb` — clone + train + evaluate in Colab

## Usage

```bash
python -m methods.autoencoder.train
python -m methods.autoencoder.evaluate
```

## A note on a fixed bug

The original exploratory notebook's training loop called an undefined
`Stage1Autoencoder(latent_dim=64)` — that class doesn't appear anywhere in
the notebook; only `Stage1AutoencoderSpatialLatent(latent_channels=...)`
does. `train.py` here uses the latter, which is the only architecture
actually defined and known to run. If you have prior results quoting a
64-dim vector latent, they came from a different, unrecorded class
definition — don't assume they transfer to this code as-is.
