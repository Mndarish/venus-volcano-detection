# Deep SVDD (Feature-Space Anomaly Detection) — **investigated and ruled out**

**For the shipped detector, use `methods/supervised/` instead.** Kept for
the same reason as `methods/autoencoder/`: the negative result and its
diagnosis are part of the project's methodology.

## What was tried

Deep SVDD (Ruff et al., 2018) sidesteps pixel-reconstruction bias entirely
— there's no decoder at inference, just an encoder (`DeepSVDDEncoder`) that
maps background patches toward a single fixed center `c`; anomaly score =
distance from `c`.

Two things mattered for correctness, not just performance:
- **No bias terms anywhere in the encoder**, including BatchNorm's affine
  params — otherwise the network can trivially minimize the loss by
  mapping every input to `c` regardless of content (hypersphere collapse).
- **`c` is fixed, not learned** — computed once via `initialize_center`
  from a forward pass over training data, then frozen, with any
  near-zero coordinate nudged away from zero (a coordinate of exactly
  zero is a free "cheat" for that dimension).

A plain Deep SVDD from random init collapsed (both bg and volcano
distances converged toward ~0), since nothing in the raw objective rewards
staying input-sensitive. Fix: pretrain the encoder as part of a standard
autoencoder (`PretrainAutoencoder`) first, then transplant only the
encoder weights (`transplant_encoder_weights`) into the bias-free SVDD
network before fine-tuning at a lower learning rate.

## Why it failed

Once collapse was resolved, separation still ran backwards across all 6
folds (background sat *closer* to the learned center than volcanoes did,
in most folds — the reverse of what a useful anomaly score needs).

**Root cause:** background is not a coherent "normal" class in this
dataset — it's a diverse catch-all (fractures, ridges, tessera, plains),
while volcanoes are visually homogeneous. Any clustering-style approach
will naturally treat the homogeneous class as tight and the diverse
catch-all as spread out, regardless of which one it trained on. The
premise anomaly detection depends on (normal = coherent, anomaly = rare
deviation) is inverted for this dataset.

**Conclusion:** both this and the autoencoder method failed via
independent mechanisms converging on the same root cause — a data-
structural mismatch with the anomaly-detection framing, not an
implementation bug in either. This motivated the pivot to the supervised
two-stage design in `methods/supervised/`.

## Files

- `model.py` — `DeepSVDDEncoder`, `PretrainAutoencoder`, `transplant_encoder_weights`
- `datasets.py` — `PatchDataset`
- `train.py` — `initialize_center`, `pretrain_encoder`, `train_deep_svdd_fold`
- `evaluate.py` — distance-from-center separation check across all 6 folds
- `run_colab.ipynb` — clone + train + evaluate in Colab

## Usage

```bash
python -m methods.deep_svdd.train
python -m methods.deep_svdd.evaluate
```
