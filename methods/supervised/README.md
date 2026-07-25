# Supervised Two-Stage Detection — **SHIPPED**

This is the primary method. It uses the 1,520 labeled volcanoes directly,
rather than working around them with an unsupervised anomaly-detection
framing (see `methods/autoencoder/` and `methods/deep_svdd/` for why those
were tried first and ruled out).

## Architecture

- **Stage 1** (`PatchClassifier(num_classes=1)`): binary volcano-vs-background
  detector. 4-stage strided-conv backbone, GAP, linear head. Bias terms and
  standard BatchNorm are fine here — the collapse failure mode that forced
  a bias-free design in Deep SVDD is specific to that method's unsupervised
  distance-to-center objective, and doesn't apply once there's a real label
  and decision boundary to learn.
- **Stage 2** (`PatchClassifier(num_classes=4)`): Cat1-4 grader, trained
  only on true positives (background never reaches Stage 2 during
  training). Uses inverse-frequency class weighting computed **per fold,
  from that fold's own training distribution** — imbalance severity varies
  by fold (Cat1 is especially thin in some), so a single global weighting
  would be wrong.
- **ResNet18 variant** (`build_resnet_grader`): ImageNet-pretrained,
  fine-tunes only `layer4` + `fc`. Used as a comparison point against the
  from-scratch Stage 2 CNN, not as a replacement.

## Files

- `model.py` — `PatchClassifier`, `build_resnet_grader`
- `datasets.py` — `LabeledPatchDataset`, `ResNetPatchDataset`
- `train.py` — training loops for all three models, all 6 HOM38 folds
- `evaluate.py` — cross-fold mean±std reporting, end-to-end pipeline
  metrics (Stage 1 → Stage 2 chained), ResNet comparison with Wilcoxon
  signed-rank tests
- `visualize.py` — optional: plot misclassified patches / false alarms
- `run_colab.ipynb` — clone + train + evaluate in Colab

## Usage

```bash
python -m methods.supervised.train      # trains Stage 1, Stage 2, and ResNet18 for all 6 folds
python -m methods.supervised.evaluate   # cross-fold report + end-to-end metrics
```

## Reporting note

Cat1 test counts are dangerously thin in some HOM38 folds — this is a
structural property of the fold scheme itself, not a bug. Trust the
cross-fold mean±std numbers in `evaluate.py`'s output over any single
fold's per-category metrics.
