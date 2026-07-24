# Venus Volcano Detection — Magellan SAR

Multi-stage machine learning pipeline for detecting and grading volcanic
features in Magellan Venus SAR (Synthetic Aperture Radar) satellite
imagery.

## Dataset

- 134 tiles of 1024×1024 grayscale SAR images (raw `.sdt`, uint8, no header)
- 1,520 labeled volcanoes across four confidence categories: Cat1
  (definite/largest) through Cat4 (most ambiguous)
- HOM38 6-fold cross-validation scheme (folds B1–B6, over image IDs 5–42)

**Known structural limitation:** Cat1 test counts are dangerously thin in
some HOM38 folds. This is a fact about the fold scheme itself, not a bug
— every method's `evaluate.py` reports cross-fold mean±std rather than
trusting any single fold's per-class metrics.

Data lives under `data/` (`data/Images`, `data/GroundTruths`, `data/Tables`)
and is committed directly to this repo.

## Pipeline

```
raw .sdt load → Lee filter despeckling (Cu² derived empirically)
             → HOM38 fold definitions
             → adaptive-crop patch extraction (positive + negative)
             → per-fold dataset build
             → [ method-specific model ]
```

This shared front end lives in `common/` and is used identically by every
method below — none of it is duplicated per method.

## Methods

| Method | Status | Folder |
|---|---|---|
| **Supervised two-stage detection** | **Shipped** | [`methods/supervised/`](methods/supervised/) |
| Autoencoder (reconstruction error) | Investigated, ruled out | [`methods/autoencoder/`](methods/autoencoder/) |
| Deep SVDD (feature-space anomaly detection) | Investigated, ruled out | [`methods/deep_svdd/`](methods/deep_svdd/) |

The two ruled-out methods are kept in the repo — and kept runnable — on
purpose: both failures were independent methods converging on the same
root cause (a data-structural mismatch with the anomaly-detection framing,
not an implementation bug in either), and that negative result is part of
the project's methodology, not something to hide. See each folder's
README for the full diagnosis.

The shipped method is a **Stage 1** binary volcano-vs-background detector
feeding a **Stage 2** Cat1-4 grader, trained directly on the 1,520 labeled
volcanoes, with a ResNet18 transfer-learning variant as a comparison point.

## Repo structure

```
venus-volcano-detection/
├── common/              # shared pipeline — data loading, folds, despeckling, patch extraction
├── methods/
│   ├── supervised/       # SHIPPED
│   ├── autoencoder/       # ruled out (documented negative result)
│   └── deep_svdd/          # ruled out (documented negative result)
├── scripts/
│   ├── verify_setup.py    # end-to-end pipeline sanity check
│   └── derive_cu2.py      # Cu2 derivation (reproducibility only — result is locked in common/config.py)
├── data/                 # committed dataset: Images/, GroundTruths/, Tables/
├── checkpoints/          # trained model weights (gitignored — see below)
├── requirements.txt
└── README.md
```

Each method folder is self-contained: `model.py`, `datasets.py`, `train.py`,
`evaluate.py`, and a `run_colab.ipynb` that clones this repo and calls
those scripts — the notebook is a thin runner, not where the logic lives.

## Getting started

```bash
git clone <this-repo>
cd venus-volcano-detection
pip install -r requirements.txt
python scripts/verify_setup.py          # confirms data, folds, despeckling, patch counts all check out
python -m methods.supervised.train      # or methods.autoencoder / methods.deep_svdd
python -m methods.supervised.evaluate
```

To run in Colab instead, open any method's `run_colab.ipynb`, fill in the
repo URL in the first code cell, and run all cells.

## Checkpoints

Trained weights are saved under `checkpoints/<method>/<stage>/` and are
**not** committed to git (see `.gitignore`) — they're large binaries that
regenerate from `train.py`, and vary by fold/method. Re-run training to
reproduce them, or add your own checkpoint-sharing convention (releases,
Git LFS, cloud storage) if you want to distribute pretrained weights.
