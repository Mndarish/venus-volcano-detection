"""
Central configuration: paths and locked, empirically-derived constants
shared by every method (supervised, autoencoder, deep_svdd).

Paths are relative to the repo root by default, so this works out of the
box after `git clone` + `pip install -r requirements.txt`, whether you're
running locally or in Colab. Override the data location with the
VOLCANO_DATA_DIR environment variable if your data lives elsewhere
(e.g. still mounted from Google Drive).
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_PATH = os.environ.get("VOLCANO_DATA_DIR", os.path.join(REPO_ROOT, "data"))
IMAGES_DIR = os.path.join(BASE_PATH, "Images")
GT_DIR = os.path.join(BASE_PATH, "GroundTruths")
TABLES_DIR = os.path.join(BASE_PATH, "Tables")
DESPECKLED_DIR = os.path.join(BASE_PATH, "Despeckled")

CHECKPOINTS_DIR = os.environ.get("VOLCANO_CHECKPOINTS_DIR", os.path.join(REPO_ROOT, "checkpoints"))

# --- Lee filter despeckling: locked production config ---
# Cu2 = median of Cu2 estimates across 24 flat, volcano-free patches (8 sample
# images spanning the ID range) — more robust than a single manual-patch
# estimate (0.01591). window_size was never in question, only Cu2 was.
# See scripts/derive_cu2.py for the derivation.
CU2_FINAL = 0.01086
WINDOW_SIZE_FINAL = 11

# --- Patch extraction ---
NETWORK_INPUT_SIZE = 64   # fixed size fed to every model in this project
CROP_MULTIPLIER = 4.0     # crop window = r * multiplier (adaptive to volcano radius)
MIN_CROP = 32             # floor, so tiny volcanoes (r=1) don't get a near-zero crop
MAX_CROP = 200            # ceiling, so the r=87.66 outlier doesn't demand an absurd crop

# --- Stage 1 background sampling (autoencoder / deep_svdd unsupervised methods) ---
# Well under the min candidate count (327 patches/image), so this cap always
# triggers uniformly across all 134 tiles.
STAGE1_NEGATIVES_PER_IMAGE = 200

# --- HOM38 cross-validation ---
HOM38_ID_RANGE = range(5, 43)  # image IDs 5-42
CAT_NAMES = ["Cat1", "Cat2", "Cat3", "Cat4"]
