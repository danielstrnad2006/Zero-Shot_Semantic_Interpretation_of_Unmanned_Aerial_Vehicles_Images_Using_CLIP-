# =============================================================================
# Import Libraries
# =============================================================================
import os
os.environ["HF_HUB_OFFLINE"] = "0"

# =============================================================================
# Toggles
# =============================================================================
SAVE_OUTPUTS = True

# =============================================================================
# File Paths
# =============================================================================
IMAGE_PATH = "uavid/uavid_val/seq67/Images/"
LABEL_PATH = "uavid/uavid_val/seq67/Labels/"
OUTPUT_PATH = "output/val/"

METRICS_PATH = "output/metrics.xlsx"
METRICS_SHEET = "val"

SAM_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
SAM_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
DINO_CONFIG = "groundingdino/config/GroundingDINO_SwinB_cfg.py"

# =============================================================================
# Hyperparameters
# =============================================================================
SAHI_OVERLAP = 0.25     # default 0.25, lower = less context overlap between patches
SAHI_PATCH_SIZE = 1000  # default 1000, lower = more detailed patches

DINO_BOX_THRESH = 0.20  # default 0.20, lower = more boxes with lower confidences
DINO_TEXT_THRESH = 0.20 # default 0.20, lower = more low-confidence category classifications
DINO_NMS_THRESH = 0.80  # default 0.80, lower = less overlapping masks of same category

SAM_MM_OUTPUT = True    # default True, True  = segment multiple masks and select best
SAM_BATCH_SIZE = 32     # default 32,   lower = slower, less memory usage

# =============================================================================
# Categories
# =============================================================================
CATEGORIES = { # idx (do not change), color, f-score weight
    'car':                  [0, (192, 0, 192),      1.0],
    'moving car':           [1, (64, 0, 128),       1.0], # merged with 'car' for evaluation
    'low vegetation':       [2, (128, 128, 0),      1.0],
    'tree':                 [3, (0, 128, 0),        1.0],
    'building':             [4, (128, 0, 0),        1.0],
    'road':                 [5, (128, 64, 128),     1.0],
    'human':                [6, (64, 64, 0),        1.0],
    'background clutter':   [7, (0, 0, 0),          1.0],
    'empty':                [8, (255, 255, 255),    1.0], # not in the label, default category
}

EXCEL_CATEGORY_ORDER = ['car', 'low vegetation', 'tree', 'building', 'road', 'human', 'background clutter'] # has to match Excel order