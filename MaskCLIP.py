"""
MaskCLIP zero-shot UAV segmentation
Paper: https://arxiv.org/abs/2112.01071

Install dependencies:
    pip install transformers torch torchvision opencv-python matplotlib Pillow scipy

Requires a CUDA GPU.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import csv
import timeit
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import median_filter, gaussian_filter

from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor

from support.scoring_general import evaluate_single_image

# ── CUDA guard ────────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not found. This script requires a CUDA-capable GPU.\n"
        "Check your drivers or PyTorch CUDA build: https://pytorch.org/get-started/locally/"
    )
DEVICE = "cuda"

# ── Folders ──────────────────────────────────────────────────────────────────

INPUT_FOLDER = r"<path_to_your_input_images>"
PRED_DIR     = r"<path_to_your_predictions>"
GT_DIR       = r"<path_to_your_ground_truth>"


if not os.path.isdir(INPUT_FOLDER):
    raise ValueError(f"Input folder not found: {INPUT_FOLDER}")
if not os.path.isdir(GT_DIR):
    raise ValueError(f"Ground truth folder not found: {GT_DIR}")
if not os.path.isdir(PRED_DIR):
    raise ValueError(f"Prediction directory not found: {PRED_DIR}")
SHOW_PLOTS = False

# ── Tuning knobs ──────────────────────────────────────────────────────────────
KEY_SMOOTHING_TAU = 0.097

CLASS_SCORE_SIGMA = {
    "Building":            7.0,
    "Road":                0.0,
    "Car":                 0.0,
    "Tree":                4.0,
    "Low vegetation":      5.0,
    "Human":               0.0,
    "Background clutter":  0.0,
}

# CLASS_MIN_BLOB_PX = {
#     "Building":           10000,
#     "Road":               5000,
#     "Car":                1000,
#     "Tree":               1000,
#     "Low vegetation":     1000,
#     "Human":               800,
#     "Background clutter": 2000,
# }


CLASS_MIN_BLOB_PX = {
    "Building":           100,
    "Road":               41500,
    "Car":                4500,
    "Tree":               2550,
    "Low vegetation":     11600,
    "Human":              790,
    "Background clutter": 2050,
}

# Context filtering: remove Car/Human blobs with no Road pixel within this
# many pixels. Eliminates rooftop false positives without touching blob sizes.
# The Road mask is dilated ONCE per class (not per blob) so this is fast.
# Raise CONTEXT_SEARCH_RADIUS if roadside/pavement cars are being removed.

CLASS_CONTEXT_RULES = {
    "Car":   { 
        "required": ["Road"],
        "radius": 30,
    },

    "Human": {
        "required": ["Road"],
        "radius": 80,
    },   # remove this line to keep detections on rooftops too
}

# Batch size for GPU inference. Lower to 8 if you get CUDA out-of-memory.
BATCH_SIZE = 16

# ── Prompt templates ──────────────────────────────────────────────────────────
PROMPT_TEMPLATES = [
    # "a photo of a {}.",
    # "a photo of the {}.",
    # "a photo of a small {}.",
    # "a photo of a large {}.",
    # "a photo of a big {}.",
    # "a photo of one {}.",
    # "a good photo of a {}.",
    # "a bad photo of a {}.",
    # "a bright photo of a {}.",
    # "a dark photo of a {}.",
    # "a blurry photo of a {}.",
    # "a low contrast photo of a {}.",
    # "a high contrast photo of a {}.",
    # "a low resolution photo of a {}.",
    # "a jpeg corrupted photo of a {}.",
    # "a pixelated photo of a {}.",
    # "a cropped photo of a {}.",
    # "a photo of the hard to see {}.",
    # "an aerial photo of a {}.",
    # "a satellite photo of a {}.",
    # "a UAV image of a {}.",
     "a drone photo of a {}.",
    # "a top-down view of a {}.",
    # "an overhead photo of a {}.",
    # "a bird's-eye view of a {}.",
    # "an aerial view of a {}.",
    # "a nadir view of a {}.",
]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_maskclip_model():
    ckpt = "openai/clip-vit-base-patch16"
    # use_safetensors=False: load pytorch .bin directly, stops the background
    # safetensors conversion thread that was keeping the process alive after exit.
    model = CLIPModel.from_pretrained(ckpt, use_safetensors=False).to(DEVICE).eval()
    tokenizer = CLIPTokenizer.from_pretrained(ckpt)
    image_processor = CLIPImageProcessor.from_pretrained(ckpt)
    return model, image_processor, tokenizer


# ── Text embeddings ───────────────────────────────────────────────────────────

'''
def get_text_embeddings(model, tokenizer, class_names: list[str]):
    """
    Returns:
        text_feats       : (C, 512) averaged+normalised embeddings used for scoring
        per_template_feats: (C, T, 512) individual per-template embeddings, kept
                            for click-to-analyse template debugging
    """
    all_averaged   = []
    all_per_template = []
    with torch.no_grad():
        for cls in class_names:
            prompts = [t.format(cls) for t in PROMPT_TEMPLATES]
            tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(DEVICE)
            embeddings = model.get_text_features(**tokens)   # (T, 512)
            embeddings = F.normalize(embeddings, dim=-1)
            all_per_template.append(embeddings)              # keep individual
            all_averaged.append(embeddings.mean(dim=0))
    text_feats        = F.normalize(torch.stack(all_averaged), dim=-1)  # (C, 512)
    per_template_feats = torch.stack(all_per_template)                   # (C, T, 512)
    return text_feats, per_template_feats
'''
def get_text_embeddings(model, tokenizer, class_names: list[str]):
    """
    Returns:
        text_feats        : (C, 512) averaged embeddings used for scoring
        per_template_feats: (C, T, 512) embeddings per template (for debugging)
    """
    all_averaged = []
    all_per_template = []

    with torch.no_grad():
        for cls in class_names:
            prompts = [t.format(cls) for t in PROMPT_TEMPLATES]
            tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(DEVICE)

            embeddings = model.get_text_features(**tokens)

            # HuggingFace compatibility
            if not isinstance(embeddings, torch.Tensor):
                if hasattr(embeddings, "pooler_output") and embeddings.pooler_output is not None:
                    embeddings = embeddings.pooler_output
                elif hasattr(embeddings, "last_hidden_state"):
                    embeddings = embeddings.last_hidden_state[:, 0, :]
                else:
                    raise TypeError(f"Unexpected text feature output type: {type(embeddings)}")

            embeddings = embeddings.float()
            embeddings = F.normalize(embeddings, dim=-1)

            all_per_template.append(embeddings)        # (T,512)
            all_averaged.append(embeddings.mean(dim=0))

    text_feats = torch.stack(all_averaged)
    text_feats = F.normalize(text_feats, dim=-1)

    per_template_feats = torch.stack(all_per_template)  # (C,T,512)

    return text_feats, per_template_feats

# ── Joint V + K feature extraction ───────────────────────────────────────────

def get_value_and_key_features(model, pixel_values: torch.Tensor):
    """
    Extract VALUE and KEY features from the last attention layer in one pass.
    VALUE features (V @ W_out) skip attention-weighted aggregation, preserving
    per-patch spatial detail (MaskCLIP core modification, Section 3.3).
    KEY features are used for key smoothing to suppress isolated wrong-class patches.
    """
    value_feats = []
    key_feats   = []
    attn = model.vision_model.encoder.layers[-1].self_attn

    def v_hook(module, input, output):
        v_proj = F.linear(output, attn.out_proj.weight, attn.out_proj.bias)
        value_feats.append(v_proj)

    def k_hook(module, input, output):
        key_feats.append(output)

    h_v = attn.v_proj.register_forward_hook(v_hook)
    h_k = attn.k_proj.register_forward_hook(k_hook)

    with torch.no_grad():
        model.vision_model(pixel_values=pixel_values)

    h_v.remove()
    h_k.remove()

    v = value_feats[0][:, 1:, :]   # (B, 196, 768) — drop CLS token
    k = key_feats[0][:, 1:, :]     # (B, 196, 768)

    with torch.no_grad():
        v = model.visual_projection(v)  # (B, 196, 512)

    return F.normalize(v, dim=-1), F.normalize(k, dim=-1)


# ── Key smoothing ─────────────────────────────────────────────────────────────

def apply_key_smoothing(sim: np.ndarray, k: torch.Tensor) -> np.ndarray:
    """
    smoothed_sim = softmax(K @ K.T / τ) @ sim
    Blends each patch's scores toward visually similar neighbours in K-space.
    """
    k_np  = k.squeeze(0).cpu().numpy()
    k_sim = k_np @ k_np.T / KEY_SMOOTHING_TAU
    k_sim = np.exp(k_sim - k_sim.max(axis=1, keepdims=True))
    k_sim /= k_sim.sum(axis=1, keepdims=True)
    return k_sim @ sim


# ── Batched inference ─────────────────────────────────────────────────────────

def process_batch(model, patches: list[np.ndarray],
                  text_feats: torch.Tensor,
                  image_processor,
                  patch_size: int) -> list[np.ndarray]:
    """
    Run a batch of patches through the model in a single forward pass.
    Returns a list of (patch_size, patch_size, C) upsampled score arrays.
    """
    pil_patches  = [Image.fromarray(p).resize((224, 224)) for p in patches]
    inputs       = image_processor(images=pil_patches, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    v, k = get_value_and_key_features(model, pixel_values)  # (B,196,512), (B,196,768)

    results = []
    for b in range(len(patches)):
        sim      = (v[b] @ text_feats.T).cpu().numpy()      # (196, C)
        sim      = apply_key_smoothing(sim, k[b:b+1])
        sim_grid = sim.reshape(14, 14, -1)
        sim_full = cv2.resize(sim_grid, (patch_size, patch_size),
                              interpolation=cv2.INTER_LINEAR)
        results.append(sim_full)
    return results


# ── Post-processing ───────────────────────────────────────────────────────────

def remove_small_blobs(mask: np.ndarray, class_names: list[str],
                       min_blob_px_map: dict) -> np.ndarray:
    out = mask.copy()
    for cls_id, cls_name in enumerate(class_names):
        threshold = min_blob_px_map.get(cls_name, 500)
        if threshold == 0:
            continue
        binary = (mask == cls_id).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < threshold:
                x0 = max(0, stats[lbl, cv2.CC_STAT_LEFT] - 16)
                y0 = max(0, stats[lbl, cv2.CC_STAT_TOP]  - 16)
                x1 = min(mask.shape[1], x0 + stats[lbl, cv2.CC_STAT_WIDTH]  + 32)
                y1 = min(mask.shape[0], y0 + stats[lbl, cv2.CC_STAT_HEIGHT] + 32)
                region      = mask[y0:y1, x0:x1]
                blob_pixels = (labels[y0:y1, x0:x1] == lbl)
                neighbour_vals = region[~blob_pixels]
                if neighbour_vals.size == 0:
                    continue
                out[labels == lbl] = int(np.bincount(neighbour_vals.ravel()).argmax())
    return out


def filter_blobs_by_context(mask: np.ndarray, class_names: list[str], context_rules: dict) -> np.ndarray:
    """
    Remove blobs that have no required neighbour class within search_radius px.

    The key efficiency fix vs the previous version: the required neighbour mask
    (e.g. Road) is dilated exactly ONCE per class pair, then ALL blobs are checked
    against that single dilated mask in one vectorised operation. The old version
    dilated each blob individually — O(n_blobs) dilation calls — which was the
    source of the slowdown.
    """
    out        = mask.copy()
    name_to_id = {name: i for i, name in enumerate(class_names)}

    for cls_name, rule in context_rules.items():

        if cls_name not in name_to_id:
            continue

        cls_id = name_to_id[cls_name]

        required_neighbours = rule["required"]
        r = rule["radius"]

        # Circular dilation kernel
        ky, kx = np.ogrid[-r:r+1, -r:r+1]
        kernel = (kx**2 + ky**2 <= r**2).astype(np.uint8)

        # Combined neighbour mask
        combined_neighbour = np.zeros(mask.shape, dtype=np.uint8)

        for neighbour_name in required_neighbours:

            if neighbour_name not in name_to_id:
                continue

            neighbour_id = name_to_id[neighbour_name]

            neighbour_mask = (mask == neighbour_id).astype(np.uint8)

            dilated = cv2.dilate(neighbour_mask, kernel)

            combined_neighbour = np.maximum(combined_neighbour, dilated)

        # Find blobs of current class
        binary = (mask == cls_id).astype(np.uint8)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )

        for lbl in range(1, n_labels):

            blob_pixels = (labels == lbl)

            # Keep blob if ANY required neighbour is nearby
            if np.any(combined_neighbour[blob_pixels]):
                continue

            # Otherwise relabel blob
            x0 = max(0, stats[lbl, cv2.CC_STAT_LEFT] - r)
            y0 = max(0, stats[lbl, cv2.CC_STAT_TOP] - r)

            x1 = min(mask.shape[1],
                     stats[lbl, cv2.CC_STAT_LEFT]
                     + stats[lbl, cv2.CC_STAT_WIDTH] + r)

            y1 = min(mask.shape[0],
                     stats[lbl, cv2.CC_STAT_TOP]
                     + stats[lbl, cv2.CC_STAT_HEIGHT] + r)

            region = out[y0:y1, x0:x1]

            blob_crop = (labels[y0:y1, x0:x1] == lbl)

            neighbour_vals = region[~blob_crop]

            if neighbour_vals.size == 0:
                continue

            replacement = int(
                np.bincount(neighbour_vals.ravel()).argmax()
            )

            out[labels == lbl] = replacement


    return out


# ── Main segmentation pipeline ────────────────────────────────────────────────

def segment_4k_uav_image(image_path: str, class_names: list[str],
                          patch_size: int = 224, stride: int = 75,
                          median_filter_size: int = 33) -> np.ndarray:
    model, image_processor, tokenizer = load_maskclip_model()
    text_feats, per_template_feats = get_text_embeddings(model, tokenizer, class_names)

    img = np.array(Image.open(image_path).convert("RGB"))
    H, W = img.shape[:2]
    n_classes = len(class_names)

    y_steps = list(dict.fromkeys(list(range(0, H - patch_size + 1, stride)) + [H - patch_size]))
    x_steps = list(dict.fromkeys(list(range(0, W - patch_size + 1, stride)) + [W - patch_size]))

    all_positions = [(y, x) for y in y_steps for x in x_steps]
    n_batches     = (len(all_positions) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Image: {W}x{H}  |  Windows: {len(all_positions)}  |  Batches: {n_batches}")

    score_map = np.zeros((H, W, n_classes), dtype=np.float32)
    count_map = np.zeros((H, W),            dtype=np.float32)

    for batch_idx in range(n_batches):
        batch_pos     = all_positions[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
        batch_patches = [img[y:y + patch_size, x:x + patch_size] for y, x in batch_pos]
        print(f"  Batch {batch_idx + 1}/{n_batches}", end="\r")

        sim_grids = process_batch(model, batch_patches, text_feats,
                                  image_processor, patch_size)
        for (y, x), sim_full in zip(batch_pos, sim_grids):
            score_map[y:y + patch_size, x:x + patch_size] += sim_full
            count_map[y:y + patch_size, x:x + patch_size] += 1.0

    print()

    count_map  = np.maximum(count_map, 1.0)
    avg_scores = score_map / count_map[..., None]

    print("Applying per-class score smoothing...")
    for cls_id, cls_name in enumerate(class_names):
        sigma = CLASS_SCORE_SIGMA.get(cls_name, 0.0)
        if sigma > 0:
            avg_scores[..., cls_id] = gaussian_filter(avg_scores[..., cls_id], sigma=sigma)

    final_mask = avg_scores.argmax(axis=-1).astype(np.int32)

    # print('Skippig blob filtering for experiment')
    print("Removing small blobs...")
    final_mask = remove_small_blobs(final_mask, class_names, CLASS_MIN_BLOB_PX)

    print("Context filtering...")
    final_mask = filter_blobs_by_context(
        final_mask, class_names, CLASS_CONTEXT_RULES
    )

    if median_filter_size > 1:
        print(f"Applying median filter (kernel={median_filter_size})...")
        final_mask = median_filter(final_mask, size=median_filter_size)
        
    return final_mask, avg_scores, per_template_feats, model, image_processor


# ── Visualisation ─────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "Building":            (255, 140,   0),  # orange
    "Road":                ( 80,  80, 255),  # blue
    "Car":                 (  0, 220, 220),  # cyan
    "Tree":                ( 34, 180,  34),  # green
    "Low vegetation":      (180, 230,  50),  # yellow-green
    "Human":               (220,  30,  30),  # red
    "Background clutter":  (160, 160, 160),  # grey
}
_FALLBACK_COLORS = [
    (255, 0, 255), (255, 165, 0), (0, 255, 128), (128, 0, 128), (0, 128, 255),
]


def get_class_colors(class_names: list[str]) -> np.ndarray:
    colors, fallback_i = [], 0
    for cls in class_names:
        if cls in CLASS_COLORS:
            colors.append(CLASS_COLORS[cls])
        else:
            colors.append(_FALLBACK_COLORS[fallback_i % len(_FALLBACK_COLORS)])
            fallback_i += 1
    return np.array(colors, dtype=np.uint8)


def create_overlay(image_path: str, mask: np.ndarray,
                   class_names: list[str], alpha: float = 0.4):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    if img.shape[:2] != mask.shape:
        raise ValueError(f"Mask {mask.shape} doesn't match image {img.shape[:2]}")
    colors     = get_class_colors(class_names)
    color_mask = colors[mask]
    overlay    = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)
    legend     = [mpatches.Patch(color=np.array(colors[i]) / 255.0, label=cls)
                  for i, cls in enumerate(class_names)]
    return overlay, legend


# ── Entry point ───────────────────────────────────────────────────────────────


def analyse_pixel(px: int, py: int, img: np.ndarray, mask: np.ndarray,
                  avg_scores: np.ndarray, per_template_feats: torch.Tensor,
                  class_names: list[str], model, image_processor,
                  patch_size: int = 224) -> None:
    """
    Re-run inference on the window centred on (px, py) and print a full
    per-template score breakdown for every class.

    This is the template debugging tool: if "pixelated photo of a road" is
    systematically hurting Road scores you will see it consistently rank last
    across many clicks on road pixels.

    The window is clamped to image bounds, so edge pixels just get a
    slightly off-centre window rather than failing.
    """
    H, W = img.shape[:2]
    # Centre the 224x224 window on the clicked pixel, clamped to bounds
    x0 = int(np.clip(px - patch_size // 2, 0, W - patch_size))
    y0 = int(np.clip(py - patch_size // 2, 0, H - patch_size))
    patch = img[y0:y0 + patch_size, x0:x0 + patch_size]

    pil_patch    = Image.fromarray(patch).resize((224, 224))
    inputs       = image_processor(images=pil_patch, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    v, _ = get_value_and_key_features(model, pixel_values)  # (1, 196, 512)

    # The clicked pixel maps to a specific token in the 14x14 grid.
    # Find which token covers it within the clamped window.
    token_x = int(np.clip((px - x0) * 14 // patch_size, 0, 13))
    token_y = int(np.clip((py - y0) * 14 // patch_size, 0, 13))
    token_idx = token_y * 14 + token_x

    patch_feat = v[0, token_idx, :]  # (512,) — the specific token for this pixel

    detected_cls = class_names[mask[py, px]]
    raw_scores   = avg_scores[py, px, :]
    exp_s        = np.exp(raw_scores - raw_scores.max())
    softmax      = exp_s / exp_s.sum()

    print(f"\n{'='*60}")
    print(f"  Pixel ({px}, {py})  →  {detected_cls}  ({softmax[mask[py,px]]*100:.1f}%)")
    print(f"  Window: ({x0},{y0}) to ({x0+patch_size},{y0+patch_size})")
    print(f"  Token:  grid[{token_y},{token_x}]  (index {token_idx})")
    print(f"{'='*60}")

    # Per-class, per-template breakdown
    # per_template_feats: (C, T, 512)
    T = per_template_feats.shape[1]
    pf = patch_feat.unsqueeze(0)  # (1, 512)

    for cls_id, cls_name in enumerate(class_names):
        tmpl_embeds = per_template_feats[cls_id]           # (T, 512)
        tmpl_scores = (pf @ tmpl_embeds.T).squeeze(0)      # (T,)
        tmpl_scores_np = tmpl_scores.cpu().numpy()

        avg_score   = tmpl_scores_np.mean()
        best_idx    = int(tmpl_scores_np.argmax())
        worst_idx   = int(tmpl_scores_np.argmin())

        marker = "  ◀ DETECTED" if cls_name == detected_cls else ""
        print(f"\n  {cls_name}{marker}")
        print(f"    avg score: {avg_score:.4f}   "
              f"softmax: {softmax[cls_id]*100:.1f}%")
        print(f"    best  template ({tmpl_scores_np[best_idx]:.4f}): "
              f"{PROMPT_TEMPLATES[best_idx].format(cls_name)}")
        print(f"    worst template ({tmpl_scores_np[worst_idx]:.4f}): "
              f"{PROMPT_TEMPLATES[worst_idx].format(cls_name)}")

        # Show all templates sorted by score so you can spot bad ones
        sorted_idx = np.argsort(tmpl_scores_np)[::-1]
        for rank, ti in enumerate(sorted_idx):
            bar = "█" * int(20 * (tmpl_scores_np[ti] - tmpl_scores_np.min()) /
                            (tmpl_scores_np.max() - tmpl_scores_np.min() + 1e-8))
            print(f"    [{rank+1:2d}] {tmpl_scores_np[ti]:.4f} {bar:<20} "
                  f"{PROMPT_TEMPLATES[ti].format(cls_name)}")

    print(f"{'='*60}\n")


def mask_to_rgb(mask, class_names):
    color_map = {
        "Building": [128, 0, 0],
        "Road": [128, 64, 128],
        "Tree": [0, 128, 0],
        "Low vegetation": [128, 128, 0],
        "Background clutter": [0, 0, 0],
        "Car": [192, 0, 192],
        "Human": [64, 64, 0],
    }

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for i, cls in enumerate(class_names):
        rgb[mask == i] = color_map[cls]

    return rgb


def summarise_eval_metrics(per_class, miou, weighted_miou):
    """
    Convert the per-class metrics dict into one macro summary for the image.
    """
    classes = list(per_class.keys())

    macro_precision = float(np.mean([per_class[c]["Precision"] for c in classes]))
    macro_recall    = float(np.mean([per_class[c]["Recall"]    for c in classes]))
    #macro_f05       = float(np.mean([per_class[c]["F0.5"]      for c in classes]))
    #macro_f1        = float(np.mean([per_class[c]["F1"]        for c in classes]))
    #macro_f2        = float(np.mean([per_class[c]["F2"]        for c in classes]))
    #mean_batch_size = 

    return {
        "mIoU": miou,
        "FWIoU": weighted_miou,
        "MacroPrecision": macro_precision,
        "MacroRecall": macro_recall,
        #"MacroF0.5": macro_f05,
        #"MacroF1": macro_f1,
        #"MacroF2": macro_f2,
        #"Computationallength": mean_batch_size,
    }

def save_eval_summary_csv(results_list, csv_path):
    """
    Save per-image results plus a final average row.
    """
    if not results_list:
        print("No evaluation results to save.")
        return

    fieldnames = [
        "Image",
        "mIoU",
        "FWIoU",
        "MacroPrecision",
        "MacroRecall",
        # "MacroF0.5",
        # "MacroF1",
        # "MacroF2",
    ]

    # Compute averages
    avg_row = {"Image": "AVERAGE"}
    for key in fieldnames[1:]:
        avg_row[key] = float(np.mean([r[key] for r in results_list]))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results_list:
            writer.writerow(row)

        writer.writerow(avg_row)

    print(f"Saved evaluation summary to: {csv_path}")

    print("\nDataset averages:")
    for key in fieldnames[1:]:
        print(f"  {key}: {avg_row[key]:.4f}")


if __name__ == "__main__":

    # start = timeit.default_timer()

    classes = [
        "Building", "Road", "Car", "Tree",
        "Low vegetation", "Human", "Background clutter"
    ]

    os.makedirs(PRED_DIR, exist_ok=True)

    all_eval_results = []

    per_class_history = {}

    image_files = sorted([
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not image_files:
        raise ValueError(f"No images found in folder: {INPUT_FOLDER}")

    for f in image_files:
        print("  ", f)

    print(f"Found {len(image_files)} image(s) in {INPUT_FOLDER}")

    for fname in image_files:
        image_path = os.path.join(INPUT_FOLDER, fname)

        print(f"\nProcessing: {fname}")
        print(f"Device: {DEVICE}")
        print("Segmenting...")

        mask, avg_scores, per_template_feats, model, image_processor = (
            segment_4k_uav_image(image_path, classes)
        )

        rgb_pred = mask_to_rgb(mask, classes)

        save_path = os.path.join(PRED_DIR, fname)
        cv2.imwrite(save_path, cv2.cvtColor(rgb_pred, cv2.COLOR_RGB2BGR))
        print(f"Saved prediction to: {save_path}")

        gt_path = os.path.join(GT_DIR, fname)
        
        if os.path.exists(gt_path):
            print("Evaluating prediction...")
        
        per_class, miou, weighted_miou = evaluate_single_image(
            save_path.replace("Predictions", "Images")
            )
        

        if (per_class is not None and miou is not None and weighted_miou is not None):
            summary = summarise_eval_metrics(per_class, miou, weighted_miou)
            summary["Image"] = fname
            all_eval_results.append(summary)

            for cls_name, metrics in per_class.items():

                if cls_name not in per_class_history:
                    per_class_history[cls_name] = {
                        "Precision": [],
                        "Recall": [],
                    }

                per_class_history[cls_name]["Precision"].append(metrics["Precision"])
                per_class_history[cls_name]["Recall"].append(metrics["Recall"])


        else:
            print(f"GT not found for {fname} at: {gt_path}")

        if SHOW_PLOTS:
            img_rgb = np.array(Image.open(image_path).convert("RGB"))

            print("Rendering...")
            overlay, legend_patches = create_overlay(image_path, mask, classes)

            fig, ax = plt.subplots(figsize=(20, 11))
            ax.imshow(overlay)
            ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1),
                      loc="upper left", borderaxespad=0)
            ax.axis("off")

            tooltip = ax.text(
                0.01, 0.99, "",
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=10,
                color="white",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.7),
                zorder=10,
            )

            def on_move(event):
                if event.inaxes != ax or event.xdata is None:
                    tooltip.set_text("")
                    fig.canvas.draw_idle()
                    return

                px = int(round(event.xdata))
                py = int(round(event.ydata))

                H, W = mask.shape
                if not (0 <= px < W and 0 <= py < H):
                    tooltip.set_text("")
                    fig.canvas.draw_idle()
                    return

                detected_cls = classes[mask[py, px]]
                raw_scores = avg_scores[py, px, :]

                exp_s = np.exp(raw_scores - raw_scores.max())
                softmax = exp_s / exp_s.sum()

                winner_conf = softmax[mask[py, px]] * 100
                sorted_idx = np.argsort(softmax)[::-1]
                runnerup_idx = sorted_idx[1]
                runnerup_cls = classes[runnerup_idx]
                runnerup_conf = softmax[runnerup_idx] * 100

                tooltip.set_text(
                    f"Detected:  {detected_cls}  ({winner_conf:.1f}%)\n"
                    f"Runner-up: {runnerup_cls}  ({runnerup_conf:.1f}%)\n"
                    f"Pixel: ({px}, {py})"
                )
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect("motion_notify_event", on_move)

            def on_click(event):
                if event.inaxes != ax or event.xdata is None or event.button != 1:
                    return
                px = int(round(event.xdata))
                py = int(round(event.ydata))
                H, W = mask.shape
                if not (0 <= px < W and 0 <= py < H):
                    return

                analyse_pixel(px, py, img_rgb, mask, avg_scores, per_template_feats,
                              classes, model, image_processor)

            fig.canvas.mpl_connect("button_press_event", on_click)
            ax.set_title("MaskCLIP Zero-Shot UAV Segmentation  |  Left-click any pixel for template breakdown in console")
            plt.tight_layout()
            plt.show()

    summary_csv_path = os.path.join(PRED_DIR, "evaluation_summary.csv")
    save_eval_summary_csv(all_eval_results, summary_csv_path)

    # stop = timeit.default_timer()
    print("\n" + "="*60)
    print("DATASET AVERAGE PRECISION / RECALL")
    print("="*60)

    for cls_name, metrics in per_class_history.items():

        avg_precision = np.mean(metrics["Precision"])
        avg_recall    = np.mean(metrics["Recall"])

        print(f"\n{cls_name}")
        print(f"  Avg Precision: {avg_precision:.4f}")
        print(f"  Avg Recall:    {avg_recall:.4f}")

    # print(f'Running time is {stop - start}')
    print("\nFinished processing all images.")
    os._exit(0)

