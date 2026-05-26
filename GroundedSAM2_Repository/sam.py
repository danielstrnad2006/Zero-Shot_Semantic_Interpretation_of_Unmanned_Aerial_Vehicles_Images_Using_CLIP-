# =============================================================================
# Import Libraries
# =============================================================================
import numpy as np
import gc
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# =============================================================================
# Load Model
# =============================================================================
def load_sam(config_path, checkpoint_path, device):
    try:
        model = build_sam2(config_path, ckpt_path=checkpoint_path, device=device)
        predictor = SAM2ImagePredictor(model)
        return predictor
    except Exception:
        return None

# =============================================================================
# Segment Bounding Boxes
# =============================================================================
def segment_bbx(predictor, boxes, batch_size, mm_output, device):
    if len(boxes) == 0:
        yield np.array([])
        return
    
    remainder = len(boxes)
    for i in range(0, boxes.shape[0], batch_size):
        batch = boxes[i : i + min(batch_size, remainder)]
        masks_batch, scores_batch, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=batch,
            multimask_output=mm_output,
        )
    
        if masks_batch.dtype != bool:
            masks_batch = (masks_batch > 0.0).astype(bool)
        
        if mm_output:
            if scores_batch.ndim == 1:
                scores_batch = scores_batch[None, :]
                masks_batch = masks_batch[None, :, :, :]
            best = np.argmax(scores_batch, axis=1)
            masks_batch = masks_batch[np.arange(masks_batch.shape[0]), best]

        yield masks_batch

        remainder -= batch_size
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()