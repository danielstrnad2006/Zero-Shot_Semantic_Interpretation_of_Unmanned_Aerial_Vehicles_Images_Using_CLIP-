# =============================================================================
# Import Libraries
# =============================================================================
import torch
from torchvision.ops import nms, box_convert
from groundingdino.util.inference import load_model, predict

# =============================================================================
# Load Model
# =============================================================================
def load_dino(config_path, checkpoint_path, device):
    try:
        model = load_model(config_path, checkpoint_path, device)
        return model
    except Exception:
        return None
    
# =============================================================================
# Predict Bounding Boxes
# =============================================================================
def predict_bbx(model, image_transformed, prompts, box_thresh, text_thresh, device):
    boxes, logits, labels = predict(
        model=model,
        image=image_transformed,
        caption=prompts,
        box_threshold=box_thresh,
        text_threshold=text_thresh,
        device=device
    )
    return boxes, logits, labels

# =============================================================================
# Process Bounding Boxes
# =============================================================================
def convert_bbx(image, input_boxes):
    h, w, _ = image.shape
    input_boxes = input_boxes * torch.Tensor([w, h, w, h])
    output_boxes = box_convert(boxes=input_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    return output_boxes

def convert_labels(input_labels):
    output_labels = []
    for label in input_labels:
        l = str(label).lower()
        if 'low' in l or 'vegetation' in l:
            output_labels.append('low vegetation')
        elif 'background' in l or 'clutter' in l:
            output_labels.append('background clutter')
        else:
            output_labels.append(l)
    return output_labels

def nms_filter(input_boxes, input_logits, input_labels, iou_thresh):
    if len(input_boxes) == 0:
        return input_boxes, input_logits, input_labels
    unique_labels = list(set(input_labels))
    label_map = {name: i for i, name in enumerate(unique_labels)}
    label_ids = torch.tensor([label_map[l] for l in input_labels], device=input_boxes.device)
    max_coordinate = input_boxes.max()
    offsets = label_ids.to(input_boxes.dtype) * (max_coordinate + 1.0)
    shifted_boxes = input_boxes + offsets[:, None]
    keep_indices = nms(shifted_boxes, input_logits, iou_thresh)
    output_boxes = input_boxes[keep_indices]
    output_logits = input_logits[keep_indices]
    output_labels = [input_labels[i] for i in keep_indices.tolist()]
    return output_boxes, output_logits, output_labels