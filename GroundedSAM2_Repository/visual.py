# =============================================================================
# Import Libraries
# =============================================================================
import numpy as np
from PIL import Image

# =============================================================================
# Render Output Image
# =============================================================================
def render_output(pred_map, categories):
    h, w = pred_map.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    output[:, :] = np.array(categories['empty'][1], dtype=np.uint8)
    for c in categories:
        color = np.array(np.array(categories[c][1], dtype=np.uint8))
        output[pred_map == categories[c][0]] = color
    return output

# =============================================================================
# Save Image to Path
# =============================================================================
def save_output(output, output_path):
    Image.fromarray(output).save(output_path)