# =============================================================================
# Import Libraries
# =============================================================================
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
from PIL import Image
import groundingdino.datasets.transforms as T

# =============================================================================
# Load Image
# =============================================================================
def load_image(image_path):
    if not os.path.exists(image_path):
        return None
    with Image.open(image_path).convert("RGB") as image_source:
        image = np.asarray(image_source)
    return image

# =============================================================================
# Find File Names in Target
# =============================================================================
def find_file_names(folder):
    file_names = os.listdir(folder)
    return file_names

# =============================================================================
# Generate Text Prompts
# =============================================================================
def generate_prompts(categories):
    prompts = ""
    for c in categories:
        if c not in ['moving car', 'empty']:
            prompts += c + ". "
    prompts = prompts.strip().casefold()
    return prompts

# =============================================================================
# Get SAHI slices
# =============================================================================
def get_slices(image_height, image_width, patch_size, overlap_ratio):
    stride = int(patch_size * (1 - overlap_ratio))
    slices = []
    for y in range(0, image_height - patch_size + stride, stride):
        for x in range(0, image_width - patch_size + stride, stride):
            y_end = min(y + patch_size, image_height)
            x_end = min(x + patch_size, image_width)
            y_start = max(0, y_end - patch_size)
            x_start = max(0, x_end - patch_size)
            slices.append((x_start, y_start, x_end, y_end))
    return slices

def transform_patch(input_patch):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    input_patch = Image.fromarray(input_patch)
    output_patch, _ = transform(input_patch, None)
    return output_patch

# =============================================================================
# Generate Histogram
# =============================================================================
def gen_histogram(mIoUs, FWIoUs, fname):
    bins = np.arange(0, 1, 0.05)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    ax.hist(mIoUs, bins=bins, weights=np.ones(len(mIoUs))/len(mIoUs), color="blue", alpha=0.6, label="mIoU")
    ax.hist(FWIoUs, bins=bins, weights=np.ones(len(FWIoUs))/len(FWIoUs), color="orange", alpha=0.6, label="FWIoU")

    ax.legend()
    ax.grid(alpha=0.25)
    ax = plt.gca()

    ax.set_xlabel("Metric")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))

    ax.set_ylabel("Relative Frequency")
    ax.set_ylim(0, 0.6)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    plt.show()
    fig.savefig(f"output/{fname}")