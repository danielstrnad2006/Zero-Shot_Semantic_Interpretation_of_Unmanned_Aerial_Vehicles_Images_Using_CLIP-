# =============================================================================
# Import Libraries
# =============================================================================
import numpy as np
import torch
import time
import gc
import xlwings as xw
import warnings
import logging

# =============================================================================
# Suppress Warnings
# =============================================================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_outputs").setLevel(logging.ERROR)

# =============================================================================
# Import Functions
# =============================================================================
import config
import data
import dino
import sam
import scoring
import visual

# =============================================================================
# Main Function
# =============================================================================
def main(patch_size, box_threshold, text_threshold, nms_threshold):

    # =========================================================================
    # Initialise Function
    # =========================================================================
    print("\n"+"="*80)
    print("Grounded SAM 2.1 Zero-Shot Image Classification")
    print("="*80)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing {DEVICE}")
    if DEVICE == "cuda":
        print(f"  with {torch.cuda.get_device_name(0)}")
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"  with tfloat32")

    # =========================================================================
    # Load Models
    # =========================================================================
    print("\nLoading Models...")
    grounding_model = dino.load_dino(
        config.DINO_CONFIG,
        config.DINO_CHECKPOINT,
        DEVICE
    )
    if grounding_model is None:
        print("✗ Failed to load GroundingDINO, exiting...")
        return
    else:
        print("✓ Successfully loaded GroundingDINO")
        print(f"  Config:     {config.DINO_CONFIG}")
        print(f"  Checkpoint: {config.DINO_CHECKPOINT}")

    predictor = sam.load_sam(
        config.SAM_CONFIG,
        config.SAM_CHECKPOINT,
        DEVICE
    )
    if predictor is None:
        print("✗ Failed to load SAM, exiting...")
        return
    else:
        print("✓ Successfully loaded SAM")
        print(f"  Config:     {config.SAM_CONFIG}")
        print(f"  Checkpoint: {config.SAM_CHECKPOINT}")

    prompts = data.generate_prompts(config.CATEGORIES)

    print("\nSettings:")
    print(f"  Patch Size:       {patch_size}")
    print(f"  Box Threshold:    {box_threshold}")
    print(f"  Text Threshold:   {text_threshold}")
    print(f"  NMS Threshold:    {nms_threshold}")
    print(f"  Prompts:          {prompts}")

    # =========================================================================
    # Loop through Images
    # =========================================================================
    if config.SAVE_OUTPUTS:
        metrics_excel = xw.Book(config.METRICS_PATH).sheets[config.METRICS_SHEET]
    mIoUs = []
    FWIoUs = []

    file_names = data.find_file_names(config.IMAGE_PATH)
    for file_id, file in enumerate(file_names):
        print("\n"+"="*80)
        print(f"\nProcessing image {file_id+1} of {len(file_names)}...")
        start = time.perf_counter()

        # =====================================================================
        # Load Image and Label (Ground Truth)
        # =====================================================================
        img_path = config.IMAGE_PATH + file
        lbl_path = config.LABEL_PATH + file
        out_path = config.OUTPUT_PATH + file

        image = data.load_image(img_path)
        label = data.load_image(lbl_path)
        if image is None or label is None:
            print("✗ Failed to load image or label, skipping this file")
            continue
        else:
            print(f"✓ Loaded image from {img_path}")
            print(f"✓ Loaded label from {lbl_path}")

        # =====================================================================
        # Classify Bounding Boxes with SAHI
        # =====================================================================
        print("\n  Classifying initial bounding boxes with GroundingDINO...")
        h, w, _ = image.shape
        slices = data.get_slices(h, w, patch_size, config.SAHI_OVERLAP)
        all_boxes, all_logits, all_labels = [], [], []
        for (x1, y1, x2, y2) in slices:
            patch = image[y1:y2, x1:x2]
            patch_transformed = data.transform_patch(patch) 
            p_boxes, p_logits, p_labels = dino.predict_bbx(
                grounding_model, 
                patch_transformed, 
                prompts, 
                box_threshold, 
                text_threshold, 
                DEVICE
            )
            if len(p_boxes) > 0:
                p_boxes = dino.convert_bbx(patch, p_boxes)
                p_boxes[:, [0, 2]] += x1
                p_boxes[:, [1, 3]] += y1
                all_boxes.append(p_boxes)
                all_logits.append(p_logits)
                all_labels.extend(p_labels)
                
        if len(all_boxes) > 0:
            boxes = torch.cat(all_boxes)
            box_logits = torch.cat(all_logits)
            box_labels = dino.convert_labels(all_labels)
            boxes, box_logits, box_labels = dino.nms_filter(boxes, box_logits, box_labels, nms_threshold)
        else:
            boxes, box_logits, box_labels = torch.empty((0,4)), torch.empty(0), []

        print(f"  ✓ Classified {len(boxes)} bounding boxes in {len(slices)} slices")

        # =====================================================================
        # Segment Bounding Boxes
        # =====================================================================
        print("\n  Segmenting bounding boxes with SAM...")
        predictor.set_image(image)

        pred_flat, min_areas_flat = scoring.init_prediction_state(h, w, config.CATEGORIES['empty'][0])

        mask_count = 0
        for i, masks_batch in enumerate(sam.segment_bbx(
            predictor,
            boxes,
            config.SAM_BATCH_SIZE,
            config.SAM_MM_OUTPUT,
            DEVICE
        )):
            if masks_batch.size == 0:
                continue
            
            start_idx = i * config.SAM_BATCH_SIZE
            end_idx = start_idx + len(masks_batch)
            labels_batch = box_labels[start_idx:end_idx]

            pred_flat, min_areas_flat = scoring.update_prediction_state(
                pred_flat,
                min_areas_flat,
                masks_batch,
                labels_batch,
                config.CATEGORIES
            )
            mask_count += len(masks_batch)

        pred_map = pred_flat.reshape(h, w)
        print(f"  ✓ Segmented {mask_count} masks")

        # =====================================================================
        # Compute and Save Performance Metric
        # =====================================================================
        print("\n  Computing performance metrics...")
        metrics, mIoU, FWIoU = scoring.calc_metrics(
            label, 
            pred_map,
            config.CATEGORIES
        )
        mIoUs.append(mIoU)
        FWIoUs.append(FWIoU)

        if config.SAVE_OUTPUTS:
            scoring.save_metrics(
                file_id, 
                metrics_excel, 
                metrics, 
                mIoU, 
                FWIoU, 
                config.EXCEL_CATEGORY_ORDER
            )
        print(f"    mIoU:  {mIoU:.3f}")
        print(f"    FWIoU: {FWIoU:.3f}")

        # =====================================================================
        # Visualise and Save Output
        # =====================================================================
        if config.SAVE_OUTPUTS:
            output = visual.render_output(pred_map, config.CATEGORIES)
            visual.save_output(output, out_path)
            print(f"  ✓ Saved output to {out_path}")

        # =====================================================================
        # Close Loop
        # =====================================================================
        elapsed = time.perf_counter() - start
        if config.SAVE_OUTPUTS:
            metrics_excel.range((file_id+5, 36)).value = elapsed
        print(f"\nImage processing took {elapsed:.1f} seconds")

        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        continue
    
    data.gen_histogram(mIoUs, FWIoUs, "histogram_1")

    print("\n"+"="*80)
    print("Finished Processing the Dataset!")
    print("="*80+"\n")
    return np.mean(mIoUs), np.mean(FWIoUs)

if __name__ == "__main__":
    mIoU, FWIoU = main(
        config.SAHI_PATCH_SIZE,
        config.DINO_BOX_THRESH,
        config.DINO_TEXT_THRESH,
        config.DINO_NMS_THRESH
    )