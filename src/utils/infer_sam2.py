#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 inference script for binary fish segmentation.
Uses SAM2AutomaticMaskGenerator for proper mask generation.
Outputs masks in YOLOv11 format.

UPDATED: Now uses SAM2AutomaticMaskGenerator instead of exhaustive grid sampling
for reasonable number of meaningful masks (~10-100 instead of ~4000).
"""

import argparse
import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import json

# SAM2 imports
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_AVAILABLE = True
except ImportError:
    print("WARNING: SAM2 not available. Install with:")
    print("  pip install git+https://github.com/facebookresearch/sam2.git")
    SAM2_AVAILABLE = False

# =============================================================================
# DEFAULT GLOBAL CONFIGURATION VARIABLES
# =============================================================================

# Model configuration
MODEL_PATH = 'weights/sam2_model.pth'
SAM2_MODEL_ID = "facebook/sam2-hiera-large"

# Dataset paths
INPUT_IMAGES_FOLDER = 'dataset/test'
OUTPUT_LABELS_FOLDER = 'dataset/inference/labels'
OUTPUT_IMAGES_FOLDER = 'dataset/inference/images'

# Model parameters
TARGET_CLASS_INDEX = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inference parameters
MASK_THRESHOLD = 0.0
MIN_MASK_AREA = 100

# Parameters for SAM2AutomaticMaskGenerator
POINTS_PER_SIDE = 32
PRED_IOU_THRESH = 0.88
STABILITY_SCORE_THRESH = 0.95
MIN_MASK_REGION_AREA = 100

# Area filtering (same as training)
MIN_MASK_AREA_ORIGINAL = [256, 100]

# Class configuration
CLASS_NAMES = {0: "Chromis chromis", 1: "Coris julis"}

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='SAM2 inference and YOLO format conversion'
    )
    
    # Path arguments
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                        help='Path to fine-tuned SAM2 weights')
    parser.add_argument('--sam2_model_id', type=str, default=SAM2_MODEL_ID,
                        help='Hugging Face model ID for SAM2')
    parser.add_argument('--input_images_folder', type=str,
                        default=INPUT_IMAGES_FOLDER,
                        help='Folder containing input images')
    parser.add_argument('--output_labels_folder', type=str,
                        default=OUTPUT_LABELS_FOLDER,
                        help='Folder to save YOLO format labels')
    parser.add_argument('--output_images_folder', type=str,
                        default=OUTPUT_IMAGES_FOLDER,
                        help='Folder to save visualization images')
    
    # Inference parameters
    parser.add_argument('--mask_threshold', type=float,
                        default=MASK_THRESHOLD,
                        help='Threshold for binary mask predictions')
    parser.add_argument('--min_mask_area', type=int,
                        default=MIN_MASK_AREA,
                        help='Minimum mask area in pixels')
    
    # Parameters from main.py
    parser.add_argument('--points_per_side', type=int,
                        default=32,
                        help='Number of points per side for grid sampling')
    parser.add_argument('--pred_iou_thresh', type=float,
                        default=0.88,
                        help='IoU threshold for mask quality')
    parser.add_argument('--stability_score_thresh', type=float,
                        default=0.95,
                        help='Stability score threshold')
    parser.add_argument('--min_mask_region_area', type=int,
                        default=100,
                        help='Minimum mask region area')
    
    # Class configuration
    parser.add_argument('--target_class_index', type=int,
                        default=TARGET_CLASS_INDEX,
                        help='Target class index (0 or 1)')
    parser.add_argument('--class_names', type=str, default=None,
                        help='JSON string with class ID to name mapping')
    
    return parser.parse_args()


def update_global_variables(args):
    """Update global variables with command-line arguments."""
    global MODEL_PATH, SAM2_MODEL_ID, INPUT_IMAGES_FOLDER, \
           OUTPUT_LABELS_FOLDER, OUTPUT_IMAGES_FOLDER, \
           MASK_THRESHOLD, MIN_MASK_AREA, TARGET_CLASS_INDEX, CLASS_NAMES, \
           POINTS_PER_SIDE, PRED_IOU_THRESH, STABILITY_SCORE_THRESH, \
           MIN_MASK_REGION_AREA
    
    MODEL_PATH = args.model_path
    SAM2_MODEL_ID = args.sam2_model_id
    INPUT_IMAGES_FOLDER = args.input_images_folder
    OUTPUT_LABELS_FOLDER = args.output_labels_folder
    OUTPUT_IMAGES_FOLDER = args.output_images_folder
    MASK_THRESHOLD = args.mask_threshold
    MIN_MASK_AREA = args.min_mask_area
    TARGET_CLASS_INDEX = args.target_class_index
    
    # Update new parameters
    POINTS_PER_SIDE = args.points_per_side
    PRED_IOU_THRESH = args.pred_iou_thresh
    STABILITY_SCORE_THRESH = args.stability_score_thresh
    MIN_MASK_REGION_AREA = args.min_mask_region_area
    
    # Update class names if provided
    if args.class_names is not None:
        CLASS_NAMES = json.loads(args.class_names)
        CLASS_NAMES = {int(k): v for k, v in CLASS_NAMES.items()}
        print(f"Updated CLASS_NAMES from arguments: {CLASS_NAMES}")


# =============================================================================
# SAM2 Model Loading - USING AUTOMATIC MASK GENERATOR
# =============================================================================

def load_sam2_mask_generator(model_id, fine_tuned_weights_path, device):
    """
    Load SAM2AutomaticMaskGenerator for proper mask generation.
    
    Args:
        model_id: Hugging Face model ID
        fine_tuned_weights_path: Path to fine-tuned weights
        device: Device to load model on
    
    Returns:
        SAM2AutomaticMaskGenerator instance
    """
    if not SAM2_AVAILABLE:
        raise ImportError("SAM2 is not installed.")
    
    print(f"Loading SAM2 Automatic Mask Generator from: {model_id}")
    
    try:
        # Build the model first
        sam2_model = build_sam2(model_id, fine_tuned_weights_path, device=device)
        
        # Load fine-tuned weights if available
        if os.path.exists(fine_tuned_weights_path):
            print(f"Loading fine-tuned weights from {fine_tuned_weights_path}")
            state_dict = torch.load(fine_tuned_weights_path, map_location=device)
            sam2_model.load_state_dict(state_dict)
            print("Fine-tuned weights loaded successfully")
        else:
            print(f"WARNING: Fine-tuned weights not found at {fine_tuned_weights_path}")
            print("Using base SAM2 model without fine-tuning")
        
        # Create the automatic mask generator with optimized parameters
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=POINTS_PER_SIDE,
            pred_iou_thresh=PRED_IOU_THRESH,
            stability_score_thresh=STABILITY_SCORE_THRESH,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=MIN_MASK_REGION_AREA,
        )
        
        print("SAM2AutomaticMaskGenerator loaded successfully")
        return mask_generator
        
    except Exception as e:
        print(f"Error loading SAM2: {e}")
        raise


# =============================================================================
# Mask Processing Functions - USING AUTOMATIC MASK GENERATOR
# =============================================================================

def generate_masks_for_image(mask_generator, image_np):
    """
    Generate masks using SAM2's automatic mask generator.
    This gives you only meaningful objects, not every possible mask.
    
    Args:
        mask_generator: SAM2AutomaticMaskGenerator instance
        image_np: Image as numpy array (H, W, 3)
    
    Returns:
        List of meaningful masks
    """
    # Use SAM2's automatic mask generator
    masks = mask_generator.generate(image_np)
    
    # Extract just the mask arrays (masks contain metadata too)
    mask_arrays = []
    
    for mask_data in masks:
        # Each mask_data is a dict with 'segmentation', 'area', 'bbox', etc.
        binary_mask = mask_data['segmentation'].astype(np.uint8)
        
        # Apply area filter
        if binary_mask.sum() >= MIN_MASK_AREA:
            mask_arrays.append(binary_mask)
    
    return mask_arrays


def mask_to_polygon(mask, min_points=6):
    """
    Convert binary mask to polygon coordinates (YOLO format).
    
    Args:
        mask: Binary mask array (H, W)
        min_points: Minimum number of polygon points
    
    Returns:
        polygon_coords: List of normalized coordinates [x1, y1, x2, y2, ...]
        status: Success or error message
    """
    # Ensure mask is binary uint8
    if mask.dtype != np.uint8:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask
    
    # Find contours
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return [], "No contours found"
    
    # Select largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Subsample if too many points
    if len(largest_contour) > 50:
        step = len(largest_contour) // 50
        working_contour = largest_contour[::step]
    else:
        working_contour = largest_contour
    
    # Get image dimensions
    height, width = mask.shape
    
    # Convert to normalized coordinates
    polygon_coords = []
    for point in working_contour:
        x, y = point[0]
        norm_x = x / width
        norm_y = y / height
        polygon_coords.extend([norm_x, norm_y])
    
    if len(polygon_coords) < min_points:
        return [], f"Too few valid points: {len(polygon_coords)//2}"
    
    return polygon_coords, "SUCCESS"


# =============================================================================
# YOLO Format Conversion
# =============================================================================

def convert_masks_to_yolo(masks, target_class, min_area):
    """
    Convert masks to YOLO format.
    
    Args:
        masks: List of mask arrays
        target_class: Class index to assign
        min_area: Minimum area threshold
    
    Returns:
        List of YOLO format annotation strings
    """
    yolo_annotations = []
    
    for mask in masks:
        # Check area
        area = mask.sum()
        if area < min_area:
            continue
        
        # Convert to polygon
        polygon_coords, status = mask_to_polygon(mask)
        
        if status != "SUCCESS" or len(polygon_coords) < 6:
            continue
        
        # Format for YOLO
        coords_str = ' '.join([f'{coord:.6f}' for coord in polygon_coords])
        yolo_line = f'{target_class} {coords_str}'
        yolo_annotations.append(yolo_line)
    
    return yolo_annotations


# =============================================================================
# Visualization Functions
# =============================================================================

def create_mask_overlay(image, masks, target_class, alpha=0.4):
    """Create visualization with mask overlays."""
    result_image = image.copy()
    
    # Define colors for different classes (BGR format)
    class_colors = {
        0: (0, 255, 0),   # Green for Chromis chromis
        1: (255, 0, 0),   # Blue for Coris julis
    }
    
    color = class_colors.get(target_class, (255, 255, 255))
    
    for mask in masks:
        # Create colored overlay
        colored_mask = np.zeros_like(result_image)
        colored_mask[mask > 0] = color
        
        # Apply transparency
        mask_area = mask > 0
        result_image[mask_area] = cv2.addWeighted(
            result_image[mask_area],
            1.0 - alpha,
            colored_mask[mask_area],
            alpha,
            0
        )
    
    return result_image


# =============================================================================
# Main Inference Function
# =============================================================================

def main():
    """Main inference function."""
    # Parse arguments and update globals
    args = parse_arguments()
    update_global_variables(args)
    
    print("\n" + "="*60)
    print("SAM2 INFERENCE CONFIGURATION - USING AUTOMATIC MASK GENERATOR")
    print("="*60)
    print(f"Using device: {DEVICE}")
    print(f"Target class: {CLASS_NAMES.get(TARGET_CLASS_INDEX, 'unknown')}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Input images folder: {INPUT_IMAGES_FOLDER}")
    print(f"Output labels folder: {OUTPUT_LABELS_FOLDER}")
    print(f"Output images folder: {OUTPUT_IMAGES_FOLDER}")
    print(f"\nMask Generator Parameters:")
    print(f"  Points per side: {POINTS_PER_SIDE}")
    print(f"  Pred IoU threshold: {PRED_IOU_THRESH}")
    print(f"  Stability score threshold: {STABILITY_SCORE_THRESH}")
    print(f"  Min mask region area: {MIN_MASK_REGION_AREA}")
    print("="*60 + "\n")
    
    # Create output folders
    os.makedirs(OUTPUT_LABELS_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_IMAGES_FOLDER, exist_ok=True)
    
    # Load SAM2 automatic mask generator
    print("Loading SAM2 Automatic Mask Generator...")
    mask_generator = load_sam2_mask_generator(SAM2_MODEL_ID, MODEL_PATH, DEVICE)
    
    # Get minimum area for this class
    min_area = MIN_MASK_AREA_ORIGINAL[TARGET_CLASS_INDEX]
    
    # Process all images
    print("\nProcessing images...")
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    processed_count = 0
    
    for filename in os.listdir(INPUT_IMAGES_FOLDER):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(INPUT_IMAGES_FOLDER, filename)
            print(f"Processing: {filename}")
            
            try:
                # Load image
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Generate masks using automatic mask generator
                masks = generate_masks_for_image(mask_generator, image_rgb)
                
                print(f"  - Generated {len(masks)} meaningful masks")
                
                # Convert to YOLO format
                yolo_annotations = convert_masks_to_yolo(
                    masks, TARGET_CLASS_INDEX, min_area
                )
                
                # Save labels
                base_filename = os.path.splitext(filename)[0]
                label_filename = f"{base_filename}.txt"
                label_path = os.path.join(OUTPUT_LABELS_FOLDER,
                                          label_filename)
                
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                # Save visualization
                output_image_path = os.path.join(OUTPUT_IMAGES_FOLDER,
                                                  filename)
                overlay = create_mask_overlay(image, masks, TARGET_CLASS_INDEX)
                cv2.imwrite(output_image_path, overlay)
                
                processed_count += 1
                print(f"  - Saved {len(yolo_annotations)} instances")
                
            except Exception as e:
                print(f"  - Error processing {filename}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE!")
    print(f"{'='*60}")
    print(f"Images processed: {processed_count}")
    print(f"Labels saved in: {OUTPUT_LABELS_FOLDER}")
    print(f"Visualizations saved in: {OUTPUT_IMAGES_FOLDER}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
