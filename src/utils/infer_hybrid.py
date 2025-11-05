#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid inference: Mask R-CNN for detection/classification + SAM2 for
mask refinement.

This combines:
- Mask R-CNN: Provides bounding boxes and class predictions
- SAM2: Refines the segmentation masks

Author: Generated for fish segmentation project
Version: 3.1.0 - Updated to accept all parameters as arguments
"""

import argparse
import os
import torch
import cv2
import numpy as np
import json

# SAM2 imports
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    print("WARNING: SAM2 not available")
    SAM2_AVAILABLE = False

# Import shared model loader
from maskrcnn_loader import load_maskrcnn_model

# =============================================================================
# CONFIGURATION - NOW ALL FROM ARGUMENTS (no hardcoded defaults)
# =============================================================================

# These will be set from command-line arguments in main()
MASKRCNN_MODEL_PATH = None
SAM2_MODEL_PATH = None
SAM2_MODEL_ID = None
INPUT_IMAGES_FOLDER = None
OUTPUT_LABELS_FOLDER = None
OUTPUT_IMAGES_FOLDER = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Detection parameters (from arguments)
DETECTION_THRESHOLD = None
MIN_MASK_AREA = None
MASK_RESOLUTION = 112

# Class names (from arguments)
CLASS_NAMES = None

# Tolerance of polygon simplification (from arguments)
TOL = None

# Threshold to consider pixel is mask (0-1) (from arguments)
THRESHOLD = None

# Erode operation after getting masks (from arguments)
ERODE_ITERATIONS = None


# =============================================================================
# REMAINING FUNCTIONS (mask_to_yolo_polygon, etc.)
# =============================================================================

def mask_to_yolo_polygon(mask, class_id, tolerance=0):
    """Convert binary mask to YOLO polygon format."""
    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return None
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    if len(largest_contour) < 3:
        return None
    
    # Simplify polygon if tolerance > 0
    if tolerance > 0:
        epsilon = tolerance * cv2.arcLength(largest_contour, True)
        largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Normalize coordinates
    h, w = mask.shape
    polygon_points = []
    for point in largest_contour:
        x, y = point[0]
        polygon_points.extend([x / w, y / h])
    
    # Format as YOLO line
    yolo_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in polygon_points])
    return yolo_line


def load_sam2_models(model_paths, model_id, device):
    """Load multiple SAM2 models (one per class)."""
    if not SAM2_AVAILABLE:
        raise ImportError("SAM2 is not installed.")
    
    predictors = {}
    for class_idx, model_path in model_paths.items():
        print(f"Loading SAM2 model for class {class_idx}: {model_path}")
        predictor = SAM2ImagePredictor.from_pretrained(model_id)
        
        # Load fine-tuned weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            predictor.model.load_state_dict(state_dict)
            print(f"  Loaded fine-tuned weights")
        else:
            print(f"  WARNING: Fine-tuned weights not found, using base model")
        
        predictor.model.to(device)
        predictor.model.eval()
        predictors[class_idx] = predictor
    
    return predictors


def process_image_hybrid(image_path, maskrcnn_model, sam2_predictors, 
                        class_names, maskrcnn_img_size=2048,
                        detection_threshold=0.5, min_mask_area=100,
                        mask_threshold=0.99, erode_iterations=2):
    """Process single image with hybrid Mask R-CNN + SAM2."""
    # Load and preprocess image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]
    
    # Resize for Mask R-CNN
    image_resized = cv2.resize(image_rgb, (maskrcnn_img_size, maskrcnn_img_size))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    # Mask R-CNN inference
    with torch.no_grad():
        outputs = maskrcnn_model(image_tensor)[0]
    
    # Filter by confidence
    keep_indices = outputs['scores'] > detection_threshold
    boxes = outputs['boxes'][keep_indices].cpu().numpy()
    labels = outputs['labels'][keep_indices].cpu().numpy()
    scores = outputs['scores'][keep_indices].cpu().numpy()
    
    yolo_annotations = []
    
    # Process each detection by class
    for class_idx in sorted(sam2_predictors.keys()):
        class_detections = labels == class_idx
        if not class_detections.any():
            continue
        
        class_boxes = boxes[class_detections]
        sam2_predictor = sam2_predictors[class_idx]
        
        # Set image for SAM2
        sam2_predictor.set_image(image_rgb)
        
        for bbox in class_boxes:
            # Scale bbox back to original size
            x1, y1, x2, y2 = bbox
            x1 = int(x1 * orig_w / maskrcnn_img_size)
            y1 = int(y1 * orig_h / maskrcnn_img_size)
            x2 = int(x2 * orig_w / maskrcnn_img_size)
            y2 = int(y2 * orig_h / maskrcnn_img_size)
            
            # Clip to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)
            
            bbox_xyxy = np.array([x1, y1, x2, y2])
            
            # SAM2 refinement
            masks, _, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox_xyxy[None, :],
                multimask_output=False
            )
            
            refined_mask = masks[0].astype(np.uint8)
            
            # Apply erosion if specified
            if erode_iterations > 0:
                kernel = np.ones((3, 3), np.uint8)
                refined_mask = cv2.erode(refined_mask, kernel, 
                                        iterations=erode_iterations)
            
            # Filter by area
            if refined_mask.sum() < min_mask_area:
                continue
            
            # Convert to YOLO format
            yolo_line = mask_to_yolo_polygon(refined_mask, class_idx, 
                                            tolerance=TOL)
            
            if yolo_line is not None:
                yolo_annotations.append(yolo_line)
    
    return yolo_annotations, image_rgb


def create_visualization(image_rgb, yolo_annotations, output_path, class_names):
    """Create visualization of detections."""
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Colors for each class
    colors = {
        0: (0, 255, 0),    # Green for Chromis
        1: (255, 0, 0)     # Blue for Coris
    }
    
    for ann in yolo_annotations:
        parts = ann.split()
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        
        # Convert normalized coords to pixels
        h, w = image.shape[:2]
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * w)
            y = int(coords[i+1] * h)
            points.append([x, y])
        
        # Draw polygon
        points = np.array(points, dtype=np.int32)
        cv2.polylines(image, [points], True, colors.get(class_id, (0, 255, 255)), 2)
    
    cv2.imwrite(output_path, image)

def update_global_variables(args, sam2_model_paths, class_names):
    """Update global variables from parsed arguments."""
    global MASKRCNN_MODEL_PATH, SAM2_MODEL_PATH, SAM2_MODEL_ID
    global INPUT_IMAGES_FOLDER, OUTPUT_LABELS_FOLDER, OUTPUT_IMAGES_FOLDER
    global DETECTION_THRESHOLD, MIN_MASK_AREA, MASK_RESOLUTION
    global CLASS_NAMES, TOL, THRESHOLD, ERODE_ITERATIONS
    
    MASKRCNN_MODEL_PATH = args.maskrcnn_model
    SAM2_MODEL_PATH = sam2_model_paths  # This is now a dict
    SAM2_MODEL_ID = args.sam2_model_id
    INPUT_IMAGES_FOLDER = args.input_folder
    OUTPUT_LABELS_FOLDER = args.output_labels
    OUTPUT_IMAGES_FOLDER = args.output_images
    DETECTION_THRESHOLD = args.detection_threshold
    MIN_MASK_AREA = args.min_mask_area
    MASK_RESOLUTION = args.mask_resolution
    CLASS_NAMES = class_names
    TOL = args.polygon_tolerance
    THRESHOLD = args.mask_threshold
    ERODE_ITERATIONS = args.erode_iterations

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description='Hybrid inference with Mask R-CNN + Multi-SAM2'
    )
    parser.add_argument('--maskrcnn_model', type=str, required=True,
                       help='Path to Mask R-CNN model')
    parser.add_argument('--sam2_model_paths', type=str, required=True,
                       help='JSON dict of {class_idx: model_path}')
    parser.add_argument('--sam2_model_id', type=str,
                       default='facebook/sam2.1-hiera-large',
                       help='SAM2 model ID')
    parser.add_argument('--input_folder', type=str, required=True,
                       help='Input folder with images')
    parser.add_argument('--output_labels', type=str, required=True,
                       help='Output folder for labels')
    parser.add_argument('--output_images', type=str, required=True,
                       help='Output folder for images')
    parser.add_argument('--detection_threshold', type=float, default=0.5,
                       help='Detection confidence threshold')
    parser.add_argument('--maskrcnn_img_size', type=int, default=2048,
                       help='IMG_SIZE used during Mask R-CNN training')
    parser.add_argument('--class_names', type=str, required=True,
                       help='JSON dict of {class_idx: class_name}')
    parser.add_argument('--mask_resolution', type=int, default=MASK_RESOLUTION,
                        help='Mask resolution (28, 56, or 112)')
    parser.add_argument('--min_mask_area', type=int, default=100,
                       help='Minimum mask area in pixels')
    parser.add_argument('--polygon_tolerance', type=int, default=0,
                       help='Polygon simplification tolerance')
    parser.add_argument('--mask_threshold', type=float, default=0.99,
                       help='Threshold to consider pixel as mask (0-1)')
    parser.add_argument('--erode_iterations', type=int, default=2,
                       help='Number of erosion iterations')
    
    args = parser.parse_args()
    
    # Parse JSON arguments
    sam2_model_paths = json.loads(args.sam2_model_paths)
    sam2_model_paths = {int(k): v for k, v in sam2_model_paths.items()}
    
    class_names = json.loads(args.class_names)
    class_names = {int(k): v for k, v in class_names.items()}
    
    # Update global variables
    update_global_variables(args, sam2_model_paths, class_names)
    
    print("="*80)
    print("HYBRID INFERENCE: Mask R-CNN + Multi-SAM2")
    print("="*80)
    print(f"Mask R-CNN model: {args.maskrcnn_model}")
    print(f"Mask R-CNN IMG_SIZE: {args.maskrcnn_img_size}")
    print(f"SAM2 models:")
    for class_idx in sorted(sam2_model_paths.keys()):
        print(f"  Class {class_idx} ({class_names[class_idx]}): "
              f"{sam2_model_paths[class_idx]}")
    print(f"Detection threshold: {args.detection_threshold}")
    print(f"Min mask area: {args.min_mask_area}")
    print(f"Polygon tolerance: {args.polygon_tolerance}")
    print(f"Mask threshold: {args.mask_threshold}")
    print(f"Mask resolution: {args.mask_resolution}")
    print(f"Erode iterations: {args.erode_iterations}")
    print(f"Input folder: {args.input_folder}")
    print(f"Output labels: {args.output_labels}")
    print("="*80)
    
    # Load models
    num_classes = len(class_names) + 1  # +1 for background
    maskrcnn_model = load_maskrcnn_model(
        args.maskrcnn_model,
        num_classes,
        args.mask_resolution,
        DEVICE)
    sam2_predictors = load_sam2_models(sam2_model_paths, args.sam2_model_id, DEVICE)
    
    # Create output directories
    os.makedirs(args.output_labels, exist_ok=True)
    os.makedirs(args.output_images, exist_ok=True)
    
    # Process all images
    image_files = [f for f in os.listdir(args.input_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nProcessing {len(image_files)} images...")
    
    for img_file in image_files:
        print(f"Processing: {img_file}")
        image_path = os.path.join(args.input_folder, img_file)
        
        # Process image
        yolo_annotations, image_rgb = process_image_hybrid(
            image_path, maskrcnn_model, sam2_predictors, class_names,
            maskrcnn_img_size=args.maskrcnn_img_size,
            detection_threshold=args.detection_threshold,
            min_mask_area=args.min_mask_area,
            mask_threshold=args.mask_threshold,
            erode_iterations=args.erode_iterations
        )
        
        # Save labels
        basename = os.path.splitext(img_file)[0]
        label_path = os.path.join(args.output_labels, f"{basename}.txt")
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        # Save visualization
        viz_path = os.path.join(args.output_images, f"{basename}.jpg")
        create_visualization(image_rgb, yolo_annotations, viz_path, class_names)
        
        print(f"  Saved {len(yolo_annotations)} annotations")
    
    print("="*80)
    print("✓ HYBRID INFERENCE COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()