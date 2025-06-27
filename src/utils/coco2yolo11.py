#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script makes inference of Mask R-CNN and transforms the resulting
inferenced masks into YoloV11 format with configurable simplification options.
"""

import argparse
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import json
from shapely import Polygon

# =============================================================================
# DEFAULT GLOBAL CONFIGURATION VARIABLES
# =============================================================================

# Paths and files
MODEL_PATH = 'weights/m01_1.pth'
INPUT_IMAGES_FOLDER = 'dataset/test'
OUTPUT_LABELS_FOLDER = 'dataset/inference/labels'
OUTPUT_IMAGES_FOLDER = 'dataset/inference/images'

# Model parameters
NUM_CLASSES = 3
IMG_SIZE = 2048
CONFIDENCE_THRESHOLD = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MASK_RESOLUTION = 56
BASE_MIN_ANCHOR = 16

# Polygon conversion parameters - ENHANCED WITH NEW OPTIONS
MIN_CONTOUR_AREA = 50
SIMPLIFICATION_TOLERANCE = 1.4

# Simplification control options
ENABLE_SIMPLIFICATION = False
MIN_POLYGON_POINTS = 25

# Default class mapping - can be overridden by arguments from main.py
CLASS_MAPPING = {1: 0, 2: 1} # Default: COCO class IDs to YOLO class IDs

# =============================================================================
# ARGUMENT PARSING - ENHANCED WITH NEW OPTIONS
# =============================================================================

def parse_arguments():
    """Parse command-line arguments with new simplification options."""
    parser = argparse.ArgumentParser(
        description='Convert Mask R-CNN predictions to YOLOv11 format'
    )
    
    # Path arguments
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                       help='Path to the trained model weights')
    parser.add_argument('--input_images_folder', type=str,
                       default=INPUT_IMAGES_FOLDER,
                       help='Folder containing input images')
    parser.add_argument('--output_labels_folder', type=str,
                       default=OUTPUT_LABELS_FOLDER,
                       help='Folder to save YOLO format labels')
    parser.add_argument('--output_images_folder', type=str,
                       default=OUTPUT_IMAGES_FOLDER,
                       help='Folder to save visualization images')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES,
                       help='Number of classes (including background)')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE,
                       help='Image size for inference')
    parser.add_argument('--confidence_threshold', type=float,
                       default=CONFIDENCE_THRESHOLD,
                       help='Confidence threshold for predictions')
    parser.add_argument('--mask_resolution', type=int,
                       default=MASK_RESOLUTION,
                       help='Mask resolution (28, 56, or 112)')
    parser.add_argument('--base_min_anchor', type=int,
                       default=BASE_MIN_ANCHOR,
                       help='Base minimum anchor size')
    
    # Enhanced polygon conversion parameters
    parser.add_argument('--min_contour_area', type=int,
                       default=MIN_CONTOUR_AREA,
                       help='Minimum contour area to consider')
    parser.add_argument('--simplification_tolerance', type=float,
                       default=SIMPLIFICATION_TOLERANCE,
                       help='Tolerance for polygon simplification')
    
    # NEW: Simplification control arguments
    parser.add_argument('--enable_simplification', action='store_true',
                       default=ENABLE_SIMPLIFICATION,
                       help='Enable polygon simplification (default: True)')
    parser.add_argument('--disable_simplification', action='store_true',
                       help='Disable polygon simplification completely')
    parser.add_argument('--min_polygon_points', type=int,
                       default=MIN_POLYGON_POINTS,
                       help='Minimum number of points to maintain in polygon')
    
    # Class configuration argument (optional - will override default if provided)
    parser.add_argument('--class_mapping', type=str, default=None,
                       help='JSON string with COCO to YOLO class mapping')
    
    return parser.parse_args()

def update_global_variables(args):
    """Update global variables with command-line arguments."""
    global MODEL_PATH, INPUT_IMAGES_FOLDER, OUTPUT_LABELS_FOLDER
    global OUTPUT_IMAGES_FOLDER, NUM_CLASSES, IMG_SIZE, CONFIDENCE_THRESHOLD
    global MASK_RESOLUTION, BASE_MIN_ANCHOR, CLASS_MAPPING
    global MIN_CONTOUR_AREA, SIMPLIFICATION_TOLERANCE
    global ENABLE_SIMPLIFICATION, MIN_POLYGON_POINTS  # NEW globals
    
    # Update standard global variables
    MODEL_PATH = args.model_path
    INPUT_IMAGES_FOLDER = args.input_images_folder
    OUTPUT_LABELS_FOLDER = args.output_labels_folder
    OUTPUT_IMAGES_FOLDER = args.output_images_folder
    NUM_CLASSES = args.num_classes
    IMG_SIZE = args.img_size
    CONFIDENCE_THRESHOLD = args.confidence_threshold
    MASK_RESOLUTION = args.mask_resolution
    BASE_MIN_ANCHOR = args.base_min_anchor
    MIN_CONTOUR_AREA = args.min_contour_area
    SIMPLIFICATION_TOLERANCE = args.simplification_tolerance
    
    # NEW: Handle simplification control logic
    if args.disable_simplification:
        ENABLE_SIMPLIFICATION = False
        print("Simplification DISABLED via --disable_simplification")
    elif args.enable_simplification:
        ENABLE_SIMPLIFICATION = True
        print("Simplification ENABLED via --enable_simplification")
    else:
        ENABLE_SIMPLIFICATION = ENABLE_SIMPLIFICATION  # Use default
        print(f"Using default simplification setting: {ENABLE_SIMPLIFICATION}")
    
    # NEW: Update minimum polygon points
    MIN_POLYGON_POINTS = args.min_polygon_points
    print(f"Minimum polygon points set to: {MIN_POLYGON_POINTS}")
    
    # Update class configuration ONLY if provided as argument
    if args.class_mapping is not None:
        CLASS_MAPPING = json.loads(args.class_mapping)
        # Convert string keys to integers (JSON converts int keys to strings)
        CLASS_MAPPING = {int(k): v for k, v in CLASS_MAPPING.items()}
        print(f"Updated CLASS_MAPPING from arguments: {CLASS_MAPPING}")
    else:
        print(f"Using default CLASS_MAPPING: {CLASS_MAPPING}")

# =============================================================================
# ENHANCED SIMPLIFICATION METHOD WITH MINIMUM POINTS CONTROL
# =============================================================================

def simplify_polygon_with_configurable_options(points, image_dimensions, 
                                              tolerance=2.0, 
                                              enable_simplification=True,
                                              min_points=25):
    """
    Enhanced simplification with full control over the process.
    
    Args:
        points: List of (x, y) coordinate tuples (normalized 0-1)
        image_dimensions: Tuple (width, height) of image
        tolerance: Simplification tolerance
        enable_simplification: Whether to apply simplification at all
        min_points: Minimum number of points to maintain
    
    Returns:
        List of simplified (x, y) coordinate tuples (normalized 0-1)
    """
    initial_point_count = len(points)
    print(f"Initial points: {initial_point_count}")
    
    # OPTION 1: Skip simplification entirely if disabled
    if not enable_simplification:
        print("Simplification DISABLED - returning original points")
        return points
    
    # OPTION 2: Skip simplification if already below minimum threshold
    if initial_point_count <= min_points:
        print(f"Already at or below minimum points ({min_points}) - "
              f"skipping simplification")
        return points
    
    # OPTION 3: Apply simplification with minimum points protection
    try:
        print(f"Applying simplification (tolerance={tolerance})")
        
        # Scale normalized coordinates (0-1) to image dimensions
        # (Exact code from your label_simplify.py lines 138-141)
        scaled_points = [
            (x * image_dimensions[0], y * image_dimensions[1])
            for x, y in points
        ]
        
        # Create Shapely polygon (your exact method from line 144)
        polygon = Polygon(scaled_points)
        
        # Apply simplification (your exact method from lines 175-178)
        simplified_polygon = polygon.simplify(
            tolerance=tolerance,
            preserve_topology=True
        )
        
        final_points = list(simplified_polygon.exterior.coords)[0:-1]  # Remove duplicate
        
        # NEW: Check if simplification resulted in too few points
        if len(final_points) < min_points:
            print(f"Warning: Simplification reduced points to {len(final_points)}, "
                  f"which is below minimum {min_points}")
            print("Trying with reduced tolerance...")
            
            # Try with progressively smaller tolerance values
            for reduced_tolerance in [tolerance * 0.5, tolerance * 0.25, tolerance * 0.1]:
                try:
                    reduced_simplified = polygon.simplify(
                        tolerance=reduced_tolerance,
                        preserve_topology=True
                    )
                    reduced_points = list(reduced_simplified.exterior.coords)[0:-1]
                    
                    if len(reduced_points) >= min_points:
                        print(f"Success with reduced tolerance {reduced_tolerance}: "
                              f"{len(reduced_points)} points")
                        final_points = reduced_points
                        break
                except Exception as e:
                    print(f"Error with tolerance {reduced_tolerance}: {e}")
                    continue
            
            # If still too few points, return original
            if len(final_points) < min_points:
                print(f"Could not maintain minimum {min_points} points, "
                      f"returning original {initial_point_count} points")
                return points
        
        # Safety check: ensure at least 3 points for valid polygon
        if len(final_points) < 3:
            print(f"Critical: Simplified to {len(final_points)} points, "
                  f"returning original polygon")
            return points
        
        # Convert back to normalized coordinates (0-1)
        # (your exact method from lines 193-196)
        new_points = []
        for x, y in final_points:
            new_points.append((x / image_dimensions[0], y / image_dimensions[1]))
        
        print(f"Final simplified points: {len(final_points)} "
              f"(reduced from {initial_point_count})")
        
        return new_points
        
    except Exception as e:
        # If there's an error, keep the original (your exact logic)
        print(f"Error during simplification: {e}")
        print("Returning original points")
        return points

def mask_to_polygon(mask, image_dimensions, min_area=50, 
                   simplification_tolerance=2.0,
                   enable_simplification=True,
                   min_polygon_points=25):
    """
    Converts a binary mask to polygon coordinates with enhanced simplification control.
    Now selects the contour with the LARGEST PERIMETER instead of middle by area.
    
    Args:
        mask: Binary mask numpy array
        image_dimensions: Tuple (width, height) of original image
        min_area: Minimum area to consider a valid contour
        simplification_tolerance: Tolerance for polygon simplification
        enable_simplification: Whether to apply simplification
        min_polygon_points: Minimum points to maintain in polygon
    
    Returns:
        List of polygon coordinates in format [x1, y1, x2, y2, ...]
    """
    # Convert mask to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours with full precision (no approximation)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    
    if not contours:
        return []
    
    # Filter contours by minimum area
    valid_contours = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            valid_contours.append(contour)
    
    if not valid_contours:
        return []
    
    # Calculate perimeter (arc length) for each valid contour
    contour_perimeters = []
    for i, contour in enumerate(valid_contours):
        perimeter = cv2.arcLength(contour, True)  # True = closed contour
        area = cv2.contourArea(contour)
        num_points = len(contour)
        contour_perimeters.append((i, perimeter, area, num_points))
        print(f"Contour {i}: perimeter={perimeter:.1f}, area={area:.1f}, points={num_points}")
    
    # Sort by perimeter (descending) and select the largest
    contour_perimeters.sort(key=lambda x: x[1], reverse=True)  # Sort by perimeter
    largest_perimeter_index = contour_perimeters[0][0]
    selected_contour = valid_contours[largest_perimeter_index]
    
    # Debug information
    selected_perimeter = contour_perimeters[0][1]
    selected_area = contour_perimeters[0][2]
    selected_points = contour_perimeters[0][3]
    
    print(f"Selected contour with LARGEST PERIMETER:")
    print(f"  - Index: {largest_perimeter_index}")
    print(f"  - Perimeter: {selected_perimeter:.1f}")
    print(f"  - Area: {selected_area:.1f}")
    print(f"  - Points: {selected_points}")
    
    # Convert contour to normalized points
    mask_height, mask_width = mask.shape
    points = []
    for point in selected_contour:
        x, y = point[0]
        # Normalize to 0-1 range
        norm_x = x / mask_width
        norm_y = y / mask_height
        points.append((norm_x, norm_y))
    
    print(f"Converted to {len(points)} normalized points")
    
    # Apply enhanced simplification method with all options
    simplified_points = simplify_polygon_with_configurable_options(
        points, 
        image_dimensions, 
        tolerance=simplification_tolerance,
        enable_simplification=enable_simplification,
        min_points=min_polygon_points
    )
    
    print(f"Final polygon has {len(simplified_points)} points")
    
    # Convert to flat coordinate list
    polygon_coords = []
    for x, y in simplified_points:
        polygon_coords.extend([x, y])
    
    return polygon_coords

# =============================================================================
# AUXILIARY FUNCTIONS
# =============================================================================

def get_anchor_sizes(img_size, base_img_size=1280, base_min_anchor=16):
    """
    Calculate anchor sizes maintaining powers-of-2 progression.
    
    Args:
        img_size: Target IMG_SIZE
        base_img_size: Baseline IMG_SIZE (default: 1280)
        base_min_anchor: Minimum anchor size at baseline (default: 16)
    
    Returns:
        Tuple of anchor sizes following geometric progression
    """
    scale_factor = img_size / base_img_size
    min_anchor = int(round(base_min_anchor * scale_factor))
    # Generate anchors as powers of 2: 1x, 2x, 4x, 8x, 16x
    anchor_sizes = tuple(min_anchor * (2**i) for i in range(5))
    return anchor_sizes

def get_model_instance_segmentation(num_classes, mask_resolution=56):
    """
    Creates Mask R-CNN model with customizable mask resolution for small objects.
    Must match exactly with the training configuration.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Calculate ROI pooling size (typically half of final mask resolution)
    roi_pool_size = mask_resolution // 2
    
    # Increase ROI pooling resolution for masks
    model.roi_heads.mask_roi_pool.output_size = (roi_pool_size, roi_pool_size)
    print(f"Mask ROI pool size set to: {roi_pool_size}×{roi_pool_size}")
    print(f"Target mask resolution: {mask_resolution}×{mask_resolution}")
    
    # Optimize anchor generator for small objects using dynamic calculation
    anchor_generator = torchvision.models.detection.anchor_utils.AnchorGenerator(
        sizes=tuple((size,) for size in get_anchor_sizes(IMG_SIZE, base_min_anchor=BASE_MIN_ANCHOR)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5  # 5 tuples for 5 feature maps
    )
    model.rpn.anchor_generator = anchor_generator
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor with high-resolution version
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = HighResMaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes, mask_resolution
    )
    
    print(f"Model configured for {mask_resolution}×{mask_resolution} masks")
    return model

def get_inference_transforms():
    """
    Transformations for inference (without augmentations).
    """
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(
            min_height=IMG_SIZE,
            min_width=IMG_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
            p=1.0
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

class HighResMaskRCNNPredictor(torch.nn.Module):
    def __init__(self, in_channels, dim_reduced, num_classes, mask_size):
        super().__init__()
        self.mask_size = mask_size
        
        # Determine number of conv layers based on resolution
        if mask_size <= 28:
            # Standard configuration for 28×28
            self.conv5_mask = torch.nn.ConvTranspose2d(dim_reduced, 
                                                      dim_reduced, 2, 2, 0)
            self.relu = torch.nn.ReLU(inplace=True)
            self.mask_fcn_logits = torch.nn.Conv2d(dim_reduced, 
                                                  num_classes, 1, 1, 0)
        elif mask_size <= 56:
            # Enhanced configuration for 56×56
            self.conv5_mask = torch.nn.ConvTranspose2d(dim_reduced, 
                                                      dim_reduced, 2, 2, 0)
            self.relu1 = torch.nn.ReLU(inplace=True)
            self.conv6_mask = torch.nn.ConvTranspose2d(dim_reduced, 
                                                      dim_reduced, 2, 2, 0)
            self.relu2 = torch.nn.ReLU(inplace=True)
            self.mask_fcn_logits = torch.nn.Conv2d(dim_reduced, 
                                                  num_classes, 1, 1, 0)
        elif mask_size <= 112:
            # Advanced configuration for 112×112
            self.conv5_mask = torch.nn.ConvTranspose2d(dim_reduced, 
                                                      dim_reduced, 2, 2, 0)
            self.relu1 = torch.nn.ReLU(inplace=True)
            self.conv6_mask = torch.nn.ConvTranspose2d(dim_reduced, 
                                                      dim_reduced, 2, 2, 0)
            self.relu2 = torch.nn.ReLU(inplace=True)
            self.conv7_mask = torch.nn.ConvTranspose2d(dim_reduced, 
                                                      dim_reduced, 2, 2, 0)
            self.relu3 = torch.nn.ReLU(inplace=True)
            self.mask_fcn_logits = torch.nn.Conv2d(dim_reduced, 
                                                  num_classes, 1, 1, 0)
        else:
            raise ValueError(f"Mask resolution {mask_size} not supported. "
                           f"Use 28, 56, or 112.")
        
        # Initialize weights
        for name, param in self.named_parameters():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param, mode="fan_out", 
                                            nonlinearity="relu")
            elif "bias" in name:
                torch.nn.init.constant_(param, 0)

    def forward(self, x):
        if self.mask_size <= 28:
            x = self.conv5_mask(x)
            x = self.relu(x)
            x = self.mask_fcn_logits(x)
        elif self.mask_size <= 56:
            x = self.conv5_mask(x)
            x = self.relu1(x)
            x = self.conv6_mask(x)
            x = self.relu2(x)
            x = self.mask_fcn_logits(x)
        elif self.mask_size <= 112:
            x = self.conv5_mask(x)
            x = self.relu1(x)
            x = self.conv6_mask(x)
            x = self.relu2(x)
            x = self.conv7_mask(x)
            x = self.relu3(x)
            x = self.mask_fcn_logits(x)
        return x

def calculate_transformation_params(original_height, original_width,
                                  target_size=IMG_SIZE):
    """
    Calculates the parameters needed to reverse the LongestMaxSize +
    PadIfNeeded transformation.
    
    Args:
        original_height: Original image height
        original_width: Original image width
        target_size: Target size used in transformation (default: IMG_SIZE)
    
    Returns:
        dict: Parameters for transformation reversal
    """
    # Calculate scale factor (same as LongestMaxSize)
    scale_factor = target_size / max(original_height, original_width)
    
    # Calculate scaled dimensions
    scaled_height = int(original_height * scale_factor)
    scaled_width = int(original_width * scale_factor)
    
    # Calculate padding offsets
    pad_top = (target_size - scaled_height) // 2
    pad_left = (target_size - scaled_width) // 2
    
    return {
        'scale_factor': scale_factor,
        'scaled_height': scaled_height,
        'scaled_width': scaled_width,
        'pad_top': pad_top,
        'pad_left': pad_left,
        'original_height': original_height,
        'original_width': original_width
    }

def reverse_mask_transformation(mask, transform_params):
    """
    Reverses the LongestMaxSize + PadIfNeeded transformation on a mask.
    
    Args:
        mask: Mask tensor of shape (H, W) at target_size (1024x1024)
        transform_params: Dictionary with transformation parameters
    
    Returns:
        numpy.ndarray: Mask resized to original image dimensions
    """
    # Convert to numpy if tensor
    if torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    # Step 1: Remove padding (crop to scaled size)
    pad_top = transform_params['pad_top']
    pad_left = transform_params['pad_left']
    scaled_height = transform_params['scaled_height']
    scaled_width = transform_params['scaled_width']
    
    # Crop the mask to remove padding
    cropped_mask = mask_np[
        pad_top:pad_top + scaled_height,
        pad_left:pad_left + scaled_width
    ]
    
    # Step 2: Resize back to original size
    original_height = transform_params['original_height']
    original_width = transform_params['original_width']
    
    resized_mask = cv2.resize(
        cropped_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )
    
    return resized_mask

def normalize_polygon(polygon_coords, image_height, image_width):
    """
    Normalizes polygon coordinates to values between 0 and 1.
    """
    normalized_coords = []
    for i in range(0, len(polygon_coords), 2):
        x = polygon_coords[i] / image_width
        y = polygon_coords[i + 1] / image_height
        normalized_coords.extend([x, y])
    return normalized_coords

def inference_on_image(model, image_path, transforms, original_size):
    """
    Performs inference on a specific image.
    
    Args:
        model: Loaded Mask R-CNN model
        image_path: Path to image
        transforms: Albumentations transformations
        original_size: Tuple (height, width) of original image size
    
    Returns:
        Dictionary with inference results
    """
    # Load and process image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transformations
    transformed = transforms(image=image_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
    
    # Inference
    model.eval()
    with torch.no_grad():
        predictions = model(input_tensor)
    
    # Process results
    pred = predictions[0]
    
    # Filter by confidence threshold
    high_conf_indices = pred['scores'] > CONFIDENCE_THRESHOLD
    
    filtered_results = {
        'boxes': pred['boxes'][high_conf_indices].cpu().numpy(),
        'labels': pred['labels'][high_conf_indices].cpu().numpy(),
        'scores': pred['scores'][high_conf_indices].cpu().numpy(),
        'masks': pred['masks'][high_conf_indices].cpu().numpy()
    }
    
    return filtered_results, original_size
    

def convert_to_yolo_format(predictions, original_size):
    """
    Converts Mask R-CNN predictions to YoloV11 format using custom simplification.
    
    Args:
        predictions: Dictionary with inference results
        original_size: Tuple (height, width) of original size
    
    Returns:
        List of strings in YoloV11 format
    """
    yolo_annotations = []
    original_height, original_width = original_size
    
    # Calculate transformation parameters
    transform_params = calculate_transformation_params(
        original_height, original_width
    )
    
    masks = predictions['masks']
    labels = predictions['labels']
    scores = predictions['scores']
    
    for i in range(len(masks)):
        mask = masks[i][0]  # Take the first channel of the mask
        label = labels[i]
        score = scores[i]
        
        # Map class if necessary
        if label in CLASS_MAPPING:
            yolo_class = CLASS_MAPPING[label]
        else:
            continue  # Skip if class is not mapped
        
        # Properly reverse the transformation
        mask_original = reverse_mask_transformation(mask, transform_params)
        
        # Convert mask to polygon using custom simplification
        polygon_coords = mask_to_polygon(
            mask_original,
            image_dimensions=(original_width, original_height),
            min_area=MIN_CONTOUR_AREA,
            simplification_tolerance=SIMPLIFICATION_TOLERANCE,
            enable_simplification=ENABLE_SIMPLIFICATION,
            min_polygon_points=MIN_POLYGON_POINTS
        )
        
        if len(polygon_coords) < 6:  # Need at least 3 points (6 coordinates)
            continue
        
        # Format for YoloV11
        coords_str = ' '.join([f'{coord:.6f}' for coord in polygon_coords])
        yolo_line = f'{yolo_class} {coords_str}'
        yolo_annotations.append(yolo_line)
    
    return yolo_annotations

def create_mask_overlay(image, predictions, original_size,
                       alpha: float = 0.6) -> np.ndarray:
    """
    Creates a transparent overlay of predicted masks on the original image.
    
    Args:
        image: Original image as numpy array (H, W, 3)
        predictions: Dictionary with inference results
        original_size: Tuple (height, width) of original size
        alpha: Transparency level for mask overlay (0.0 to 1.0)
    
    Returns:
        np.ndarray: Image with transparent mask overlays
    """
    # Create a copy of the original image
    result_image = image.copy()
    original_height, original_width = original_size
    
    # Calculate transformation parameters
    transform_params = calculate_transformation_params(
        original_height, original_width
    )
    
    # Define colors for different classes (BGR format for OpenCV)
    class_colors = {
        0: (0, 255, 0),    # Green for Chromis
        1: (255, 0, 0),    # Blue for Coris
        2: (0, 0, 255),    # Red for additional class if needed
    }
    
    masks = predictions['masks']
    labels = predictions['labels']
    
    # Process each mask
    for i in range(len(masks)):
        mask = masks[i][0]  # Take the first channel
        label = labels[i]
        
        # Map class if necessary
        if label in CLASS_MAPPING:
            yolo_class = CLASS_MAPPING[label]
        else:
            continue
        
        # Properly reverse the transformation
        mask_original = reverse_mask_transformation(mask, transform_params)
        
        # Create binary mask
        binary_mask = (mask_original > 0.5).astype(np.uint8)
        
        # Get color for this class
        color = class_colors.get(yolo_class, (255, 255, 255))
        
        # Create colored mask overlay
        colored_mask = np.zeros_like(result_image)
        colored_mask[binary_mask == 1] = color
        
        # Apply transparency only where mask exists
        mask_area = binary_mask == 1
        result_image[mask_area] = cv2.addWeighted(
            result_image[mask_area],
            1.0 - alpha,
            colored_mask[mask_area],
            alpha,
            0
        )
    
    return result_image

def save_visualization(image_path: str, predictions: dict,
                      original_size: tuple, output_path: str,
                      alpha: float = 0.6) -> None:
    """
    Saves an image with transparent mask overlays only.
    
    Args:
        image_path: Path to the original image
        predictions: Dictionary with inference results
        original_size: Tuple (height, width) of original size
        output_path: Path where to save the visualization
        alpha: Transparency level (0.0 = invisible, 1.0 = opaque)
    
    Returns:
        None
    """
    # Load original image
    original_image = cv2.imread(image_path)
    
    # Create overlay with transparent masks only
    overlay_image = create_mask_overlay(
        original_image, predictions, original_size, alpha
    )
    
    # Save the visualization
    cv2.imwrite(output_path, overlay_image)

# =============================================================================
# MAIN FUNCTION WITH ENHANCED ARGUMENT SUPPORT
# =============================================================================

def main():
    """
    Main function with enhanced simplification control options.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Update global variables with arguments
    update_global_variables(args)
    
    print(f"Using device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Using class mapping: {CLASS_MAPPING}")
    print(f"Simplification enabled: {ENABLE_SIMPLIFICATION}")
    print(f"Simplification tolerance: {SIMPLIFICATION_TOLERANCE}")
    print(f"Minimum polygon points: {MIN_POLYGON_POINTS}")
    print(f"Minimum contour area: {MIN_CONTOUR_AREA}")
    print(f"Input images folder: {INPUT_IMAGES_FOLDER}")
    print(f"Output labels folder: {OUTPUT_LABELS_FOLDER}")
    print(f"Output images folder: {OUTPUT_IMAGES_FOLDER}")
    
    # Create output folders
    os.makedirs(OUTPUT_LABELS_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_IMAGES_FOLDER, exist_ok=True)
    
    # Load trained model
    print("Loading Mask R-CNN model...")
    model = get_model_instance_segmentation(NUM_CLASSES, MASK_RESOLUTION)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Get transformations
    transforms = get_inference_transforms()
    
    # Process all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    processed_count = 0
    
    for filename in os.listdir(INPUT_IMAGES_FOLDER):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(INPUT_IMAGES_FOLDER, filename)
            print(f"Processing: {filename}")
            
            # Get original image size
            with Image.open(image_path) as img:
                original_width, original_height = img.size
            original_size = (original_height, original_width)
            
            try:
                # Perform inference
                predictions, _ = inference_on_image(
                    model, image_path, transforms, original_size
                )
                
                # Convert to YoloV11 format with enhanced control
                yolo_annotations = convert_to_yolo_format(
                    predictions, original_size
                )
                
                # Save labels
                base_filename = os.path.splitext(filename)[0]
                label_filename = f"{base_filename}.txt"
                label_path = os.path.join(OUTPUT_LABELS_FOLDER, label_filename)
                
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
                
                # Save visualization
                output_image_path = os.path.join(
                    OUTPUT_IMAGES_FOLDER, filename
                )
                
                save_visualization(
                    image_path,
                    predictions,
                    original_size,
                    output_image_path,
                    alpha=0.4  # Adjust transparency level (0.0-1.0)
                )
                
                processed_count += 1
                print(f" - Detected {len(yolo_annotations)} instances")
                
            except Exception as e:
                print(f" - Error processing {filename}: {str(e)}")
    
    print(f"\nProcessing completed!")
    print(f"Images processed: {processed_count}")
    print(f"Labels saved in: {OUTPUT_LABELS_FOLDER}")
    print(f"Visualizations saved in: {OUTPUT_IMAGES_FOLDER}")

# =============================================================================
# EXECUTE SCRIPT
# =============================================================================

if __name__ == "__main__":
    main()