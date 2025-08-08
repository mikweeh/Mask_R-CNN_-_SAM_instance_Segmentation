#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script makes inference of Mask R-CNN and transforms the resulting
inferenced masks into YoloV11 format with improved mask processing.
"""

# TO DO: get_model_instance_segmentation() is used here and also
# at train.py (and most recently at pruebas_masks), so it should be placed
# as a helper function and imported instead of being duplicated

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import json

# =============================================================================
# DEFAULT GLOBAL CONFIGURATION VARIABLES
# =============================================================================

# Paths and files
MODEL_PATH = 'weights/m01_x.pth'
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

# NEW: Class-specific area filtering (from pruebas_masks.py)
MIN_MASK_AREA_ORIGINAL = [256, 100]  # Minimum area in original image pixels 
                                     # (256 for chromis and 100 for coris)
MIN_CONTOUR_AREA = 64  # Minimum pixels in inference space (2048×2048)

# Default class mapping - can be overridden by arguments from main.py
CLASS_MAPPING = {1: 0, 2: 1}  # Default: COCO class IDs to YOLO class IDs

# =============================================================================
# ARGUMENT PARSING - UPDATED
# =============================================================================

def parse_arguments():
    """Parse command-line arguments with updated options."""
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
    
    # NEW: Class-specific area thresholds
    parser.add_argument('--min_contour_area', type=int,
                       default=MIN_CONTOUR_AREA,
                       help='Minimum contour area in inference space')
    
    # NOTE: Removed simplification arguments as they're no longer used
    
    # Class configuration argument
    parser.add_argument('--class_mapping', type=str, default=None,
                       help='JSON string with COCO to YOLO class mapping')
    
    return parser.parse_args()

def update_global_variables(args):
    """Update global variables with command-line arguments."""
    global MODEL_PATH, INPUT_IMAGES_FOLDER, OUTPUT_LABELS_FOLDER
    global OUTPUT_IMAGES_FOLDER, NUM_CLASSES, IMG_SIZE, CONFIDENCE_THRESHOLD
    global MASK_RESOLUTION, BASE_MIN_ANCHOR, CLASS_MAPPING
    global MIN_CONTOUR_AREA
    
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
    
    # Update class configuration ONLY if provided as argument
    if args.class_mapping is not None:
        CLASS_MAPPING = json.loads(args.class_mapping)
        # Convert string keys to integers
        CLASS_MAPPING = {int(k): v for k, v in CLASS_MAPPING.items()}
        print(f"Updated CLASS_MAPPING from arguments: {CLASS_MAPPING}")
    else:
        print(f"Using default CLASS_MAPPING: {CLASS_MAPPING}")

# =============================================================================
# AUXILIARY FUNCTIONS (UNCHANGED)
# =============================================================================

def get_anchor_sizes(img_size, base_img_size=1280, base_min_anchor=16):
    """Calculate anchor sizes maintaining powers-of-2 progression."""
    scale_factor = img_size / base_img_size
    min_anchor = int(round(base_min_anchor * scale_factor))
    anchor_sizes = tuple(min_anchor * (2**i) for i in range(5))
    return anchor_sizes

def get_model_instance_segmentation(num_classes, mask_resolution=28):
    """
    Creates Mask R-CNN model with customizable mask resolution for small objects.
    
    Args:
        num_classes: Number of classes including background
        mask_resolution: Final mask output resolution.
                        Common values: 28 (default), 56, 112
    
    Returns:
        Enhanced Mask R-CNN model with higher resolution masks
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Calculate ROI pooling size (typically half of final mask resolution)
    if mask_resolution <= 56:
        roi_pool_size = mask_resolution // 2
    else:
        # Cap ROI pooling size at 28 for higher resolutions to save memory
        roi_pool_size = 28
    
    # Increase ROI pooling resolution for masks
    model.roi_heads.mask_roi_pool.output_size = (roi_pool_size, roi_pool_size)
    print(f"Mask ROI pool size set to: {roi_pool_size}×{roi_pool_size}")
    print(f"Target mask resolution: {mask_resolution}×{mask_resolution}")
    
    # Create custom mask predictor for higher resolution
    class HighResMaskRCNNPredictor(nn.Module):
        def __init__(self, in_channels, dim_reduced, num_classes, mask_size):
            super().__init__()
            self.mask_size = mask_size

            # Reduce hidden dimension for high-resolution masks
            if mask_size <= 56:
                hidden_dim = dim_reduced
            else:
                hidden_dim = dim_reduced // 2  # 256 -> 128 for memory

            # Determine number of conv layers based on resolution
            if mask_size <= 28:
                # Standard configuration for 28×28
                self.conv5_mask = nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)
                self.relu = nn.ReLU(inplace=True)
                self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
                
            elif mask_size <= 56:
                # Enhanced configuration for 56×56
                self.conv5_mask = nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)
                self.relu1 = nn.ReLU(inplace=True)
                self.conv6_mask = nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)
                self.relu2 = nn.ReLU(inplace=True)
                self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
                
            elif mask_size <= 112:
                # Advanced configuration for 112×112
                self.conv5_mask = nn.ConvTranspose2d(dim_reduced, hidden_dim, 2, 2, 0)
                self.relu1 = nn.ReLU(inplace=True)
                self.conv6_mask = nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, 2, 0)
                self.relu2 = nn.ReLU(inplace=True)
                self.conv7_mask = nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, 2, 0)
                self.relu3 = nn.ReLU(inplace=True)
                self.mask_fcn_logits = nn.Conv2d(hidden_dim, num_classes, 1, 1, 0)
                
            else:
                raise ValueError(f"Mask resolution {mask_size} not supported. Use 28, 56, or 112.")
            
            # Initialize weights
            for name, param in self.named_parameters():
                if "weight" in name:
                    nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
                elif "bias" in name:
                    nn.init.constant_(param, 0)
        
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
                x = self.relu1(self.conv5_mask(x))
                torch.cuda.empty_cache()
                x = self.relu2(self.conv6_mask(x))
                torch.cuda.empty_cache()
                x = self.relu3(self.conv7_mask(x))
                torch.cuda.empty_cache()
                x = self.mask_fcn_logits(x)
            
                # Crop to exact size if needed
                if x.shape[-1] != self.mask_size:
                    x = F.interpolate(x, size=(self.mask_size, self.mask_size), 
                                    mode='bilinear', align_corners=False)
            return x
    
    # Optimize anchor generator for small objects
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
    """Transformations for inference (without augmentations)."""
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
        
        if mask_size <= 28:
            self.conv5_mask = torch.nn.ConvTranspose2d(dim_reduced,
                                                      dim_reduced, 2, 2, 0)
            self.relu = torch.nn.ReLU(inplace=True)
            self.mask_fcn_logits = torch.nn.Conv2d(dim_reduced,
                                                  num_classes, 1, 1, 0)
        elif mask_size <= 56:
            self.conv5_mask = torch.nn.ConvTranspose2d(dim_reduced,
                                                      dim_reduced, 2, 2, 0)
            self.relu1 = torch.nn.ReLU(inplace=True)
            self.conv6_mask = torch.nn.ConvTranspose2d(dim_reduced,
                                                      dim_reduced, 2, 2, 0)
            self.relu2 = torch.nn.ReLU(inplace=True)
            self.mask_fcn_logits = torch.nn.Conv2d(dim_reduced,
                                                  num_classes, 1, 1, 0)
        elif mask_size <= 112:
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
    """Calculate parameters for transformation reversal."""
    scale_factor = target_size / max(original_height, original_width)
    scaled_height = int(original_height * scale_factor)
    scaled_width = int(original_width * scale_factor)
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
    """Reverse the LongestMaxSize + PadIfNeeded transformation on a mask."""
    if torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    # Remove padding
    pad_top = transform_params['pad_top']
    pad_left = transform_params['pad_left']
    scaled_height = transform_params['scaled_height']
    scaled_width = transform_params['scaled_width']
    
    cropped_mask = mask_np[
        pad_top:pad_top + scaled_height,
        pad_left:pad_left + scaled_width
    ]
    
    # Resize back to original
    original_height = transform_params['original_height']
    original_width = transform_params['original_width']
    
    resized_mask = cv2.resize(
        cropped_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )
    
    return resized_mask

# =============================================================================
# NEW: UPDATED INFERENCE WITH FILTERING (from pruebas_masks.py)
# =============================================================================

def perform_single_inference_with_filtering(model, image_path, transforms):
    """
    Perform ONE inference and return filtered results.
    Based on pruebas_masks.py algorithm.
    """
    # Load and process image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    model.eval()
    transformed = transforms(image=image_rgb)
    tensor_image = transformed['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred = model(tensor_image)[0]
    
    # Filter by confidence
    high_conf_indices = pred['scores'] > CONFIDENCE_THRESHOLD
    
    # Get original image dimensions for area filtering
    with Image.open(image_path) as img:
        original_width, original_height = img.size
    
    transform_params = calculate_transformation_params(original_height,
                                                     original_width)
    
    # Filter by mask area in original image space
    valid_indices = []
    for i, idx in enumerate(high_conf_indices.nonzero().flatten()):
        mask_tensor = pred['masks'][idx]
        current_label = CLASS_MAPPING[pred['labels'][idx].item()]
        
        # Transform mask to original size and check area
        if mask_tensor.ndim == 3:
            mask = mask_tensor[0]
        else:
            mask = mask_tensor
        
        # Apply morphological closing to fill holes
        mask_np = mask.cpu().numpy()
        mask_binary = (mask_np > 0.5).astype(np.uint8)
        
        # Fill holes using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_filled = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        
        # Update the mask tensor with filled version
        pred['masks'][idx] = torch.from_numpy(
            mask_filled.astype(np.float32)
        ).unsqueeze(0)
        
        # Transform to original size and check area
        mask_original = reverse_mask_transformation(mask_filled,
                                                  transform_params)
        original_area = np.sum(mask_original > 0.5)
        
        if original_area >= MIN_MASK_AREA_ORIGINAL[current_label]:
            valid_indices.append(idx.item())
    
    # Create filtered results
    if valid_indices:
        valid_indices = torch.tensor(valid_indices,
                                   device=pred['masks'].device)
        filtered_results = {
            'boxes': pred['boxes'][valid_indices].cpu().numpy(),
            'labels': pred['labels'][valid_indices].cpu().numpy(),
            'scores': pred['scores'][valid_indices].cpu().numpy(),
            'masks': pred['masks'][valid_indices].cpu().numpy()
        }
    else:
        filtered_results = {
            'boxes': np.array([]),
            'labels': np.array([]),
            'scores': np.array([]),
            'masks': np.array([])
        }
    
    return filtered_results

# =============================================================================
# NEW: COORDINATE TRANSFORMATION FUNCTION (from pruebas_masks.py)
# =============================================================================

def transform_inference_point_to_original(x_inf, y_inf, transform_params):
    """Transform point from inference space back to original image space."""
    # Remove padding
    x_scaled = x_inf - transform_params['pad_left']
    y_scaled = y_inf - transform_params['pad_top']
    
    # Scale back to original size
    original_x = x_scaled / transform_params['scale_factor']
    original_y = y_scaled / transform_params['scale_factor']
    
    return original_x, original_y

# =============================================================================
# NEW: UPDATED POLYGON EXTRACTION (from pruebas_masks.py)
# =============================================================================

def extract_polygon(mask_tensor, transform_params, min_mask_pixels=20):
    """
    Extract polygon WITHOUT simplification - just use raw contour points.
    Based on pruebas_masks.py algorithm.
    """
    if torch.is_tensor(mask_tensor):
        mask = mask_tensor.cpu().numpy()
    else:
        mask = mask_tensor
    
    if mask.ndim == 3:
        mask = mask[0]
    
    # Check if mask is too small
    mask_binary = mask > 0.5
    total_pixels = np.sum(mask_binary)
    
    if total_pixels < min_mask_pixels:
        return [], f"Mask too small: {total_pixels} pixels"
    
    # Convert to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours with CHAIN_APPROX_SIMPLE (changed from CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return [], "No contours found"
    
    # Select largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # NO SIMPLIFICATION - just subsample if too many points
    if len(largest_contour) > 50:  # Only limit if excessive
        step = len(largest_contour) // 50
        working_contour = largest_contour[::step]
    else:
        working_contour = largest_contour  # Use all points
    
    # Convert to normalized coordinates
    polygon_coords = []
    for point in working_contour:
        x_inf, y_inf = point[0]
        
        # Transform coordinates
        original_x, original_y = transform_inference_point_to_original(
            x_inf, y_inf, transform_params
        )
        
        # Validate coordinates
        if (0 <= original_x < transform_params['original_width'] and
                0 <= original_y < transform_params['original_height']):
            norm_x = original_x / transform_params['original_width']
            norm_y = original_y / transform_params['original_height']
            polygon_coords.extend([norm_x, norm_y])
    
    if len(polygon_coords) < 6:
        return [], f"Too few valid points: {len(polygon_coords)//2}"
    
    return polygon_coords, "SUCCESS"

# =============================================================================
# UPDATED CONVERSION FUNCTIONS
# =============================================================================

def convert_to_yolo_format(predictions, original_size):
    """
    Convert Mask R-CNN predictions to YoloV11 format using new algorithm.
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
        mask = masks[i]
        label = labels[i]
        score = scores[i]
        
        # Map class if necessary
        if label in CLASS_MAPPING:
            yolo_class = CLASS_MAPPING[label]
        else:
            continue  # Skip if class is not mapped
        
        # Extract polygon using new method (no simplification)
        polygon_coords, status = extract_polygon(
            mask, transform_params, MIN_CONTOUR_AREA
        )
        
        if status != "SUCCESS" or len(polygon_coords) < 6:
            continue
        
        # Format for YoloV11
        coords_str = ' '.join([f'{coord:.6f}' for coord in polygon_coords])
        yolo_line = f'{yolo_class} {coords_str}'
        yolo_annotations.append(yolo_line)
    
    return yolo_annotations

def create_mask_overlay(image, predictions, original_size,
                       alpha: float = 0.6) -> np.ndarray:
    """Create transparent overlay of predicted masks on original image."""
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
        if masks[i].ndim == 3:
            mask = masks[i][0]  # Take the first channel
        else:
            mask = masks[i]
        
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
    """Save image with transparent mask overlays."""
    # Load original image
    original_image = cv2.imread(image_path)
    
    # Create overlay with transparent masks only
    overlay_image = create_mask_overlay(
        original_image, predictions, original_size, alpha
    )
    
    # Save the visualization
    cv2.imwrite(output_path, overlay_image)

# =============================================================================
# MAIN FUNCTION WITH UPDATED ALGORITHM
# =============================================================================

def main():
    """Main function with updated mask processing algorithm."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Update global variables with arguments
    update_global_variables(args)
    
    print(f"Using device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Using class mapping: {CLASS_MAPPING}")
    print(f"Class-specific area thresholds: {MIN_MASK_AREA_ORIGINAL}")
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
                # Perform inference with filtering
                predictions = perform_single_inference_with_filtering(
                    model, image_path, transforms
                )
                
                # Convert to YoloV11 format
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
                    alpha=0.4
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
