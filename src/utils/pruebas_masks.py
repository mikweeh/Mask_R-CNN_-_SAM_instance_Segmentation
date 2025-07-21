#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Debug script with individual mask visualization for detailed analysis.
"""


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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import argparse


# =============================================================================
# CONFIGURATION - COPY FROM YOUR SCRIPTS
# =============================================================================


# Default paths and parameters (matching your scripts)
MODEL_PATH = 'weights/m02.pth'
DATASET_PATH = "dataset"
TEST_IMAGES_PATH = os.path.join(DATASET_PATH, "valid")
IMG_SIZE = 2048
CONFIDENCE_THRESHOLD = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MASK_RESOLUTION = 112
BASE_MIN_ANCHOR = 16
NUM_CLASSES = 3


# Class configuration
CLASS_MAPPING = {1: 0, 2: 1}
CLASS_NAMES = {0: "Chromis chromis", 1: "Coris julis"}


# Polygon parameters
MIN_MASK_AREA_ORIGINAL = [256, 100]  # Minimum area in original image pixels (256 for chromis and 100 for coris)
MIN_CONTOUR_AREA = 64  # Minimum number of pixels in the mask area in inference space (2048×2048)
SIMPLIFICATION_TOLERANCE = 1.4
ENABLE_SIMPLIFICATION = False


# Debug configuration
SHOW_INDIVIDUAL_MASKS = True  # Set to False to disable individual mask display
PAUSE_FOR_EACH_MASK = False   # Pause to see each mask
ENABLE_DETAILED_DEBUG = True   # Print detailed debugging info


# Output configuration
OUTPUT_PATH = "results/debug3"
GENERATE_PDF = True  # Control PDF generation
SAVE_JSON = False     # Control JSON saving


# =============================================================================
# MODEL AND TRANSFORMATION FUNCTIONS (SAME AS BEFORE)
# =============================================================================


def get_anchor_sizes(img_size, base_img_size=1280, base_min_anchor=16):
    """Calculate anchor sizes maintaining powers-of-2 progression."""
    scale_factor = img_size / base_img_size
    min_anchor = int(round(base_min_anchor * scale_factor))
    anchor_sizes = tuple(min_anchor * (2**i) for i in range(5))
    return anchor_sizes


class HighResMaskRCNNPredictor(torch.nn.Module):
    """High resolution mask predictor for Mask R-CNN."""
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
    """Get transforms for inference."""
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
    """Reverse the transformation applied to masks."""
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
# SINGLE INFERENCE FUNCTION WITH FILTERING
# =============================================================================


def perform_single_inference_with_filtering(model, image_path, transforms):
    """Perform ONE inference and return filtered results."""
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
    
    transform_params = calculate_transformation_params(original_height, original_width)
    
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
        pred['masks'][idx] = torch.from_numpy(mask_filled.astype(np.float32)).unsqueeze(0)
        
        # Transform to original size and check area
        mask_original = reverse_mask_transformation(mask_filled, transform_params)
        original_area = np.sum(mask_original > 0.5)
        
        if original_area >= MIN_MASK_AREA_ORIGINAL[current_label]:
            valid_indices.append(idx.item())
    
    # Create filtered results
    if valid_indices:
        valid_indices = torch.tensor(valid_indices, device=pred['masks'].device)
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
    
    print(f"Original detections: {len(pred['masks'])}")
    print(f"After confidence filtering: {torch.sum(high_conf_indices).item()}")
    print(f"After area filtering: {len(filtered_results['masks'])}")
    
    return filtered_results


# =============================================================================
# INDIVIDUAL MASK VISUALIZATION FUNCTION
# =============================================================================


def visualize_individual_mask(mask_id, original_image, mask_tensor, label, score, 
                              transform_params, polygon_coords=None, 
                              polygon_status="", save_path=None):
    """Visualize individual mask with detailed information."""
    
    # Extract mask
    if mask_tensor.ndim == 3:
        mask = mask_tensor[0]
    else:
        mask = mask_tensor
    
    # Transform mask to original size
    mask_original = reverse_mask_transformation(mask, transform_params)
    mask_binary = (mask_original > 0.5)
    
    # Get mask bounding box and area
    mask_area = np.sum(mask_binary)
    
    if mask_area == 0:
        print(f"Mask {mask_id}: Empty mask, skipping visualization")
        return
    
    # Find bounding box of mask
    coords = np.where(mask_binary)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    # Calculate crop region (twice the mask size)
    mask_height = y_max - y_min + 1
    mask_width = x_max - x_min + 1
    
    # Expand by 2x but keep within image bounds
    expand_h = max(mask_height, 100)  # Minimum 100 pixels height
    expand_w = max(mask_width, 100)   # Minimum 100 pixels width
    
    crop_y_min = max(0, y_min - expand_h // 2)
    crop_y_max = min(original_image.shape[0], y_max + expand_h // 2)
    crop_x_min = max(0, x_min - expand_w // 2)
    crop_x_max = min(original_image.shape[1], x_max + expand_w // 2)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Original image crop
    img_crop = original_image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    axes[0].imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Original Crop\nMask {mask_id}')
    axes[0].axis('off')
    
    # 2. Mask overlay on original
    mask_crop = mask_binary[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
    overlay = img_crop.copy()
    
    # Apply colored mask with transparency
    color = [0, 255, 0] if label in CLASS_MAPPING and CLASS_MAPPING[label] == 0 else [255, 0, 0]
    colored_mask = np.zeros_like(overlay)
    colored_mask[mask_crop] = color
    overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
    
    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Mask Overlay\nArea: {mask_area} pixels\nScore: {score:.3f}')
    axes[1].axis('off')
    
    # 3. Binary mask only
    axes[2].imshow(mask_crop, cmap='gray')
    axes[2].set_title(f'Binary Mask\nSize: {mask_width}x{mask_height}')
    axes[2].axis('off')
    
    # 4. Polygon visualization (if available) - MODIFIED TO SHOW CONTOUR LINES
    if polygon_coords and len(polygon_coords) >= 6:
        # Convert normalized polygon coords to crop coordinates
        points = []
        for i in range(0, len(polygon_coords), 2):
            # Convert to original image coordinates
            orig_x = polygon_coords[i] * transform_params['original_width']
            orig_y = polygon_coords[i + 1] * transform_params['original_height']
            
            # Convert to crop coordinates
            crop_x = orig_x - crop_x_min
            crop_y = orig_y - crop_y_min
            
            # Check if point is within crop
            if 0 <= crop_x < img_crop.shape[1] and 0 <= crop_y < img_crop.shape[0]:
                points.append([int(crop_x), int(crop_y)])
        
        if len(points) >= 3:
            points_array = np.array(points, dtype=np.int32)
            
            # Create RGB comparison image
            comparison = np.zeros((mask_crop.shape[0], mask_crop.shape[1], 3), dtype=np.uint8)
            comparison[:, :, 0] = mask_crop.astype(np.uint8) * 255  # Original mask in red
            
            # Draw polygon contour lines instead of filled polygon
            cv2.polylines(comparison, [points_array], isClosed=True, 
                         color=(255, 255, 0), thickness=0)
            
            axes[3].imshow(comparison)
            axes[3].set_title(f'Polygon Comparison\n{len(polygon_coords)//2} points\n{polygon_status}')
        else:
            axes[3].text(0.5, 0.5, f'Polygon Failed\n{polygon_status}', 
                        ha='center', va='center', transform=axes[3].transAxes)
            axes[3].set_title('Polygon - FAILED')
    else:
        axes[3].text(0.5, 0.5, f'No Polygon\n{polygon_status}', 
                    ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title('No Polygon Data')
    
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # Save the visualization if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
    
    plt.show()
    
    if PAUSE_FOR_EACH_MASK:
        input(f"Press Enter to continue to next mask... (Mask {mask_id} done)")



# =============================================================================
# POLYGON CONVERSION FUNCTIONS
# =============================================================================


def extract_polygon(mask_tensor, transform_params, min_mask_pixels=20):
    """Extract polygon WITHOUT simplification - just use raw contour points."""
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
    
    # Find contours
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
    
    # Debug: print contour info
    print(f"    Contour info: {len(contours)} contours found")
    print(f"    Largest contour: {len(largest_contour)} points, area: {cv2.contourArea(largest_contour):.1f}")
    print(f"    Working contour: {len(working_contour)} points")
    
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


def transform_inference_point_to_original(x_inf, y_inf, transform_params):
    """Transform point from inference space back to original image space."""
    # Remove padding
    x_scaled = x_inf - transform_params['pad_left']
    y_scaled = y_inf - transform_params['pad_top']
    
    # Scale back to original size
    original_x = x_scaled / transform_params['scale_factor']
    original_y = y_scaled / transform_params['scale_factor']
    
    return original_x, original_y


def polygon_to_mask_robust(polygon_coords, image_dimensions):
    """Robust polygon to mask conversion with validation."""
    if len(polygon_coords) < 6:  # Need at least 3 points
        return None
    
    width, height = image_dimensions
    
    # Validate image dimensions
    if width <= 0 or height <= 0:
        return None
    
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Convert normalized coordinates to pixel coordinates
    points = []
    for i in range(0, len(polygon_coords), 2):
        x = polygon_coords[i] * width
        y = polygon_coords[i + 1] * height
        
        # Clamp to valid range
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        
        points.append([int(x), int(y)])
    
    if len(points) < 3:
        return None
    
    points = np.array(points, dtype=np.int32)
    
    try:
        cv2.fillPoly(mask, [points], 255)
    except Exception:
        return None
    
    return mask.astype(np.float32) / 255.0


# =============================================================================
# PDF GENERATION FUNCTION
# =============================================================================


def create_comprehensive_pdf_report(output_path, all_results):
    """
    Create comprehensive PDF report for all processed images.
    
    Args:
        output_path (str): Path where PDF will be saved
        all_results (list): List of dictionaries containing results for each image
    
    Returns:
        str: Path to created PDF file
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportlabImage, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
    except ImportError:
        print("Error: reportlab package not installed. Install with: pip install reportlab")
        return None
    
    pdf_path = os.path.join(output_path, "complete_segmentation_report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("<b>Complete Segmentation Analysis Report</b>", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Overall summary statistics
    total_images = len(all_results)
    total_masks = sum(len(result['polygon_info']) for result in all_results)
    total_successful = sum(sum(1 for info in result['polygon_info'] 
                              if info['status'] == 'SUCCESS') for result in all_results)
    total_failed = total_masks - total_successful
    
    story.append(Paragraph("<b>Overall Summary Statistics</b>", styles['Heading2']))
    story.append(Paragraph(f"• Total images processed: {total_images}", 
                          styles['Normal']))
    story.append(Paragraph(f"• Total masks processed: {total_masks}", 
                          styles['Normal']))
    story.append(Paragraph(f"• Successful conversions: {total_successful}", 
                          styles['Normal']))
    story.append(Paragraph(f"• Failed conversions: {total_failed}", 
                          styles['Normal']))
    story.append(Paragraph(f"• Overall success rate: {(total_successful/total_masks*100):.1f}%" 
                          if total_masks > 0 else "• Overall success rate: 0%", 
                          styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Process each image
    for result_idx, result in enumerate(all_results):
        image_file = result['image_file']
        polygon_info = result['polygon_info']
        original_image_path = result['original_image_path']
        
        # Add page break between images (except for the first one)
        if result_idx > 0:
            story.append(PageBreak())
        
        # Image-specific title
        story.append(Paragraph(f"<b>Image {result_idx + 1}: {image_file}</b>", 
                              styles['Heading1']))
        story.append(Spacer(1, 12))
        
        # Image-specific summary
        total_masks_image = len(polygon_info)
        successful_image = sum(1 for info in polygon_info 
                              if info['status'] == 'SUCCESS')
        failed_image = total_masks_image - successful_image
        
        story.append(Paragraph("<b>Image Summary</b>", styles['Heading2']))
        story.append(Paragraph(f"• Masks processed: {total_masks_image}", 
                              styles['Normal']))
        story.append(Paragraph(f"• Successful conversions: {successful_image}", 
                              styles['Normal']))
        story.append(Paragraph(f"• Failed conversions: {failed_image}", 
                              styles['Normal']))
        story.append(Paragraph(f"• Success rate: {(successful_image/total_masks_image*100):.1f}%" 
                              if total_masks_image > 0 else "• Success rate: 0%", 
                              styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Add original image (bigger size)
        if os.path.exists(original_image_path):
            try:
                story.append(Paragraph("<b>Original Image</b>", styles['Heading2']))
                img = ReportlabImage(original_image_path)
                img._restrictSize(7*inch, 7*inch)  # Increased from 6*inch
                story.append(img)
                story.append(Spacer(1, 12))
            except Exception as e:
                story.append(Paragraph(f"Error adding original image: {str(e)}", 
                                     styles['Normal']))
        
        # Individual mask analysis
        if polygon_info:
            story.append(Paragraph("<b>Individual Mask Analysis</b>", 
                                  styles['Heading2']))
            story.append(Spacer(1, 12))
            
            for info in polygon_info:
                story.append(Paragraph(f"<b>Mask {info['mask_id']}: {info['status']}</b>", 
                                      styles['Heading3']))
                
                if info['status'] == 'SUCCESS':
                    story.append(Paragraph(f"• Polygon points: {info['polygon_points']}", 
                                          styles['Normal']))
                    story.append(Paragraph(f"• Original size: "
                                          f"{info['mask_width_original']}x"
                                          f"{info['mask_height_original']} pixels", 
                                          styles['Normal']))
                    story.append(Paragraph(f"• Area retention: "
                                          f"{info['area_retention']:.1f}%", 
                                          styles['Normal']))
                    story.append(Spacer(1, 6))
                    
                    # Add mask visualization (much bigger size)
                    mask_img_path = os.path.join(output_path, 
                                               f"{image_file}_mask_{info['mask_id']}.png")
                    if os.path.exists(mask_img_path):
                        try:
                            img = ReportlabImage(mask_img_path)
                            img._restrictSize(7.5*inch, 4*inch)  # Increased from 5*inch, 3*inch
                            story.append(img)
                            story.append(Spacer(1, 12))
                        except Exception as e:
                            story.append(Paragraph(f"Error adding mask image: {str(e)}", 
                                                 styles['Normal']))
                else:
                    story.append(Paragraph(f"• Failure reason: {info['status']}", 
                                         styles['Normal']))
                
                story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    return pdf_path


# =============================================================================
# JSON SAVING FUNCTION
# =============================================================================


def save_results_json(image_file, polygon_info, predictions, output_path):
    """Save results in JSON format for further analysis."""
    json_path = os.path.join(output_path, f"{image_file}_results.json")
    
    results = {
        'image_file': image_file,
        'total_masks': len(polygon_info),
        'successful_conversions': sum(1 for info in polygon_info if info['status'] == 'SUCCESS'),
        'masks': polygon_info
    }
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"JSON results saved: {json_path}")
    return json_path


# =============================================================================
# MAIN COMPARISON FUNCTION WITH DETAILED DEBUGGING
# =============================================================================


def compare_all_test_images(model_path, test_images_path, output_path):
    """Compare mask processing with detailed individual mask analysis."""
    # Load model
    print("Loading model...")
    model = get_model_instance_segmentation(NUM_CLASSES, MASK_RESOLUTION)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get all test images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    test_images = [f for f in os.listdir(test_images_path) 
                   if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not test_images:
        print(f"No images found in {test_images_path}")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Get transforms
    transforms = get_inference_transforms()
    
    # Store results for all images
    all_results = []
    
    # Process each image
    for idx, image_file in enumerate(test_images):
        print(f"\n{'='*60}")
        print(f"Processing {idx+1}/{len(test_images)}: {image_file}")
        print('='*60)
        
        image_path = os.path.join(test_images_path, image_file)
        
        # Load original image and get size
        original_image = cv2.imread(image_path)
        with Image.open(image_path) as img:
            original_width, original_height = img.size
        original_size = (original_height, original_width)
        
        # Perform single inference with filtering
        print("Performing inference with filtering...")
        predictions = perform_single_inference_with_filtering(model, image_path, transforms)
        
        if len(predictions['masks']) == 0:
            print("No masks detected after filtering, skipping...")
            continue
        
        print(f"Final masks after filtering: {len(predictions['masks'])}")
        
        # Get transformation parameters
        transform_params = calculate_transformation_params(original_height, original_width)
        
        # Process each mask individually
        polygon_info = []
        temp_image_paths = []  # Track temporary image paths for cleanup
        
        for i in range(len(predictions['masks'])):
            mask_tensor = predictions['masks'][i]
            label = predictions['labels'][i]
            score = predictions['scores'][i]
            
            print(f"\n--- Processing Mask {i} ---")
            print(f"Label: {label}, Score: {score:.3f}")
            
            if label in CLASS_MAPPING:
                yolo_class = CLASS_MAPPING[label]
                class_name = CLASS_NAMES[yolo_class]
                print(f"Class: {class_name}")
            else:
                print("Unknown class, skipping...")
                continue
            
            # Extract polygon
            polygon_coords, status = extract_polygon(
                mask_tensor, transform_params, MIN_CONTOUR_AREA
            )
            
            print(f"Polygon extraction: {status}")
            if status == "SUCCESS":
                # Calculate area retention
                original_mask_transformed = reverse_mask_transformation(
                    mask_tensor[0] if mask_tensor.ndim == 3 else mask_tensor, 
                    transform_params
                )
                original_area = np.sum(original_mask_transformed > 0.5)
                
                # Calculate mask bounding box size in original image
                mask_binary_original = original_mask_transformed > 0.5
                if np.any(mask_binary_original):
                    coords = np.where(mask_binary_original)
                    y_min, y_max = coords[0].min(), coords[0].max()
                    x_min, x_max = coords[1].min(), coords[1].max()
                    mask_width_original = x_max - x_min + 1
                    mask_height_original = y_max - y_min + 1
                else:
                    mask_width_original = 0
                    mask_height_original = 0

                polygon_mask = polygon_to_mask_robust(
                    polygon_coords, 
                    (original_width, original_height)
                )
                
                if polygon_mask is not None:
                    polygon_area = np.sum(polygon_mask > 0.5)
                    area_retention = (polygon_area / original_area * 100) if original_area > 0 else 0
                    print(f"Area retention: {area_retention:.1f}%")
                    
                    polygon_info.append({
                        'mask_id': i,
                        'status': 'SUCCESS',
                        'polygon_points': len(polygon_coords) // 2,
                        'original_area': original_area,
                        'polygon_area': polygon_area,
                        'area_retention': area_retention,
                        'mask_width_original': mask_width_original,
                        'mask_height_original': mask_height_original
                    })

                else:
                    polygon_info.append({
                        'mask_id': i,
                        'status': 'FAILED - Polygon to mask conversion',
                        'polygon_points': len(polygon_coords) // 2
                    })
                
                print(f"Polygon points: {len(polygon_coords)//2}")
                print(f"Original mask size (bounding box): {mask_width_original}x{mask_height_original} pixels")
            else:
                polygon_info.append({
                    'mask_id': i,
                    'status': f'FAILED - {status}',
                    'polygon_points': 0
                })
            
            # Show individual mask visualization and save temporarily for PDF
            save_path = None
            if SHOW_INDIVIDUAL_MASKS or GENERATE_PDF:
                save_path = os.path.join(output_path, f"{image_file}_mask_{i}.png")
                temp_image_paths.append(save_path)  # Track for cleanup
                
            visualize_individual_mask(
                i, original_image, mask_tensor, label, score,
                transform_params, polygon_coords, status, save_path
            )
        
        # Store results for this image
        all_results.append({
            'image_file': image_file,
            'polygon_info': polygon_info,
            'predictions': predictions,
            'original_image_path': image_path,
            'temp_image_paths': temp_image_paths  # Store temp paths for cleanup
        })
        
        # Save JSON results if enabled
        if SAVE_JSON:
            save_results_json(image_file, polygon_info, predictions, output_path)
        
        # Print final summary for this image
        print(f"\n{'='*40}")
        print(f"SUMMARY FOR {image_file}")
        print('='*40)
        successful = sum(1 for info in polygon_info if info['status'] == 'SUCCESS')
        failed = len(polygon_info) - successful
        print(f"Total masks processed: {len(polygon_info)}")
        print(f"Successful conversions: {successful}")
        print(f"Failed conversions: {failed}")
        
        # Show detailed results
        for info in polygon_info:
            print(f"  - Mask {info['mask_id']}: {info['status']}")
            if info['status'] == 'SUCCESS':
                print(f"    * Polygon points: {info['polygon_points']}")
                print(f"    * Original size: {info['mask_width_original']}x{info['mask_height_original']} px")
                print(f"    * Area retention: {info['area_retention']:.1f}%")
    
    # Generate single PDF report for all images if enabled
    if GENERATE_PDF and all_results:
        print("\nCreating comprehensive PDF report for all images...")
        pdf_path = create_comprehensive_pdf_report(output_path, all_results)
        
        if pdf_path:
            print(f"PDF report created: {pdf_path}")
            
            # Clean up temporary mask images after PDF generation
            print("Cleaning up temporary mask images...")
            for result in all_results:
                for temp_path in result.get('temp_image_paths', []):
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        print(f"Removed temporary image: {temp_path}")
            
            print("Cleanup completed. Only PDF report remains in output folder.")
        else:
            print("Failed to create PDF report")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Debug script to compare mask processing approaches'
    )
    parser.add_argument('--model_path', type=str, default=MODEL_PATH,
                        help='Path to trained model')
    parser.add_argument('--test_images_path', type=str, default=TEST_IMAGES_PATH,
                        help='Path to test images folder')
    parser.add_argument('--output_path', type=str, default=OUTPUT_PATH,
                        help='Output path for comparison images')
    parser.add_argument('--confidence_threshold', type=float, 
                        default=CONFIDENCE_THRESHOLD,
                        help='Confidence threshold')
    parser.add_argument('--specific_image', type=str, default=None,
                        help='Process only specific image (optional)')
    parser.add_argument('--min_area', type=int, default=MIN_CONTOUR_AREA,
                        help='Minimum mask area threshold')
    
    # Mutually exclusive group for pause control
    pause_group = parser.add_mutually_exclusive_group()
    pause_group.add_argument('--pause', action='store_true',
                            help='Pause for each mask (enable pausing)')
    pause_group.add_argument('--no_pause', action='store_true',
                            help='Do not pause for each mask (disable pausing)')
    
    # Mutually exclusive group for individual mask visualization
    individual_group = parser.add_mutually_exclusive_group()
    individual_group.add_argument('--individual', action='store_true',
                                 help='Show individual mask visualization (enable)')
    individual_group.add_argument('--no_individual', action='store_true',
                                 help='Skip individual mask visualization (disable)')
    
    # Mutually exclusive group for detailed debugging
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument('--debug', action='store_true',
                            help='Enable detailed debugging output')
    debug_group.add_argument('--no_debug', action='store_true',
                            help='Disable detailed debugging output')
    
    # PDF generation control
    pdf_group = parser.add_mutually_exclusive_group()
    pdf_group.add_argument('--pdf', action='store_true',
                          help='Generate PDF report')
    pdf_group.add_argument('--no_pdf', action='store_true',
                          help='Do not generate PDF report')
    
    # JSON saving control
    json_group = parser.add_mutually_exclusive_group()
    json_group.add_argument('--json', action='store_true',
                           help='Save JSON results')
    json_group.add_argument('--no_json', action='store_true',
                           help='Do not save JSON results')
    
    return parser.parse_args()


def main():
    """Main function with argument parsing."""
    args = parse_arguments()
    
    # Update global variables
    global CONFIDENCE_THRESHOLD, MODEL_PATH, TEST_IMAGES_PATH, OUTPUT_PATH
    global MIN_CONTOUR_AREA, PAUSE_FOR_EACH_MASK, SHOW_INDIVIDUAL_MASKS
    global ENABLE_DETAILED_DEBUG, GENERATE_PDF, SAVE_JSON
    
    CONFIDENCE_THRESHOLD = args.confidence_threshold
    MODEL_PATH = args.model_path
    TEST_IMAGES_PATH = args.test_images_path
    OUTPUT_PATH = args.output_path
    MIN_CONTOUR_AREA = args.min_area
    
    # Handle mutually exclusive pause arguments
    if args.pause:
        PAUSE_FOR_EACH_MASK = True
    elif args.no_pause:
        PAUSE_FOR_EACH_MASK = False
    # If neither is specified, keep global variable value
    
    # Handle mutually exclusive individual visualization arguments
    if args.individual:
        SHOW_INDIVIDUAL_MASKS = True
    elif args.no_individual:
        SHOW_INDIVIDUAL_MASKS = False
    # If neither is specified, keep global variable value
    
    # Handle mutually exclusive debug arguments
    if args.debug:
        ENABLE_DETAILED_DEBUG = True
    elif args.no_debug:
        ENABLE_DETAILED_DEBUG = False
    # If neither is specified, keep global variable value
    
    # Handle mutually exclusive PDF arguments
    if args.pdf:
        GENERATE_PDF = True
    elif args.no_pdf:
        GENERATE_PDF = False
    # If neither is specified, keep global variable value
    
    # Handle mutually exclusive JSON arguments
    if args.json:
        SAVE_JSON = True
    elif args.no_json:
        SAVE_JSON = False
    # If neither is specified, keep global variable value
    
    print(f"Model: {MODEL_PATH}")
    print(f"Test images path: {TEST_IMAGES_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Minimum mask area: {MIN_CONTOUR_AREA}")
    print(f"Show individual masks: {SHOW_INDIVIDUAL_MASKS}")
    print(f"Pause for each mask: {PAUSE_FOR_EACH_MASK}")
    print(f"Detailed debugging: {ENABLE_DETAILED_DEBUG}")
    print(f"Generate PDF: {GENERATE_PDF}")
    print(f"Save JSON: {SAVE_JSON}")
    
    # Process specific image or all test images
    if args.specific_image:
        # Create a temporary folder with just the specific image
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        specific_path = os.path.join(TEST_IMAGES_PATH, args.specific_image)
        if os.path.exists(specific_path):
            shutil.copy(specific_path, temp_dir)
            print(f"Processing only: {args.specific_image}")
            compare_all_test_images(MODEL_PATH, temp_dir, OUTPUT_PATH)
            shutil.rmtree(temp_dir)
        else:
            print(f"Image {args.specific_image} not found in {TEST_IMAGES_PATH}")
    else:
        # Process all test images
        compare_all_test_images(MODEL_PATH, TEST_IMAGES_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()
