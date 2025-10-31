#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid inference: Mask R-CNN for detection/classification + SAM2 for
mask refinement.

This combines:
- Mask R-CNN: Provides bounding boxes and class predictions
- SAM2: Refines the segmentation masks

Author: Generated for fish segmentation project
"""

import argparse
import os
import torch
import cv2
import numpy as np
import json
from pathlib import Path
from shapely import Polygon as ShapelyPolygon

# SAM2 imports
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    print("WARNING: SAM2 not available")
    SAM2_AVAILABLE = False

# Mask R-CNN imports
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# =============================================================================
# CONFIGURATION
# =============================================================================

MASKRCNN_MODEL_PATH = 'weights/maskrcnn_model.pth'
SAM2_MODEL_PATH = 'weights/sam2_fish_unified.pth'
SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"
INPUT_IMAGES_FOLDER = 'dataset/test'
OUTPUT_LABELS_FOLDER = 'dataset/inference/labels_hybrid'
OUTPUT_IMAGES_FOLDER = 'dataset/inference/images_hybrid'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Detection parameters
DETECTION_THRESHOLD = 0.5  # Confidence threshold for Mask R-CNN
MIN_MASK_AREA = 100  # Minimum mask size in pixels

# Class names
CLASS_NAMES = {0: "Chromis chromis", 1: "Coris julis"}

# Tolerance of polygon simplification
TOL = 0

# Threshold to consider pixel is mask (0-1)
THRESHOLD = 0.99

# Erode operation after getting masks
ERODE_ITERATIONS = 2

# =============================================================================
# MASK R-CNN LOADING
# =============================================================================

def load_maskrcnn_model(model_path, num_classes=2, device='cuda'):
    """
    Load your trained Mask R-CNN model with CUSTOM high-resolution predictor.
    
    This matches your train.py architecture with HighResMaskRCNNPredictor.
    
    Args:
        model_path: Path to your trained Mask R-CNN weights
        num_classes: Number of fish classes (2: Chromis, Coris)
        device: Device to load model on
    
    Returns:
        Loaded Mask R-CNN model in eval mode
    """
    print(f"Loading Mask R-CNN model from: {model_path}")
    
    # Import your custom predictor class
    import torch.nn as nn
    import torch.nn.functional as F
    
    # Define the SAME custom predictor from your train.py
    class HighResMaskRCNNPredictor(nn.Module):
        """Custom high-resolution mask predictor (from your train.py)."""
        
        def __init__(self, in_channels, dim_reduced, num_classes,
                     mask_size):
            super().__init__()
            self.mask_size = mask_size
            
            # Determine hidden dimension based on mask size
            if mask_size > 56:
                hidden_dim = dim_reduced // 2  # 256 -> 128 for memory
            else:
                hidden_dim = dim_reduced
            
            # Build layers based on mask resolution
            if mask_size == 28:
                # Standard configuration for 28x28
                self.conv5_mask = nn.ConvTranspose2d(dim_reduced,
                                                      dim_reduced, 2, 2, 0)
                self.relu = nn.ReLU(inplace=True)
                self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes,
                                                   1, 1, 0)
            elif mask_size == 56:
                # Enhanced configuration for 56x56
                self.conv5_mask = nn.ConvTranspose2d(dim_reduced,
                                                      dim_reduced, 2, 2, 0)
                self.relu1 = nn.ReLU(inplace=True)
                self.conv6_mask = nn.ConvTranspose2d(dim_reduced,
                                                      dim_reduced, 2, 2, 0)
                self.relu2 = nn.ReLU(inplace=True)
                self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes,
                                                   1, 1, 0)
            elif mask_size == 112:
                # Advanced configuration for 112x112
                self.conv5_mask = nn.ConvTranspose2d(dim_reduced,
                                                      hidden_dim, 2, 2, 0)
                self.relu1 = nn.ReLU(inplace=True)
                self.conv6_mask = nn.ConvTranspose2d(hidden_dim,
                                                      hidden_dim, 2, 2, 0)
                self.relu2 = nn.ReLU(inplace=True)
                self.conv7_mask = nn.ConvTranspose2d(hidden_dim,
                                                      hidden_dim, 2, 2, 0)
                self.relu3 = nn.ReLU(inplace=True)
                self.mask_fcn_logits = nn.Conv2d(hidden_dim, num_classes,
                                                   1, 1, 0)
            else:
                raise ValueError(f"Mask resolution {mask_size} not "
                               f"supported. Use 28, 56, or 112.")
            
            # Initialize weights
            for name, param in self.named_parameters():
                if "weight" in name:
                    nn.init.kaiming_normal_(param, mode="fan_out",
                                           nonlinearity="relu")
                elif "bias" in name:
                    nn.init.constant_(param, 0)
        
        def forward(self, x):
            if self.mask_size == 28:
                x = self.conv5_mask(x)
                x = self.relu(x)
                x = self.mask_fcn_logits(x)
            elif self.mask_size == 56:
                x = self.conv5_mask(x)
                x = self.relu1(x)
                x = self.conv6_mask(x)
                x = self.relu2(x)
                x = self.mask_fcn_logits(x)
            elif self.mask_size == 112:
                x = self.relu1(self.conv5_mask(x))
                x = self.relu2(self.conv6_mask(x))
                x = self.relu3(self.conv7_mask(x))
                x = self.mask_fcn_logits(x)
            
            # Crop to exact size if needed
            if x.shape[-1] != self.mask_size:
                x = F.interpolate(x, size=(self.mask_size, self.mask_size),
                                 mode='bilinear', align_corners=False)
            return x
    
    # Load pre-trained base model
    model = maskrcnn_resnet50_fpn(weights=None)
    
    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                        num_classes + 1)
    
    # Detect mask resolution from your saved model
    # Load state dict first to check architecture
    checkpoint = torch.load(model_path, map_location=device)
    
    # Detect mask resolution from layer names
    if 'roi_heads.mask_predictor.conv7_mask.weight' in checkpoint:
        mask_resolution = 112
    elif 'roi_heads.mask_predictor.conv6_mask.weight' in checkpoint:
        mask_resolution = 56
    else:
        mask_resolution = 28
    
    print(f"Detected mask resolution: {mask_resolution}x{mask_resolution}")
    
    # Replace mask predictor with YOUR custom high-resolution version
    in_features_mask = (model.roi_heads.mask_predictor.
                        conv5_mask.in_channels)
    hidden_layer = 256
    
    model.roi_heads.mask_predictor = HighResMaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes + 1,
        mask_resolution
    )
    
    # NOW load your trained weights
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    print("Mask R-CNN model loaded successfully")
    return model



# =============================================================================
# SAM2 LOADING
# =============================================================================

def load_sam2_predictor(model_path, model_id, device='cuda'):
    """
    Load SAM2 predictor with fine-tuned weights.
    
    Args:
        model_path: Path to fine-tuned SAM2 checkpoint
        model_id: SAM2 model ID (e.g., 'facebook/sam2.1-hiera-large')
        device: Device to load on
    
    Returns:
        SAM2ImagePredictor with loaded weights
    """
    print(f"Loading SAM2 model from: {model_path}")
    
    # First load the base model from HuggingFace
    predictor = SAM2ImagePredictor.from_pretrained(model_id)
    
    # Then load your fine-tuned weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume the entire dict is the state_dict
            state_dict = checkpoint
    else:
        # Checkpoint is directly the state_dict
        state_dict = checkpoint
    
    # Load the state dict into the model
    predictor.model.load_state_dict(state_dict, strict=False)
    predictor.model.to(device)
    predictor.model.eval()
    
    print(f"✓ SAM2 model loaded successfully")
    return predictor


# =============================================================================
# HYBRID INFERENCE
# =============================================================================

def run_maskrcnn_inference(model, image_rgb, confidence_threshold=0.5,
                          img_size=2048):
    """
    Run Mask R-CNN inference with proper image transformation.
    
    Args:
        model: Mask R-CNN model
        image_rgb: Image as numpy array (H, W, 3) in RGB
        confidence_threshold: Minimum confidence score
        img_size: Image size used during training
    
    Returns:
        List of detections with boxes in original image coordinates
    """
    import albumentations as A
    
    # Store original dimensions
    original_height, original_width = image_rgb.shape[:2]
    
    # Apply transformations (same as training)
    transform = A.Compose([
        A.LongestMaxSize(max_size=img_size, p=1.0),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=1.0
        ),
    ])
    
    transformed = transform(image=image_rgb)
    image_transformed = transformed['image']
    
    # Calculate transformation parameters
    scale_factor = img_size / max(original_height, original_width)
    scaled_height = int(original_height * scale_factor)
    scaled_width = int(original_width * scale_factor)
    pad_top = (img_size - scaled_height) // 2
    pad_left = (img_size - scaled_width) // 2
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_transformed).permute(
        2, 0, 1
    ).float()
    image_tensor = image_tensor / 255.0
    image_tensor = image_tensor.to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        predictions = model([image_tensor])[0]
    
    # Extract detections and reverse transformation
    detections = []
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence_threshold:
            # Reverse transformation
            x1, y1, x2, y2 = box
            x1 = (x1 - pad_left) / scale_factor
            y1 = (y1 - pad_top) / scale_factor
            x2 = (x2 - pad_left) / scale_factor
            y2 = (y2 - pad_top) / scale_factor
            
            # Clip to image bounds
            x1 = max(0, min(x1, original_width))
            y1 = max(0, min(y1, original_height))
            x2 = max(0, min(x2, original_width))
            y2 = max(0, min(y2, original_height))
            
            # Mask R-CNN uses 1-indexed labels
            class_idx = int(label) - 1
            
            detections.append({
                'box': np.array([x1, y1, x2, y2], dtype=int),
                'class': class_idx,
                'score': float(score)
            })
    
    return detections


def mask_to_yolo_segmentation(mask, class_id, image_shape, tolerance=None):
    """
    Convert binary mask to YOLO segmentation format with proper polygon
    simplification.
    
    Uses Shapely's Douglas-Peucker algorithm for professional polygon
    simplification (same method as label_simplify.py).
    
    Args:
        mask: Binary mask (H, W) - can be 0/1 or probability values
        class_id: Class ID for this mask
        image_shape: Tuple (height, width) of the image
        tolerance: Simplification tolerance (default: 2.0)
                  Higher values = more simplification
    
    Returns:
        YOLO format string: "class_id x1 y1 x2 y2 ... xn yn"
        Returns None if mask is empty or invalid
    """
    # Convert to numpy if tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    # Ensure mask is 2D
    if mask.ndim == 3:
        mask = mask.squeeze()
    
    # Binarize mask
    unique_vals = np.unique(mask)
    if len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1])):
        binary_mask = mask.astype(np.uint8)
    else:
        binary_mask = (mask > THRESHOLD).astype(np.uint8)
    
    # Check if mask has any positive pixels
    if not np.any(binary_mask):
        return None
    
    if ERODE_ITERATIONS > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.erode(binary_mask, kernel, 
                               iterations=ERODE_ITERATIONS)
        
        # Check again after erosion
        if not np.any(binary_mask):
            return None

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return None
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Check minimum area
    if cv2.contourArea(largest_contour) < 10:
        return None
    
    # Convert contour to list of points
    contour_points = largest_contour.reshape(-1, 2)
    
    # Convert to list of tuples for Shapely
    points = [(int(pt[0]), int(pt[1])) for pt in contour_points]
    
    # Check minimum points
    if len(points) < 3:
        return None
    
    try:
        # Create Shapely polygon
        polygon = ShapelyPolygon(points)
        
        # Simplify using Douglas-Peucker algorithm
        if tolerance is not None and tolerance > 0:
            simplified_polygon = polygon.simplify(
                tolerance=tolerance,
                preserve_topology=True
            )
            final_points = list(simplified_polygon.exterior.coords)[:-1]
        else:
            # Or no simplification
            final_points = points
        
        # Safety check: ensure we have at least 3 points
        if len(final_points) < 3:
            # Fall back to original points if over-simplified
            final_points = points
        
        # Normalize coordinates to [0, 1]
        height, width = image_shape
        normalized_points = []
        
        for x, y in final_points:
            x_norm = max(0.0, min(1.0, x / width))
            y_norm = max(0.0, min(1.0, y / height))
            normalized_points.extend([x_norm, y_norm])
        
        # Format as YOLO string
        yolo_str = (f"{class_id} " + 
                   " ".join(f"{p:.6f}" for p in normalized_points))
        
        return yolo_str
        
    except Exception as e:
        print(f"Warning: Error in polygon simplification: {e}")
        return None


def save_visualization_image(image_rgb, masks_data, class_names, 
                            output_path, alpha=0.4):
    """
    Create and save visualization with overlaid masks.
    
    Args:
        image_rgb: Original image in RGB format
        masks_data: List of (mask, class_id) tuples
        class_names: Dict mapping class_id to class_name
        output_path: Where to save the visualization
        alpha: Transparency of overlay (default: 0.4)
    """
    # Convert RGB to BGR for OpenCV
    result_image = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    
    # Define colors for different classes (BGR format for OpenCV)
    class_colors = {
        0: (0, 255, 0),    # Green for class 0
        1: (255, 0, 0),    # Blue for class 1
        2: (0, 0, 255),    # Red for class 2 (if needed)
    }
    
    # Process each mask
    for mask, class_id in masks_data:
        # Convert mask to binary
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        if mask.ndim == 3:
            mask = mask.squeeze()
        
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        if not np.any(binary_mask):
            continue
        
        # Get color for this class
        color = class_colors.get(class_id, (255, 255, 255))
        
        # Create colored mask overlay
        colored_mask = np.zeros_like(result_image)
        colored_mask[binary_mask == 1] = color
        
        # Apply transparency
        mask_area = binary_mask == 1
        if np.any(mask_area):
            result_image[mask_area] = cv2.addWeighted(
                result_image[mask_area],
                1.0 - alpha,
                colored_mask[mask_area],
                alpha,
                0
            )
    
    # Save the visualization
    cv2.imwrite(output_path, result_image)


def process_all_images_hybrid(maskrcnn_model, sam2_predictors,
                              class_names, input_folder, output_labels,
                              output_images, maskrcnn_img_size=2048):
    """
    Process all images with sequential class-specific SAM2 refinement.
    
    Args:
        maskrcnn_model: Mask R-CNN model
        sam2_predictors: Dict {class_idx: SAM2ImagePredictor}
        class_names: Dict {class_idx: class_name}
        input_folder: Folder with test images
        output_labels: Output folder for YOLO labels
        output_images: Output folder for images (not used)
        maskrcnn_img_size: IMG_SIZE used during Mask R-CNN training
    """
    # Create output folders
    os.makedirs(output_labels, exist_ok=True)
    os.makedirs(output_images, exist_ok=True)
    
    # Get all images
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    print(f"\nFound {len(image_files)} images to process")

    # PHASE 1: Mask R-CNN detection
    # =============================
    print("\n" + "="*80)
    print("PHASE 1: Mask R-CNN Detection")
    print("="*80)
    
    all_detections = {}
    
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(input_folder, img_file)
        print(f"[{idx}/{len(image_files)}] Detecting: {img_file}")
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"  ✗ Failed to load, skipping")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        detections = run_maskrcnn_inference(
            maskrcnn_model,
            image_rgb,
            DETECTION_THRESHOLD,
            img_size=maskrcnn_img_size
        )
        
        all_detections[img_path] = {
            'detections': detections,
            'image_rgb': image_rgb,
            'img_file': img_file
        }
        
        # Print detection summary
        class_counts = {cls_idx: 0 for cls_idx in class_names.keys()}
        for d in detections:
            class_counts[d['class']] += 1
        
        summary = ", ".join(
            f"{class_counts[cls_idx]} {class_names[cls_idx]}"
            for cls_idx in sorted(class_names.keys())
        )
        print(f"  Detected: {summary}")
    
    # PHASE 2: Sequential SAM2 processing per class
    # ==============================================
    annotations_by_image = {img_path: [] for img_path in all_detections}
    
    for class_idx in sorted(sam2_predictors.keys()):
        class_name = class_names[class_idx]
        predictor = sam2_predictors[class_idx]
        
        print("\n" + "="*80)
        print(f"PHASE {class_idx + 2}: SAM2 Refinement for {class_name}")
        print("="*80)
        
        # Count total instances for this class
        total_instances = sum(
            sum(1 for d in data['detections'] if d['class'] == class_idx)
            for data in all_detections.values()
        )
        print(f"Processing {total_instances} {class_name} instances")
        
        processed = 0
        for img_path, data in all_detections.items():
            img_file = data['img_file']
            image_rgb = data['image_rgb']
            class_dets = [
                d for d in data['detections'] if d['class'] == class_idx
            ]
            
            if not class_dets:
                continue
            
            print(f"\nProcessing {img_file}: {len(class_dets)} {class_name}")
            
            for det in class_dets:
                processed += 1
                print(f"  [{processed}/{total_instances}] "
                      f"{class_name} (score: {det['score']:.3f})")
                
                # Refine with class-specific SAM2
                refined_mask = refine_mask_with_sam2(
                    predictor,
                    image_rgb,
                    det['box']
                )
                
                if refined_mask is not None:
                    yolo_str = mask_to_yolo_segmentation(
                        refined_mask,
                        det['class'],
                        image_rgb.shape[:2],
                        tolerance=TOL
                    )
                    if yolo_str:
                        annotations_by_image[img_path].append(yolo_str)
        
        # Free GPU memory after each class
        print(f"\n  Unloading SAM2 model for {class_name}...")
        del predictor
        torch.cuda.empty_cache()

    # PHASE 3: Save visualization images
    # =========================================================================
    print("\n" + "="*80)
    print("SAVING VISUALIZATION IMAGES")
    print("="*80)
    
    for img_path, yolo_annotations in annotations_by_image.items():
        img_file = all_detections[img_path]['img_file']
        image_rgb = all_detections[img_path]['image_rgb']
        base_name = os.path.splitext(img_file)[0]
        
        # Reconstruct masks from YOLO annotations for visualization
        h, w = image_rgb.shape[:2]
        masks_for_viz = []
        
        for yolo_str in yolo_annotations:
            parts = yolo_str.split()
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            # Convert normalized coordinates back to pixels
            points = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i+1] * h)
                points.append([x, y])
            
            # Create mask from polygon
            mask = np.zeros((h, w), dtype=np.uint8)
            if len(points) >= 3:
                cv2.fillPoly(mask, [np.array(points)], 1)
                masks_for_viz.append((mask, class_id))
        
        # Save visualization
        viz_output_path = os.path.join(
            output_images, 
            f"{base_name}.jpg"
        )
        save_visualization_image(
            image_rgb,
            masks_for_viz,
            class_names,
            viz_output_path,
            alpha=0.4
        )
        print(f"  Saved visualization: {base_name}.jpg")
    
    # PHASE FINAL: Save annotations
    # =============================
    print("\n" + "="*80)
    print("FINAL PHASE: Saving Annotations")
    print("="*80)
    
    for img_path, yolo_annotations in annotations_by_image.items():
        img_file = all_detections[img_path]['img_file']
        base_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(output_labels, f"{base_name}.txt")
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        print(f"  Saved: {base_name}.txt ({len(yolo_annotations)} masks)")
    
    # Summary
    total_masks = sum(len(anns) for anns in annotations_by_image.values())
    print(f"\n✓ Processed {len(image_files)} images")
    print(f"  Total masks: {total_masks}")
    for class_idx, class_name in sorted(class_names.items()):
        class_total = sum(
            sum(1 for d in data['detections'] if d['class'] == class_idx)
            for data in all_detections.values()
        )
        print(f"  - {class_name}: {class_total} instances")


def refine_mask_with_sam2(predictor, image_rgb, bbox):
    """
    Use SAM2 to generate a refined mask for a detected object.
    
    Args:
        predictor: SAM2ImagePredictor instance
        image_rgb: Image as numpy array (H, W, 3)
        bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
        Refined binary mask as numpy array (H, W)
    """
    # Set image for SAM2
    predictor.set_image(image_rgb)
    
    # Use bbox as prompt
    masks, scores, logits = predictor.predict(
        box=bbox,
        multimask_output=True,
        return_logits=False
    )
    
    if len(masks) > 0:
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        return best_mask.astype(np.uint8)
    
    return None


def mask_to_yolo_polygon(mask, class_id):
    """
    Convert binary mask to YOLO polygon format.
    
    Args:
        mask: Binary mask (H, W)
        class_id: Class ID (0 or 1)
    
    Returns:
        YOLO format string: "class_id x1 y1 x2 y2 ..."
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Subsample if too many points
    if len(largest_contour) > 50:
        step = max(1, len(largest_contour) // 50)
        largest_contour = largest_contour[::step]
    
    # Convert to normalized coordinates
    height, width = mask.shape
    polygon_coords = []
    
    for point in largest_contour:
        x, y = point[0]
        norm_x = x / width
        norm_y = y / height
        polygon_coords.extend([norm_x, norm_y])
    
    if len(polygon_coords) < 6:
        return None
    
    # Format as YOLO string
    coords_str = ' '.join([f'{c:.6f}' for c in polygon_coords])
    return f'{class_id} {coords_str}'


def process_image_hybrid(maskrcnn_model, sam2_predictor, image_path,
                        maskrcnn_img_size=2048):
    """
    Process a single image with the hybrid pipeline.
    
    Args:
        maskrcnn_model: Mask R-CNN model
        sam2_predictor: SAM2 predictor
        image_path: Path to image file
        maskrcnn_img_size: IMG_SIZE used during Mask R-CNN training
    
    Returns:
        List of YOLO format annotation strings
    """
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Step 1: Get detections from Mask R-CNN (with proper img_size)
    detections = run_maskrcnn_inference(
        maskrcnn_model,
        image_rgb,
        DETECTION_THRESHOLD,
        img_size=maskrcnn_img_size
    )
    
    if len(detections) == 0:
        return []
    
    # Step 2: Refine each detection with SAM2
    yolo_annotations = []
    
    for det in detections:
        bbox = det['box']
        class_id = det['class']
        
        # Use SAM2 to refine the mask
        refined_mask = refine_mask_with_sam2(sam2_predictor, image_rgb,
                                               bbox)
        
        # Filter by area
        if refined_mask.sum() < MIN_MASK_AREA:
            continue
        
        # Convert to YOLO format
        yolo_line = mask_to_yolo_polygon(refined_mask, class_id)
        
        if yolo_line is not None:
            yolo_annotations.append(yolo_line)
    
    return yolo_annotations


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualization(image_path, yolo_annotations, output_path):
    """Create visualization of detections."""
    image = cv2.imread(image_path)
    
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
        cv2.polylines(image, [points], True, colors[class_id], 2)
        
        # Add class label
        # cx = int(np.mean(points[:, 0]))
        # cy = int(np.mean(points[:, 1]))
        # cv2.putText(image, CLASS_NAMES[class_id], (cx, cy),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)
    
    cv2.imwrite(output_path, image)


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
    
    args = parser.parse_args()
    
    # Parse JSON arguments
    sam2_model_paths = json.loads(args.sam2_model_paths)
    # Convert string keys to int
    sam2_model_paths = {int(k): v for k, v in sam2_model_paths.items()}
    
    class_names = json.loads(args.class_names)
    class_names = {int(k): v for k, v in class_names.items()}
    
    # Set global threshold
    global DETECTION_THRESHOLD
    DETECTION_THRESHOLD = args.detection_threshold
    
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
    print(f"Input folder: {args.input_folder}")
    print(f"Output labels: {args.output_labels}")
    print("="*80)
    
    # Check SAM2 availability
    if not SAM2_AVAILABLE:
        print("ERROR: SAM2 not available. Install with:")
        print("  pip install git+https://github.com/facebookresearch/sam2.git")
        sys.exit(1)
    
    # Load Mask R-CNN model
    print("\nLoading Mask R-CNN model...")
    num_classes = len(class_names)
    maskrcnn_model = load_maskrcnn_model(args.maskrcnn_model, num_classes)
    print("✓ Mask R-CNN model loaded")
    
    # Load all SAM2 predictors
    print("\nLoading SAM2 predictors...")
    sam2_predictors = {}
    for class_idx in sorted(sam2_model_paths.keys()):
        class_name = class_names[class_idx]
        model_path = sam2_model_paths[class_idx]
        print(f"  Loading SAM2 for {class_name}...")
        sam2_predictors[class_idx] = load_sam2_predictor(
            model_path,
            args.sam2_model_id,
            DEVICE
        )
        print(f"  ✓ SAM2 for {class_name} loaded")
    
    # Process all images
    process_all_images_hybrid(
        maskrcnn_model,
        sam2_predictors,
        class_names,
        args.input_folder,
        args.output_labels,
        args.output_images,
        maskrcnn_img_size=args.maskrcnn_img_size
    )
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETED ✓")
    print("="*80)


if __name__ == "__main__":
    main()
