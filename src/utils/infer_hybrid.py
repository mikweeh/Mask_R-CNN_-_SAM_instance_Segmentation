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
    model = maskrcnn_resnet50_fpn(pretrained=False)
    
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

def load_sam2_predictor(model_id, fine_tuned_weights_path, device):
    """
    Load fine-tuned SAM2 predictor.
    
    Args:
        model_id: Hugging Face model ID
        fine_tuned_weights_path: Path to fine-tuned weights
        device: Device to load model on
    
    Returns:
        SAM2ImagePredictor instance
    """
    print(f"Loading SAM2 predictor from: {model_id}")
    
    predictor = SAM2ImagePredictor.from_pretrained(model_id)
    predictor.model.to(device)
    
    if os.path.exists(fine_tuned_weights_path):
        print(f"Loading fine-tuned SAM2 weights: "
              f"{fine_tuned_weights_path}")
        state_dict = torch.load(fine_tuned_weights_path,
                                 map_location=device)
        predictor.model.load_state_dict(state_dict)
        print("Fine-tuned SAM2 weights loaded successfully")
    else:
        print(f"WARNING: Fine-tuned weights not found, using base SAM2")
    
    predictor.model.eval()
    return predictor


# =============================================================================
# HYBRID INFERENCE
# =============================================================================

def run_maskrcnn_inference(model, image_rgb, confidence_threshold=0.5):
    """
    Run Mask R-CNN inference to get detections.
    
    Args:
        model: Mask R-CNN model
        image_rgb: Image as numpy array (H, W, 3) in RGB
        confidence_threshold: Minimum confidence score
    
    Returns:
        List of detections: [{'box': [x1,y1,x2,y2], 'class': int,
                              'score': float}, ...]
    """
    # Convert to tensor
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
    image_tensor = image_tensor / 255.0
    image_tensor = image_tensor.to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        predictions = model([image_tensor])[0]
    
    # Extract detections above threshold
    detections = []
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence_threshold:
            # Convert to class index (Mask R-CNN uses 1-indexed)
            class_idx = int(label) - 1
            
            detections.append({
                'box': box.astype(int),  # [x1, y1, x2, y2]
                'class': class_idx,  # 0 or 1
                'score': float(score)
            })
    
    return detections


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
        multimask_output=False  # Single best mask
    )
    
    # Return the mask (shape: H, W)
    return masks[0].astype(np.uint8)


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


def process_image_hybrid(maskrcnn_model, sam2_predictor, image_path):
    """
    Process a single image with the hybrid pipeline.
    
    Args:
        maskrcnn_model: Mask R-CNN model
        sam2_predictor: SAM2 predictor
        image_path: Path to image file
    
    Returns:
        List of YOLO format annotation strings
    """
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Step 1: Get detections from Mask R-CNN
    detections = run_maskrcnn_inference(maskrcnn_model, image_rgb,
                                         DETECTION_THRESHOLD)
    
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
        cx = int(np.mean(points[:, 0]))
        cy = int(np.mean(points[:, 1]))
        cv2.putText(image, CLASS_NAMES[class_id], (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)
    
    cv2.imwrite(output_path, image)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function for hybrid inference."""
    parser = argparse.ArgumentParser(
        description='Hybrid Mask R-CNN + SAM2 inference'
    )
    parser.add_argument('--maskrcnn_model', type=str,
                        default=MASKRCNN_MODEL_PATH)
    parser.add_argument('--sam2_model', type=str,
                        default=SAM2_MODEL_PATH)
    parser.add_argument('--sam2_model_id', type=str,
                        default=SAM2_MODEL_ID)
    parser.add_argument('--input_folder', type=str,
                        default=INPUT_IMAGES_FOLDER)
    parser.add_argument('--output_labels', type=str,
                        default=OUTPUT_LABELS_FOLDER)
    parser.add_argument('--output_images', type=str,
                        default=OUTPUT_IMAGES_FOLDER)
    parser.add_argument('--detection_threshold', type=float,
                        default=DETECTION_THRESHOLD)
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("HYBRID INFERENCE: Mask R-CNN + SAM2")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Mask R-CNN model: {args.maskrcnn_model}")
    print(f"SAM2 model: {args.sam2_model}")
    print(f"Detection threshold: {args.detection_threshold}")
    print("="*60 + "\n")
    
    # Create output folders
    os.makedirs(args.output_labels, exist_ok=True)
    os.makedirs(args.output_images, exist_ok=True)
    
    # Load models
    print("Loading models...")
    maskrcnn_model = load_maskrcnn_model(args.maskrcnn_model,
                                          num_classes=2, device=DEVICE)
    sam2_predictor = load_sam2_predictor(args.sam2_model_id,
                                          args.sam2_model, DEVICE)
    print("Models loaded successfully\n")
    
    # Process all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(args.input_folder)
                   if any(f.lower().endswith(ext)
                          for ext in image_extensions)]
    
    print(f"Processing {len(image_files)} images...\n")
    
    for img_file in image_files:
        img_path = os.path.join(args.input_folder, img_file)
        print(f"Processing: {img_file}")
        
        try:
            # Run hybrid inference
            yolo_annotations = process_image_hybrid(maskrcnn_model,
                                                     sam2_predictor,
                                                     img_path)
            
            print(f"  Detected {len(yolo_annotations)} fish\n")
            
            # Save labels
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(args.output_labels, label_file)
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            # Save visualization
            vis_path = os.path.join(args.output_images, img_file)
            create_visualization(img_path, yolo_annotations, vis_path)
            
        except Exception as e:
            print(f"  Error: {e}\n")
            continue
    
    print("="*60)
    print("HYBRID INFERENCE COMPLETE!")
    print(f"Labels saved in: {args.output_labels}")
    print(f"Visualizations saved in: {args.output_images}")
    print("="*60)


if __name__ == "__main__":
    main()
