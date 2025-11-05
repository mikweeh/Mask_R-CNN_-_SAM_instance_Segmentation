#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Mask R-CNN Inference
Performs object detection and segmentation using only Mask R-CNN model.
"""

import argparse
import os
import torch
import cv2
import numpy as np
import json

# Import shared model loader
from maskrcnn_loader import load_maskrcnn_model

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DETECTION_THRESHOLD = 0.5
MASK_RESOLUTION = 112
MIN_MASK_AREA = 100
TOL = 0  # Tolerance for polygon simplification
THRESHOLD = 0.8  # Threshold to consider pixel is mask


def run_maskrcnn_inference(model, image_rgb, detection_threshold, img_size=2048):
    """
    Run Mask R-CNN inference on a single image.
    
    Args:
        model: Trained Mask R-CNN model
        image_rgb: Image as RGB numpy array
        detection_threshold: Confidence threshold
        img_size: Image size for inference (must match training size)
        
    Returns:
        List of detections with boxes, scores, classes, and masks
    """
    # Resize image to training size
    h, w = image_rgb.shape[:2]
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image_rgb, (new_w, new_h))
    
    # Pad to square
    pad_h = img_size - new_h
    pad_w = img_size - new_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    
    padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    
    # Convert to tensor
    img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        predictions = model(img_tensor)
    
    pred = predictions[0]
    
    # Filter by confidence threshold
    keep = pred['scores'] > detection_threshold
    boxes = pred['boxes'][keep].cpu().numpy()
    scores = pred['scores'][keep].cpu().numpy()
    labels = pred['labels'][keep].cpu().numpy()
    masks = pred['masks'][keep].cpu().numpy()
    
    # Transform boxes and masks back to original image coordinates
    detections = []
    for box, score, label, mask in zip(boxes, scores, labels, masks):
        # Remove padding from box
        box[0] = max(0, box[0] - pad_left)
        box[1] = max(0, box[1] - pad_top)
        box[2] = max(0, box[2] - pad_left)
        box[3] = max(0, box[3] - pad_top)
        
        # Scale back to original size
        box = box / scale
        
        # Remove padding and scale mask
        mask = mask.squeeze()
        mask_unpadded = mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
        mask_original = cv2.resize(mask_unpadded, (w, h))
        
        detections.append({
            'box': box,
            'score': score,
            'class': int(label),
            'mask': mask_original
        })
    
    return detections


def mask_to_yolo_segmentation(mask, class_id, img_height, img_width, tolerance=0):
    """
    Convert binary mask to YOLO segmentation format.
    
    Args:
        mask: Binary mask (H, W)
        class_id: Class ID
        img_height: Original image height
        img_width: Original image width
        tolerance: Polygon simplification tolerance
        
    Returns:
        YOLO format string or None if conversion fails
    """
    try:
        binary_mask = (mask > THRESHOLD).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < MIN_MASK_AREA:
            return None
        
        # Simplify polygon if tolerance > 0
        if tolerance > 0:
            epsilon = tolerance * cv2.arcLength(largest_contour, True)
            largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to normalized coordinates
        polygon_coords = []
        for point in largest_contour.squeeze():
            if len(point.shape) == 0 or len(point) < 2:
                continue
            x_norm = float(point[0]) / img_width
            y_norm = float(point[1]) / img_height
            polygon_coords.extend([x_norm, y_norm])
        
        if len(polygon_coords) < 6:  # At least 3 points
            return None
        
        # Format as YOLO string
        yolo_str = f"{class_id} " + " ".join([f"{p:.6f}" for p in polygon_coords])
        return yolo_str
        
    except Exception as e:
        print(f"Warning: Error in mask to YOLO conversion: {e}")
        return None


def save_visualization_image(image_rgb, detections, class_names, output_path, alpha=0.4):
    """
    Create and save visualization with overlaid masks.
    
    Args:
        image_rgb: Original image in RGB format
        detections: List of detection dictionaries
        class_names: Dict mapping class_id to class_name
        output_path: Where to save visualization
        alpha: Transparency of overlay
    """
    result_image = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    
    # Define colors for different classes (BGR format)
    class_colors = {
        0: (0, 255, 0),    # Green
        1: (255, 0, 0),    # Blue
        2: (0, 0, 255)     # Red
    }
    
    for det in detections:
        mask = det['mask']
        class_id = det['class']
        
        binary_mask = (mask > THRESHOLD).astype(np.uint8)
        
        if not np.any(binary_mask):
            continue
        
        color = class_colors.get(class_id, (255, 255, 255))
        
        # Create colored mask overlay
        colored_mask = np.zeros_like(result_image)
        colored_mask[binary_mask == 1] = color
        
        # Apply transparency
        mask_area = (binary_mask == 1)
        if np.any(mask_area):
            result_image[mask_area] = cv2.addWeighted(
                result_image[mask_area], 1.0 - alpha,
                colored_mask[mask_area], alpha, 0
            )
    
    cv2.imwrite(output_path, result_image)


def process_all_images(maskrcnn_model, class_names, input_folder, 
                       output_labels, output_images, maskrcnn_img_size=2048):
    """
    Process all images with Mask R-CNN only.
    
    Args:
        maskrcnn_model: Trained Mask R-CNN model
        class_names: Dict mapping class_id to class_name
        input_folder: Folder with test images
        output_labels: Output folder for YOLO labels
        output_images: Output folder for visualization images
        maskrcnn_img_size: Image size for Mask R-CNN inference
    """
    os.makedirs(output_labels, exist_ok=True)
    os.makedirs(output_images, exist_ok=True)
    
    # Get all images
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    print(f"{len(image_files)} images to process")
    print("="*80)
    
    total_detections = 0
    class_counts = {cls_id: 0 for cls_id in class_names.keys()}
    
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(input_folder, img_file)
        print(f"[{idx}/{len(image_files)}] Processing {img_file}")
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"  Failed to load, skipping")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # Run Mask R-CNN inference
        detections = run_maskrcnn_inference(
            maskrcnn_model, image_rgb, DETECTION_THRESHOLD, 
            img_size=maskrcnn_img_size
        )
        
        # Convert detections to YOLO format
        yolo_annotations = []
        for det in detections:
            yolo_str = mask_to_yolo_segmentation(
                det['mask'], det['class'], h, w, tolerance=TOL
            )
            if yolo_str is not None:
                yolo_annotations.append(yolo_str)
                class_counts[det['class']] += 1
                total_detections += 1
        
        # Save label file
        basename = os.path.splitext(img_file)[0]
        label_path = os.path.join(output_labels, f"{basename}.txt")
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        
        # Save visualization
        viz_output_path = os.path.join(output_images, f"{basename}.jpg")
        save_visualization_image(
            image_rgb, detections, class_names, viz_output_path, alpha=0.4
        )
        
        print(f"  Detected {len(detections)} instances")
        print(f"  Saved: {basename}.txt ({len(yolo_annotations)} masks)")
    
    # Summary
    print("="*80)
    print("INFERENCE COMPLETED")
    print("="*80)
    print(f"Processed {len(image_files)} images")
    print(f"Total detections: {total_detections}")
    for class_id, count in sorted(class_counts.items()):
        class_name = class_names[class_id]
        print(f"  - {class_name}: {count} instances")


def update_global_variables(args):
    """Update global variables with command-line arguments."""
    global DETECTION_THRESHOLD, MASK_RESOLUTION, MIN_MASK_AREA, TOL, THRESHOLD
    
    DETECTION_THRESHOLD = args.detection_threshold
    MASK_RESOLUTION = args.mask_resolution
    MIN_MASK_AREA = args.min_mask_area
    TOL = args.polygon_tolerance
    THRESHOLD = args.maskrcnn_mask_threshold


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description='Standalone Mask R-CNN Inference'
    )
    parser.add_argument('--maskrcnn_model', type=str, required=True,
                        help='Path to Mask R-CNN model')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Input folder with images')
    parser.add_argument('--output_labels', type=str, required=True,
                        help='Output folder for labels')
    parser.add_argument('--output_images', type=str, required=True,
                        help='Output folder for images')
    parser.add_argument('--detection_threshold', type=float, default=DETECTION_THRESHOLD,
                        help='Detection confidence threshold')
    parser.add_argument('--maskrcnn_img_size', type=int, default=2048,
                        help='Image size for Mask R-CNN inference')
    parser.add_argument('--mask_resolution', type=int, default=MASK_RESOLUTION,
                        help='Mask resolution (28, 56, or 112)')
    parser.add_argument('--class_names', type=str, required=True,
                        help='JSON dict of class_id: class_name')
    parser.add_argument('--min_mask_area', type=int, default=MIN_MASK_AREA,
                        help='Minimum mask area in pixels')
    parser.add_argument('--polygon_tolerance', type=int, default=TOL,
                        help='Polygon simplification tolerance')
    parser.add_argument('--maskrcnn_mask_threshold', type=float, default=THRESHOLD,
                        help='Threshold to consider pixel as mask (0-1) for Mask R-CNN masks')
    
    args = parser.parse_args()
    
    # Update global variables using the function
    update_global_variables(args)

    # Parse class names
    class_names = json.loads(args.class_names)
    class_names = {int(k): v for k, v in class_names.items()}

    print("="*80)
    print("STANDALONE MASK R-CNN INFERENCE")
    print("="*80)
    print(f"Mask R-CNN model: {args.maskrcnn_model}")
    print(f"Mask R-CNN IMG_SIZE: {args.maskrcnn_img_size}")
    print(f"Mask Resolution: {MASK_RESOLUTION}")
    print(f"Detection threshold: {args.detection_threshold}")
    print(f"Min mask area: {MIN_MASK_AREA}")
    print(f"Polygon tolerance: {TOL}")  
    print(f"Mask threshold: {THRESHOLD}")
    print(f"Input folder: {args.input_folder}")
    print(f"Output labels: {args.output_labels}")
    print(f"Output images: {args.output_images}")
    print("="*80)
    
    # Load Mask R-CNN model
    num_classes = len(class_names) + 1  # Including background
    maskrcnn_model = load_maskrcnn_model(
        args.maskrcnn_model, 
        num_classes=num_classes,
        mask_resolution=MASK_RESOLUTION,
        device=DEVICE
    )
    
    # Process all images
    process_all_images(
        maskrcnn_model, class_names, args.input_folder,
        args.output_labels, args.output_images,
        maskrcnn_img_size=args.maskrcnn_img_size
    )
    
    print("="*80)
    print("✓ MASK R-CNN INFERENCE COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
