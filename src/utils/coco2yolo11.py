#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script makes inference of Mask R-CNN and transforms the resulting
inferenced masks into YoloV11 format.
"""

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

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================

# Paths and files
MODEL_PATH = 'best_mask_rcnn_model.pth'
INPUT_IMAGES_FOLDER = 'dataset/test'
OUTPUT_LABELS_FOLDER = 'dataset/inference/labels'
OUTPUT_IMAGES_FOLDER = 'dataset/inference/images'

# Model parameters
NUM_CLASSES = 3
IMG_SIZE = 1024
CONFIDENCE_THRESHOLD = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class mapping
CLASS_MAPPING = {
    1: 0,
    2: 1,
}

# =============================================================================
# AUXILIARY FUNCTIONS
# =============================================================================

def get_model_instance_segmentation(num_classes):
    """
    Creates the Mask R-CNN model with the same architecture as training.
    Must match exactly with the training configuration.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights="DEFAULT"
    )
    
    # Apply the same custom anchor generator as training
    anchor_generator = torchvision.models.detection.anchor_utils.AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),  # Match training exactly
        aspect_ratios=((0.5, 1.0, 2.0),) * 5  # Match training exactly
    )
    
    model.rpn.anchor_generator = anchor_generator
    
    # Replace box classifier (same as before)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes
    )
    
    # Replace mask predictor (same as before)
    in_features_mask = (
        model.roi_heads.mask_predictor.conv5_mask.in_channels
    )
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    
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

def mask_to_polygon(mask, min_area=50):
    """
    Converts a binary mask to polygon coordinates.
    
    Args:
        mask: Binary mask numpy array
        min_area: Minimum area to consider a valid contour
    
    Returns:
        List of polygon coordinates in format [x1, y1, x2, y2, ...]
    """
    # Convert mask to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Select the largest contour
    if not contours:
        return []
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Check minimum area
    if cv2.contourArea(largest_contour) < min_area:
        return []
    
    # Simplify contour
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    simplified_contour = cv2.approxPolyDP(
        largest_contour, epsilon, True
    )
    
    # Convert to coordinate list
    polygon_coords = []
    for point in simplified_contour:
        x, y = point[0]
        polygon_coords.extend([x, y])
    
    return polygon_coords

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

def predict_with_tta(model, image, device, scales=[0.8, 1.0, 1.2]):
    """
    Test Time Augmentation for improved small object detection.
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for scale in scales:
            # Scale image
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_image = cv2.resize(image, (new_w, new_h))
            
            # Apply transforms
            transformed = get_inference_transforms()(image=scaled_image)
            tensor_image = transformed['image'].unsqueeze(0).to(device)
            
            # Predict
            pred = model(tensor_image)[0]
            
            # Scale back predictions
            if len(pred['boxes']) > 0:
                pred['boxes'] /= scale
                pred['masks'] = torch.nn.functional.interpolate(
                    pred['masks'], size=(h, w), mode='bilinear'
                )
            
            predictions.append(pred)
    
    # Use the prediction from scale=1.0 for simplicity
    return predictions[1] if len(predictions) > 1 else predictions[0]


def convert_to_yolo_format(predictions, original_size):
    """
    Converts Mask R-CNN predictions to YoloV11 format.
    
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
        
        # Convert mask to polygon
        polygon_coords = mask_to_polygon(mask_original)
        
        if len(polygon_coords) < 6:  # Need at least 3 points (6 coordinates)
            continue
        
        # Normalize coordinates
        normalized_coords = normalize_polygon(
            polygon_coords, original_height, original_width
        )
        
        # Format for YoloV11
        coords_str = ' '.join([f'{coord:.6f}' for coord in normalized_coords])
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
# MAIN INFERENCE AND CONVERSION FUNCTION
# =============================================================================

def main():
    """
    Main function that executes complete inference and conversion.
    """
    print(f"Using device: {DEVICE}")
    
    # Create output folders
    os.makedirs(OUTPUT_LABELS_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_IMAGES_FOLDER, exist_ok=True)
    
    # Load trained model
    print("Loading Mask R-CNN model...")
    model = get_model_instance_segmentation(NUM_CLASSES)
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
                
                # Optional: Use TTA for better results (slower but more accurate)
                # image_for_tta = cv2.imread(image_path)
                # image_for_tta = cv2.cvtColor(image_for_tta, cv2.COLOR_BGR2RGB)
                # predictions_tta = predict_with_tta(model, image_for_tta, DEVICE)
                # Use predictions_tta instead of predictions if using TTA

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
                print(f"  - Detected {len(yolo_annotations)} instances")
                
            except Exception as e:
                print(f"  - Error processing {filename}: {str(e)}")
    
    print(f"\nProcessing completed!")
    print(f"Images processed: {processed_count}")
    print(f"Labels saved in: {OUTPUT_LABELS_FOLDER}")
    print(f"Visualizations saved in: {OUTPUT_IMAGES_FOLDER}")

# =============================================================================
# EXECUTE SCRIPT
# =============================================================================

if __name__ == "__main__":
    main()
