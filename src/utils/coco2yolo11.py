#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script makes inference of Mask R-CNN and transforms the resulting
inferenced masks into YoloV11 format.
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

# =============================================================================
# DEFAULT GLOBAL CONFIGURATION VARIABLES
# =============================================================================

# Paths and files
MODEL_PATH = 'weights/MaskRCNN_5.pth'
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

# Default class mapping - can be overridden by arguments from main.py
CLASS_MAPPING = {1: 0, 2: 1}  # Default: COCO class IDs to YOLO class IDs

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
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
    
    # Class configuration argument (optional - will override default if provided)
    parser.add_argument('--class_mapping', type=str, default=None,
                       help='JSON string with COCO to YOLO class mapping')
    
    return parser.parse_args()

def update_global_variables(args):
    """Update global variables with command-line arguments."""
    global MODEL_PATH, INPUT_IMAGES_FOLDER, OUTPUT_LABELS_FOLDER
    global OUTPUT_IMAGES_FOLDER, NUM_CLASSES, IMG_SIZE, CONFIDENCE_THRESHOLD
    global MASK_RESOLUTION, BASE_MIN_ANCHOR, CLASS_MAPPING
    
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
    
    # Update class configuration ONLY if provided as argument
    if args.class_mapping is not None:
        CLASS_MAPPING = json.loads(args.class_mapping)
        # Convert string keys to integers (JSON converts int keys to strings)
        CLASS_MAPPING = {int(k): v for k, v in CLASS_MAPPING.items()}
        print(f"Updated CLASS_MAPPING from arguments: {CLASS_MAPPING}")
    else:
        print(f"Using default CLASS_MAPPING: {CLASS_MAPPING}")

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
            self.conv5_mask = torch.nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)
            self.relu = torch.nn.ReLU(inplace=True)
            self.mask_fcn_logits = torch.nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
            
        elif mask_size <= 56:
            # Enhanced configuration for 56×56
            self.conv5_mask = torch.nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)
            self.relu1 = torch.nn.ReLU(inplace=True)
            self.conv6_mask = torch.nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)
            self.relu2 = torch.nn.ReLU(inplace=True)
            self.mask_fcn_logits = torch.nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
            
        elif mask_size <= 112:
            # Advanced configuration for 112×112
            self.conv5_mask = torch.nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)
            self.relu1 = torch.nn.ReLU(inplace=True)
            self.conv6_mask = torch.nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)
            self.relu2 = torch.nn.ReLU(inplace=True)
            self.conv7_mask = torch.nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)
            self.relu3 = torch.nn.ReLU(inplace=True)
            self.mask_fcn_logits = torch.nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
            
        else:
            raise ValueError(f"Mask resolution {mask_size} not supported. Use 28, 56, or 112.")
        
        # Initialize weights
        for name, param in self.named_parameters():
            if "weight" in name:
                torch.nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
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
# MAIN FUNCTION WITH ARGUMENT SUPPORT
# =============================================================================

def main():
    """
    Main function that executes complete inference and conversion with argument support.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Update global variables with arguments
    update_global_variables(args)
    
    print(f"Using device: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Using class mapping: {CLASS_MAPPING}")
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
