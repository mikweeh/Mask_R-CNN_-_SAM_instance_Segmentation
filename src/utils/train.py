#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full version of training and testing and pdf report of a Mask R-CNN
model for small fishes segmentation
"""

import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for PDF generation
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import random
from PIL import Image

# PDF generation imports
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, PageBreak,
                                    Image as RLImage)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    print("ReportLab not available. Install with: pip install reportlab")
    REPORTLAB_AVAILABLE = False

# =============================================================================
# DEFAULT GLOBAL CONFIGURATION VARIABLES
# =============================================================================

# Model name. Relative path from ws
MODEL_PATH = 'weights/ms01.pth'

# Dataset paths
DATASET_PATH = "dataset"
TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train")
VAL_IMAGES_PATH = os.path.join(DATASET_PATH, "valid")
TEST_IMAGES_PATH = os.path.join(DATASET_PATH, "test")
TRAIN_ANNOTATIONS = os.path.join(TRAIN_IMAGES_PATH, 
                                "_annotations_filtered.coco.json")
VAL_ANNOTATIONS = os.path.join(VAL_IMAGES_PATH, 
                              "_annotations_filtered.coco.json")

# Training parameters
NUM_CLASSES = 3
BATCH_SIZE = 1
NUM_EPOCHS = 200
LEARNING_RATE = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
IMG_SIZE = 2048
CONFIDENCE_THRESHOLD = 0.3
DICE_WEIGHT = 0.0
USE_FOCAL_DICE = False
OVERSAMPLE_SMALL_OBJECTS = True
USE_COPY_PASTE = True
MASK_RESOLUTION = 56
BASE_MIN_ANCHOR = 16

# RPN parameters
RPN_PRE_NMS_TOP_N_TRAIN = 1500
RPN_POST_NMS_TOP_N_TRAIN = 600
RPN_NMS_THRESH = 0.6

# Report configuration
REPORT_OUTPUT_PATH = "results"
TEMP_FIGURES_PATH = os.path.join(REPORT_OUTPUT_PATH, "imgs")

# Default class configuration - can be overridden by arguments from main.py
CLASS_MAPPING = {1: 0, 2: 1}  # Default: COCO class IDs to YOLO class IDs
CLASS_NAMES = {0: "Chromis chromis", 1: "Coris julis"}  # Default class names

# Other
SMALL_OBJECT_MAX_AREA = 1000  # Number of pixels to be considered small
LOSS_MASK_WEIGHT = 1.5

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def get_next_model_name_train():
    """
    Get the next available model name using the global MODEL_PATH variable.
    """
    global MODEL_PATH
    
    weights_dir = os.path.dirname(MODEL_PATH)
    base_name_with_ext = os.path.basename(MODEL_PATH)
    base_name, ext = os.path.splitext(base_name_with_ext)
    
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        print(f"Created weights directory: {weights_dir}")
    
    base_path = MODEL_PATH
    if not os.path.exists(base_path):
        return base_path
    
    counter = 1
    while True:
        new_path = os.path.join(weights_dir, f"{base_name}_{counter}{ext}")
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN model for small fish segmentation'
    )
    
    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH,
                       help='Path to the dataset directory')
    parser.add_argument('--model_path', type=str, default=get_next_model_name_train(),
                       help='Model name for saving weights')
    
    # Training parameters
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES,
                       help='Number of classes (including background)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    
    # Model parameters
    parser.add_argument('--img_size', type=int, default=IMG_SIZE,
                       help='Image size for training')
    parser.add_argument('--confidence_threshold', type=float, 
                       default=CONFIDENCE_THRESHOLD,
                       help='Confidence threshold for predictions')
    parser.add_argument('--dice_weight', type=float, default=DICE_WEIGHT,
                       help='Weight for dice loss')
    parser.add_argument('--mask_resolution', type=int, 
                       default=MASK_RESOLUTION,
                       help='Mask resolution (28, 56, or 112)')
    parser.add_argument('--base_min_anchor', type=int, 
                       default=BASE_MIN_ANCHOR,
                       help='Base minimum anchor size')
    
    # Training options
    # Boolean flags with mutually exclusive groups
    # Focal Dice
    group_focal = parser.add_mutually_exclusive_group()
    group_focal.add_argument('--use_focal_dice', dest='use_focal_dice', action='store_true')
    group_focal.add_argument('--no_use_focal_dice', dest='use_focal_dice', action='store_false')
    parser.set_defaults(use_focal_dice=USE_FOCAL_DICE)  # Use script's global as default

    # Oversampling
    group_oversample = parser.add_mutually_exclusive_group()
    group_oversample.add_argument('--oversample_small_objects', dest='oversample_small_objects', action='store_true')
    group_oversample.add_argument('--no_oversample_small_objects', dest='oversample_small_objects', action='store_false')
    parser.set_defaults(oversample_small_objects=OVERSAMPLE_SMALL_OBJECTS)

    # Copy-Paste
    group_copypaste = parser.add_mutually_exclusive_group()
    group_copypaste.add_argument('--use_copy_paste', dest='use_copy_paste', action='store_true')
    group_copypaste.add_argument('--no_use_copy_paste', dest='use_copy_paste', action='store_false')
    parser.set_defaults(use_copy_paste=USE_COPY_PASTE)
    
    # RPN parameters
    parser.add_argument('--rpn_pre_nms_top_n_train', type=int,
                       default=RPN_PRE_NMS_TOP_N_TRAIN,
                       help='RPN pre-NMS top N during training')
    parser.add_argument('--rpn_post_nms_top_n_train', type=int,
                       default=RPN_POST_NMS_TOP_N_TRAIN,
                       help='RPN post-NMS top N during training')
    parser.add_argument('--rpn_nms_thresh', type=float, 
                       default=RPN_NMS_THRESH,
                       help='RPN NMS threshold')
    
    # Output parameters
    parser.add_argument('--report_output_path', type=str, 
                       default=REPORT_OUTPUT_PATH,
                       help='Path for output reports')
    
    # Class configuration arguments (optional - will override defaults if provided)
    parser.add_argument('--class_mapping', type=str, default=None,
                       help='JSON string with COCO to YOLO class mapping')
    parser.add_argument('--class_names', type=str, default=None,
                       help='JSON string with YOLO class ID to name mapping')
    
    return parser.parse_args()

def update_global_variables(args):
    """Update global variables with command-line arguments."""
    global MODEL_PATH, DATASET_PATH, TRAIN_IMAGES_PATH, VAL_IMAGES_PATH
    global TEST_IMAGES_PATH, TRAIN_ANNOTATIONS, VAL_ANNOTATIONS
    global NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
    global IMG_SIZE, CONFIDENCE_THRESHOLD, DICE_WEIGHT, USE_FOCAL_DICE
    global OVERSAMPLE_SMALL_OBJECTS, USE_COPY_PASTE, MASK_RESOLUTION
    global BASE_MIN_ANCHOR, RPN_PRE_NMS_TOP_N_TRAIN, RPN_POST_NMS_TOP_N_TRAIN
    global RPN_NMS_THRESH, REPORT_OUTPUT_PATH, TEMP_FIGURES_PATH
    global CLASS_MAPPING, CLASS_NAMES
    
    # Update standard global variables
    MODEL_PATH = args.model_path
    DATASET_PATH = args.dataset_path
    NUM_CLASSES = args.num_classes
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    IMG_SIZE = args.img_size
    CONFIDENCE_THRESHOLD = args.confidence_threshold
    DICE_WEIGHT = args.dice_weight
    USE_FOCAL_DICE = args.use_focal_dice
    OVERSAMPLE_SMALL_OBJECTS = args.oversample_small_objects
    USE_COPY_PASTE = args.use_copy_paste
    MASK_RESOLUTION = args.mask_resolution
    BASE_MIN_ANCHOR = args.base_min_anchor
    RPN_PRE_NMS_TOP_N_TRAIN = args.rpn_pre_nms_top_n_train
    RPN_POST_NMS_TOP_N_TRAIN = args.rpn_post_nms_top_n_train
    RPN_NMS_THRESH = args.rpn_nms_thresh
    REPORT_OUTPUT_PATH = args.report_output_path
    
    # Update derived paths
    TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train")
    VAL_IMAGES_PATH = os.path.join(DATASET_PATH, "valid")
    TEST_IMAGES_PATH = os.path.join(DATASET_PATH, "test")
    TRAIN_ANNOTATIONS = os.path.join(TRAIN_IMAGES_PATH, 
                                    "_annotations_filtered.coco.json")
    VAL_ANNOTATIONS = os.path.join(VAL_IMAGES_PATH, 
                                  "_annotations_filtered.coco.json")
    TEMP_FIGURES_PATH = os.path.join(REPORT_OUTPUT_PATH, "imgs")
    
    # Update class configuration ONLY if provided as arguments
    if args.class_mapping is not None:
        CLASS_MAPPING = json.loads(args.class_mapping)
        # Convert string keys to integers (JSON converts int keys to strings)
        CLASS_MAPPING = {int(k): v for k, v in CLASS_MAPPING.items()}
        print(f"Updated CLASS_MAPPING from arguments: {CLASS_MAPPING}")
    else:
        print(f"Using default CLASS_MAPPING: {CLASS_MAPPING}")
    
    if args.class_names is not None:
        CLASS_NAMES = json.loads(args.class_names)
        # Convert string keys to integers (JSON converts int keys to strings)
        CLASS_NAMES = {int(k): v for k, v in CLASS_NAMES.items()}
        print(f"Updated CLASS_NAMES from arguments: {CLASS_NAMES}")
    else:
        print(f"Using default CLASS_NAMES: {CLASS_NAMES}")

# =============================================================================
# COPY-PASTE AUGMENTATION FOR SMALL OBJECTS
# =============================================================================

def copy_paste_small_objects(image, masks, boxes, labels, max_copies=2):
    """
    Memory-optimized copy-paste augmentation for small objects.
    
    Args:
        image: Input image array
        masks: Object masks
        boxes: Bounding boxes
        labels: Object labels
        max_copies: Maximum number of copies per object
        
    Returns:
        Augmented image, masks, boxes, and labels
    """
    if len(boxes) == 0:
        return image, masks, boxes, labels
        
    # Limit augmentation for very small datasets or single-item batches
    if len(boxes) > 20:  # Skip copy-paste if too many objects already
        return image, masks, boxes, labels
        
    h, w = image.shape[:2]
    new_masks = masks.copy()
    new_boxes = boxes.copy()
    new_labels = labels.copy()
    
    # Reduce number of copies for memory efficiency
    max_copies = min(max_copies, 6)
    
    for i in range(len(boxes)):
        # Only copy small objects
        box_area = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
        if box_area < (h * w * 0.01):
            num_copies = np.random.randint(1, max_copies + 1)
            
            for _ in range(num_copies):
                try:
                    # Extract object region with error handling
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    mask = masks[i]
                    mask_h, mask_w = y2 - y1, x2 - x1
                    
                    if mask_h <= 0 or mask_w <= 0:
                        continue
                    
                    # Try to place the copied object (fewer attempts)
                    for attempt in range(5):  # Reduced from 10 to 5
                        new_x = np.random.randint(0, max(1, w - mask_w))
                        new_y = np.random.randint(0, max(1, h - mask_h))
                        
                        # Check for overlap
                        overlap = False
                        for existing_box in new_boxes:
                            if (new_x < existing_box[2] and new_x + mask_w > existing_box[0] and
                                new_y < existing_box[3] and new_y + mask_h > existing_box[1]):
                                overlap = True
                                break
                        
                        if not overlap:
                            # Create new mask with bounds checking
                            new_mask = np.zeros_like(masks[0])
                            object_mask = mask[y1:y2, x1:x2]
                            
                            end_y = min(new_y + mask_h, h)
                            end_x = min(new_x + mask_w, w)
                            actual_h = end_y - new_y
                            actual_w = end_x - new_x
                            
                            if actual_h > 0 and actual_w > 0:
                                # Apply augmentation
                                object_region = image[y1:y2, x1:x2]
                                resized_region = cv2.resize(object_region, (actual_w, actual_h))
                                resized_mask = cv2.resize(object_mask.astype(np.uint8), (actual_w, actual_h))
                                
                                mask_bool = resized_mask > 0.5
                                image[new_y:end_y, new_x:end_x][mask_bool] = resized_region[mask_bool]
                                new_mask[new_y:end_y, new_x:end_x] = resized_mask
                                
                                # Add new annotations
                                new_masks = np.concatenate([new_masks, new_mask[None, ...]])
                                new_boxes = np.concatenate([new_boxes, 
                                                          [[new_x, new_y, end_x, end_y]]])
                                new_labels = np.concatenate([new_labels, [labels[i]]])
                                break
                                
                except Exception as e:
                    # Skip this copy if there's an error
                    continue
    
    return image, new_masks, new_boxes, new_labels

# =============================================================================
# ENHANCED AUGMENTATION CONFIGURATION
# =============================================================================

def get_train_transforms():
    """
    Enhanced training augmentations with multi-scale support for small objects.
    """
    return A.Compose([
        # Multi-scale training - randomly vary image size
        A.OneOf([
            A.LongestMaxSize(max_size=IMG_SIZE),
            A.LongestMaxSize(max_size=int(round(0.9 * IMG_SIZE / 32) * 32)),
            A.LongestMaxSize(max_size=int(round(0.8 * IMG_SIZE / 32) * 32)),
        ], p=1.0),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, 
                     border_mode=cv2.BORDER_CONSTANT, p=1.0),
        
        # Enhanced augmentations for small objects
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomScale(scale_limit=0.2, p=0.5),  # Scale variations
        
        # Color augmentations
        A.RandomBrightnessContrast(brightness_limit=0.2, 
                                  contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, 
                           val_shift_limit=10, p=0.5),
        
        # Noise and blur (careful with small objects)
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
        ], p=0.3),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']),
    additional_targets={'masks': 'masks'})

def get_val_transforms():
    """
    Defines validation transformations (without augmentation).
    This version preserves aspect ratio.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, 
                     border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']),
    additional_targets={'masks': 'masks'})

def get_inference_transforms():
    """
    Transformations for inference only (no bbox_params needed).
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

def clear_gpu_memory():
    """
    Comprehensive memory cleanup to prevent epoch-to-epoch accumulation.
    """
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

# =============================================================================
# DICE LOSS FUNCTIONS
# =============================================================================

def dice_loss_for_masks(pred_masks, target_masks, smooth=1e-6):
    """
    Compute dice loss for mask segmentation in Mask R-CNN.
    """
    if pred_masks.numel() == 0 or target_masks.numel() == 0:
        return torch.zeros(1, device=pred_masks.device,
                           requires_grad=True).mean()
    
    # Ensure pred_masks are in the right format and apply sigmoid
    if pred_masks.dim() == 4 and pred_masks.size(1) == 1:
        pred_masks = pred_masks.squeeze(1)  # Remove channel dimension if 1
    pred_masks = torch.sigmoid(pred_masks)
    
    # Ensure target_masks have the same shape as pred_masks
    if target_masks.dim() == 4 and target_masks.size(1) == 1:
        target_masks = target_masks.squeeze(1)
    
    # Ensure target masks are binary (0 or 1)
    target_masks = (target_masks > 0.5).float()
    
    # Flatten the spatial dimensions for computation
    pred_flat = pred_masks.view(pred_masks.size(0), -1)
    target_flat = target_masks.view(target_masks.size(0), -1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    target_sum = target_flat.sum(dim=1)
    
    # Compute dice coefficient for each mask
    dice_coeff = (2.0 * intersection + smooth) / (pred_sum + target_sum + 
                                                 smooth)
    
    # Return dice loss (1 - dice coefficient)
    dice_loss = 1.0 - dice_coeff.mean()
    
    return dice_loss

def focal_dice_loss(pred_masks, target_masks, alpha=0.25, gamma=2.0, 
                   smooth=1e-6):
    """
    Enhanced dice loss with focal weighting for hard examples.
    """
    if pred_masks.numel() == 0 or target_masks.numel() == 0:
        return torch.tensor(0.0, device=pred_masks.device, 
                          requires_grad=True)
    
    # Apply sigmoid and prepare tensors
    if pred_masks.dim() == 4 and pred_masks.size(1) == 1:
        pred_masks = pred_masks.squeeze(1)
    pred_masks = torch.sigmoid(pred_masks)
    
    if target_masks.dim() == 4 and target_masks.size(1) == 1:
        target_masks = target_masks.squeeze(1)
    
    # Flatten spatial dimensions
    pred_flat = pred_masks.view(pred_masks.size(0), -1)
    target_flat = target_masks.view(target_masks.size(0), -1)
    
    # Standard dice computation
    intersection = (pred_flat * target_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    target_sum = target_flat.sum(dim=1)
    dice_coeff = (2.0 * intersection + smooth) / (pred_sum + target_sum + 
                                                 smooth)
    
    # Apply focal weighting
    focal_weight = alpha * (1 - dice_coeff) ** gamma
    focal_dice_loss = focal_weight * (1.0 - dice_coeff)
    
    return focal_dice_loss.mean()

def compute_mask_iou(mask1: torch.Tensor, mask2: torch.Tensor, 
                     threshold: float = 0.5) -> float:
    """
    Computes IoU between two binary masks.
    """
    # Binarize masks
    mask1_bin = (mask1 > threshold).float()
    mask2_bin = (mask2 > threshold).float()
    
    # Compute intersection and union
    intersection = (mask1_bin * mask2_bin).sum()
    union = mask1_bin.sum() + mask2_bin.sum() - intersection
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return (intersection / union).item()

def match_masks_by_iou(pred_masks: torch.Tensor, 
                       target_masks: torch.Tensor,
                       pred_boxes: torch.Tensor,
                       target_boxes: torch.Tensor,
                       pred_scores: torch.Tensor,
                       iou_threshold: float = 0.3,
                       score_threshold: float = 0.5):
    """
    Matches predicted masks to ground truth masks using IoU scoring.
    """
    matches = []
    
    # Filter predictions by confidence score
    valid_pred_indices = (pred_scores > score_threshold).nonzero(
        as_tuple=True
    )[0]
    
    if len(valid_pred_indices) == 0 or len(target_masks) == 0:
        return matches
    
    # Compute IoU matrix between all prediction-target pairs
    iou_matrix = torch.zeros((len(valid_pred_indices), len(target_masks)))
    
    for i, pred_idx in enumerate(valid_pred_indices):
        for j in range(len(target_masks)):
            iou_score = compute_mask_iou(
                pred_masks[pred_idx], 
                target_masks[j]
            )
            iou_matrix[i, j] = iou_score
    
    # Greedy matching: find best matches iteratively
    used_targets = set()
    
    for i in range(len(valid_pred_indices)):
        # Find best available target for this prediction
        best_target_idx = -1
        best_iou = iou_threshold
        
        for j in range(len(target_masks)):
            if j not in used_targets and iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_target_idx = j
        
        # Add match if valid
        if best_target_idx >= 0:
            matches.append((valid_pred_indices[i].item(), best_target_idx))
            used_targets.add(best_target_idx)
    
    return matches

# =============================================================================
# ENHANCED DATASET WITH OVERSAMPLING FOR SMALL OBJECTS
# =============================================================================

class COCOInstanceDataset(Dataset):
    """
    Enhanced dataset with oversampling for small objects and copy-paste augmentation.
    """

    def __init__(self, images_path, annotations_file, transforms=None, 
                 oversample_small_objects=True, use_copy_paste=True):
        self.images_path = images_path
        self.transforms = transforms
        self.use_copy_paste = use_copy_paste
        self.coco = COCO(annotations_file)
        self.image_ids = list(self.coco.imgs.keys())
        
        if oversample_small_objects:
            self.image_ids = self._oversample_small_objects()

    def _oversample_small_objects(self):
        """
        Oversample images containing small objects.
        """
        small_object_images = []
        regular_images = []
        
        for image_id in self.image_ids:
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            annotations = self.coco.loadAnns(ann_ids)
            
            has_small_objects = False
            for ann in annotations:
                if ann['area'] < SMALL_OBJECT_MAX_AREA:
                    has_small_objects = True
                    break
            
            if has_small_objects:
                small_object_images.append(image_id)
            else:
                regular_images.append(image_id)
        
        # Oversample small object images (repeat n times)
        oversampled_ids = regular_images + small_object_images * 3
        return oversampled_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Load image information
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image_path = os.path.join(self.images_path, image_info['file_name'])

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)

        # Process masks and bounding boxes
        masks = []
        boxes = []
        labels = []

        for ann in annotations:
            if 'segmentation' in ann:
                mask = self.coco.annToMask(ann)
                masks.append(mask)

                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])

        # Convert to arrays for copy-paste augmentation
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            masks = np.array(masks, dtype=np.float32)
            
            # Apply copy-paste augmentation during training
            if self.use_copy_paste and self.transforms is not None:
                image, masks, boxes, labels = copy_paste_small_objects(
                    image, masks, boxes, labels, max_copies=2
                )
            
            # Convert back to tensors
            boxes = torch.FloatTensor(boxes)
            labels = torch.LongTensor(labels)
            masks = torch.FloatTensor(masks)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, image.shape[0], image.shape[1]), dtype=torch.float32)

        # Apply transformations
        if self.transforms:
            bboxes_coco = boxes.clone()
            if len(bboxes_coco) > 0:
                bboxes_coco[:, 2] = bboxes_coco[:, 2] - bboxes_coco[:, 0]
                bboxes_coco[:, 3] = bboxes_coco[:, 3] - bboxes_coco[:, 1]

            transformed = self.transforms(
                image=image,
                bboxes=bboxes_coco.tolist() if len(bboxes_coco) > 0 else [],
                labels=labels.tolist() if len(labels) > 0 else [],
                masks=masks.numpy() if len(masks) > 0 else []
            )

            image = transformed['image']
            if transformed['bboxes']:
                new_boxes = torch.FloatTensor(transformed['bboxes'])
                new_boxes[:, 2] = new_boxes[:, 0] + new_boxes[:, 2]
                new_boxes[:, 3] = new_boxes[:, 1] + new_boxes[:, 3]
                boxes = new_boxes
                labels = torch.LongTensor(transformed['labels'])
                masks = torch.FloatTensor(transformed['masks'])

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([image_id])
        }

        return image, target

# =============================================================================
# ENHANCED MODEL WITH OPTIMIZED ANCHORS FOR SMALL OBJECTS
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


# =============================================================================
# ENHANCED TRAINING WITH WEIGHTED LOSSES
# =============================================================================

def dice_coefficient(pred_mask, true_mask, smooth=1e-6):
    """
    Calculates the Dice coefficient between predicted and true masks.
    """
    pred_mask = (pred_mask > 0.5).float()
    true_mask = (true_mask > 0.5).float()

    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

def train_one_epoch(model, optimizer, data_loader, device, epoch,
                   dice_weight: float = 2.0, use_focal_dice: bool = True):
    """
    Enhanced training with IoU-based mask matching for accurate dice loss.
    """
    model.train()
    total_loss = 0
    total_standard_loss = 0
    total_dice_loss = 0
    num_batches = 0
    num_matched_pairs = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()}
                  for t in targets]

        optimizer.zero_grad(set_to_none=True)

        # Forward pass - get both losses and predictions
        loss_dict = model(images, targets)
        
        # Calculate standard Mask R-CNN losses with optimized weights
        standard_losses = (
            loss_dict['loss_classifier'] * 1.0 +         # Classification loss
            loss_dict['loss_box_reg'] * 1.0 +            # Box regression loss
            loss_dict['loss_mask'] * LOSS_MASK_WEIGHT +  # Standard mask loss
            loss_dict['loss_objectness'] * 1.0 +         # RPN objectness loss
            loss_dict['loss_rpn_box_reg'] * 1.0          # RPN box regression
        )
        
        # Skip dice calculations if dice_weight is 0
        if dice_weight == 0:
            dice_loss_value = torch.tensor(0.0, device=device, 
                                         requires_grad=True)
            total_batch_loss = standard_losses
        else:
            # Initialize dice_loss_value as None
            dice_loss_value = None
            
            # Get predictions for dice loss computation
            model.eval()
            with torch.no_grad():
                predictions = model(images)
            model.train()
            
            # Compute dice loss for each image in the batch
            batch_dice_losses = []
            batch_matches = 0
            
            for pred, target in zip(predictions, targets):
                if (len(pred['masks']) > 0 and len(target['masks']) > 0 and
                    pred['scores'].max() > 0.1):
                    
                    # Resize predicted masks to match target size if needed
                    pred_masks = pred['masks']
                    target_masks = target['masks']
                    
                    if pred_masks.shape[-2:] != target_masks.shape[-2:]:
                        target_h, target_w = target_masks.shape[-2:]
                        pred_masks = F.interpolate(
                            pred_masks, size=(target_h, target_w),
                            mode='bilinear', align_corners=False
                        )
                    
                    # Match masks using IoU scoring
                    matches = match_masks_by_iou(
                        pred_masks=pred_masks,
                        target_masks=target_masks,
                        pred_boxes=pred['boxes'],
                        target_boxes=target['boxes'],
                        pred_scores=pred['scores'],
                        iou_threshold=0.3,  # Minimum IoU for valid match
                        score_threshold=0.1 # Minimum confidence for predictions
                    )
                    
                    # Compute dice loss only for matched pairs
                    if matches:
                        matched_dice_losses = []
                        for pred_idx, target_idx in matches:
                            pred_mask = pred_masks[pred_idx:pred_idx+1]
                            target_mask = target_masks[target_idx:target_idx+1]
                            
                            if use_focal_dice:
                                pair_dice_loss = focal_dice_loss(pred_mask,
                                                               target_mask)
                            else:
                                pair_dice_loss = dice_loss_for_masks(pred_mask,
                                                                   target_mask)
                            matched_dice_losses.append(pair_dice_loss)
                        
                        # Average dice loss for matched pairs in this image
                        if matched_dice_losses:
                            image_dice_loss = torch.stack(matched_dice_losses).mean()
                            batch_dice_losses.append(image_dice_loss)
                            batch_matches += len(matches)
            
            # Average dice loss for the batch
            if batch_dice_losses:
                dice_loss_value = torch.stack(batch_dice_losses).mean()
            else:
                # Create a zero tensor connected to the computation graph
                dice_loss_value = standard_losses * 0.0
            
            num_matched_pairs += batch_matches
            
            # Combine losses with weighting
            total_batch_loss = standard_losses + dice_weight * dice_loss_value
        
        # Backward pass
        total_batch_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()

        # Track losses
        total_loss += total_batch_loss.item()
        total_standard_loss += standard_losses.item()
        total_dice_loss += dice_loss_value.item()
        num_batches += 1

        # Memory cleanup
        del images, targets, loss_dict
        del standard_losses, dice_loss_value, total_batch_loss
        if dice_weight != 0:
            del predictions
        
        if batch_idx % 4 == 0:
            torch.cuda.empty_cache()

        # Print detailed loss information for first epoch
        # if epoch == 1 and batch_idx < 5:
        #     if dice_weight != 0:
        #         print(f"Batch {batch_idx}: Total={total_loss/num_batches:.4f}, "
        #               f"Standard={total_standard_loss/num_batches:.4f}, "
        #               f"Dice={total_dice_loss/num_batches:.4f}, "
        #               f"Matches={batch_matches if dice_weight != 0 else 'N/A'}")
        #     else:
        #         print(f"Batch {batch_idx}: Total={total_loss/num_batches:.4f}, "
        #               f"Standard={total_standard_loss/num_batches:.4f}, "
        #               f"Dice=Skipped")
    
    avg_total_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_dice_loss = total_dice_loss / num_batches if num_batches > 0 else 0.0
    avg_matches_per_batch = num_matched_pairs / num_batches if num_batches > 0 else 0.0
    
    if dice_weight != 0:
        print(f"Epoch {epoch} - Total Loss: {avg_total_loss:.4f}, "
              f"Dice Loss Component: {avg_dice_loss:.4f}, "
              f"Avg Matches/Batch: {avg_matches_per_batch:.1f}")
    else:
        print(f"Epoch {epoch} - Total Loss: {avg_total_loss:.4f}, "
              f"Dice Loss: Skipped (dice_weight=0)")
    
    return avg_total_loss


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(model, data_loader, device):
    """
    Evaluates the model and calculates metrics with proper mask matching.
    """
    model.eval()
    dice_scores = []
    matched_pairs = 0
    total_predictions = 0
    total_ground_truth = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                if len(target['masks']) == 0:  # Skip images with no ground truth
                    continue
                    
                total_ground_truth += len(target['masks'])
                
                if len(pred['masks']) == 0:  # No predictions for this image
                    continue
                    
                total_predictions += len(pred['masks'])
                
                # Use your existing match_masks_by_iou function!
                matches = match_masks_by_iou(
                    pred['masks'].cpu(),
                    target['masks'].cpu(),
                    pred['boxes'].cpu(),
                    target['boxes'].cpu(),
                    pred['scores'].cpu(),
                    iou_threshold=0.3,
                    score_threshold=CONFIDENCE_THRESHOLD
                )
                
                # Compute dice only for matched pairs
                image_dice_scores = []
                for pred_idx, target_idx in matches:
                    dice = dice_coefficient(
                        pred['masks'][pred_idx].cpu(),
                        target['masks'][target_idx].cpu()
                    )
                    image_dice_scores.append(dice)
                    matched_pairs += 1
                
                # Add dice scores for this image
                if image_dice_scores:
                    dice_scores.extend(image_dice_scores)

    # Compute statistics
    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    recall = matched_pairs / total_ground_truth if total_ground_truth > 0 else 0.0
    precision = matched_pairs / total_predictions if total_predictions > 0 else 0.0
    
    print(f"Evaluation Results:")
    print(f"  Mean Dice Coefficient: {mean_dice:.4f}")
    print(f"  Matched pairs: {matched_pairs}")
    print(f"  Total GT masks: {total_ground_truth}")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    
    return mean_dice


def calculate_validation_loss(model, data_loader, device):
    """
    Calculates validation loss without updating model parameters.
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()}
                      for t in targets]
            
            loss_dict = model(images, targets)
            
            # Apply same weights as in training
            weighted_losses = (
                loss_dict['loss_classifier'] * 1.0 +
                loss_dict['loss_box_reg'] * 1.0 +
                loss_dict['loss_mask'] * LOSS_MASK_WEIGHT +
                loss_dict['loss_objectness'] * 1.0 +
                loss_dict['loss_rpn_box_reg'] * 1.0
            )
            
            total_loss += weighted_losses.item()
            num_batches += 1
    
    model.eval()
    return total_loss / num_batches if num_batches > 0 else 0.0

# =============================================================================
# INFERENCE FUNCTIONS FOR REPORT GENERATION
# =============================================================================

def calculate_transformation_params(original_height, original_width, 
                                  target_size=IMG_SIZE):
    """
    Calculates parameters needed to reverse the LongestMaxSize + PadIfNeeded transformation.
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
    Fixed version with proper None checking.
    """
    # Check for None input
    if mask is None:
        print("Input mask is None")
        return None
    
    # Convert to numpy if tensor
    if torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    # Check if mask_np is valid
    if mask_np is None:
        print("Converted mask is None")
        return None
    
    try:
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
        
    except Exception as e:
        print(f"Error in reverse_mask_transformation: {e}")
        return None

def inference_on_image(model, image_path, transforms):
    """
    Performs standard inference on a specific image.
    """
    # Load and process image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Use standard inference
    model.eval()
    
    # Transform image
    transformed = transforms(image=image_rgb)
    tensor_image = transformed['image'].unsqueeze(0).to(DEVICE)
    
    # Make prediction
    with torch.no_grad():
        pred = model(tensor_image)[0]
    
    # Filter by confidence threshold
    high_conf_indices = pred['scores'] > CONFIDENCE_THRESHOLD
    
    filtered_results = {
        'boxes': pred['boxes'][high_conf_indices].cpu().numpy(),
        'labels': pred['labels'][high_conf_indices].cpu().numpy(),
        'scores': pred['scores'][high_conf_indices].cpu().numpy(),
        'masks': pred['masks'][high_conf_indices].cpu().numpy()
    }
    
    return filtered_results

def create_mask_overlay(image, predictions, original_size, alpha: float = 0.6):
    """
    Creates a transparent overlay of predicted masks on the original image.
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
        0: (0, 255, 0),  # Green for Chromis
        1: (255, 0, 0),  # Blue for Coris
        2: (0, 0, 255),  # Red for additional class if needed
    }
    
    masks = predictions['masks']
    labels = predictions['labels']
    
    # Check if we have any masks to process
    if len(masks) == 0:
        print("No masks to process")
        return result_image
    
    # Process each mask
    for i in range(len(masks)):
        try:
            # Check if mask exists and is not None
            if masks[i] is None:
                print(f"Mask {i} is None, skipping")
                continue
            
            # Safely extract mask data
            mask_data = masks[i]
            
            # Handle different mask dimensions
            if mask_data.ndim == 3:
                mask = mask_data[0]  # Take the first channel if 3D
            elif mask_data.ndim == 2:
                mask = mask_data  # Use directly if 2D
            else:
                print(f"Mask {i} has unexpected dimensions: "
                      f"{mask_data.ndim}, skipping")
                continue
            
            # Check if mask has valid data
            if mask is None:
                print(f"Mask data {i} is None after extraction, skipping")
                continue
            
            label = labels[i]
            
            # Map class if necessary
            if label in CLASS_MAPPING:
                yolo_class = CLASS_MAPPING[label]
            else:
                print(f"Label {label} not in class mapping, skipping")
                continue
            
            # Ensure mask is at IMG_SIZE before reversing transformation
            if isinstance(mask, np.ndarray):
                mask_tensor = torch.from_numpy(mask)
            else:
                mask_tensor = mask
            
            # Resize mask to IMG_SIZE if it's not already
            if mask_tensor.shape[0] != IMG_SIZE or mask_tensor.shape[1] != IMG_SIZE:
                mask_resized = F.interpolate(
                    mask_tensor.unsqueeze(0).unsqueeze(0),
                    size=(IMG_SIZE, IMG_SIZE),
                    mode='bilinear',
                    align_corners=False
                )[0, 0]
                mask = mask_resized.numpy() if torch.is_tensor(mask_resized) else mask_resized
            
            # Properly reverse the transformation
            mask_original = reverse_mask_transformation(mask, transform_params)
            
            # Check if transformed mask is valid
            if mask_original is None:
                print(f"Transformed mask {i} is None, skipping")
                continue
            
            # Create binary mask
            binary_mask = (mask_original > 0.5).astype(np.uint8)
            
            # Check if binary mask has any positive pixels
            if not np.any(binary_mask):
                print(f"Binary mask {i} is empty, skipping")
                continue
            
            # Get color for this class
            color = class_colors.get(yolo_class, (255, 255, 255))
            
            # Create colored mask overlay
            colored_mask = np.zeros_like(result_image)
            colored_mask[binary_mask == 1] = color
            
            # Apply transparency only where mask exists
            mask_area = binary_mask == 1
            
            # Check if mask_area has any True values before applying
            if not np.any(mask_area):
                print(f"Mask area {i} is empty, skipping")
                continue
            
            # Ensure we have valid arrays for cv2.addWeighted
            try:
                if (result_image[mask_area].size > 0 and 
                    colored_mask[mask_area].size > 0):
                    result_image[mask_area] = cv2.addWeighted(
                        result_image[mask_area],
                        1.0 - alpha,
                        colored_mask[mask_area],
                        alpha,
                        0
                    )
                else:
                    print(f"Empty mask regions for mask {i}, skipping overlay")
            except Exception as e:
                print(f"Error applying overlay for mask {i}: {e}")
                continue
                
        except Exception as e:
            print(f"Error processing mask {i}: {e}")
            continue
    
    return result_image


def save_inference_examples(model, output_path):
    """
    Generate inference examples for the training report with enhanced error handling.
    Uses all images from the test folder instead of train/validation datasets.
    """
    model.eval()
    transforms = get_inference_transforms()
    
    try:
        # Get list of all test images
        test_image_files = [f for f in os.listdir(TEST_IMAGES_PATH) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(test_image_files) == 0:
            print("No test images found.")
            return []
        
        processed_paths = []
        
        # Process all test images
        for idx, image_file in enumerate(test_image_files):
            test_image_path = os.path.join(TEST_IMAGES_PATH, image_file)
            print(f"Processing test image {idx+1}/{len(test_image_files)}: {test_image_path}")
            
            with Image.open(test_image_path) as img:
                test_original_size = (img.height, img.width)
            
            test_predictions = inference_on_image(
                model, test_image_path, transforms
            )
            
            test_image = cv2.imread(test_image_path)
            test_overlay = create_mask_overlay(
                test_image, test_predictions, test_original_size, alpha=0.4
            )
            
            # Save with numbered filename
            test_output_path = os.path.join(output_path, f"test_inference_{idx+1}.jpg")
            cv2.imwrite(test_output_path, test_overlay)
            print(f"Test inference {idx+1} saved: {test_output_path}")
            
            processed_paths.append(test_output_path)
        
        return processed_paths
        
    except Exception as e:
        print(f"Error in save_inference_examples: {e}")
        return []



# =============================================================================
# PDF REPORT GENERATION
# =============================================================================

def generate_training_report(train_losses, val_losses, final_dice, 
                           test_image_paths, 
                           training_start_time, training_end_time):
    """
    Generate a comprehensive PDF training report.
    """
    if not REPORTLAB_AVAILABLE:
        print("ReportLab not available. Install with: pip install reportlab")
        return None
    
    # Create output directory
    os.makedirs(REPORT_OUTPUT_PATH, exist_ok=True)
    
    # Generate report filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = (f"{timestamp}__"
        f"{os.path.basename(MODEL_PATH).split('.')[0]}.pdf")
    report_path = os.path.join(REPORT_OUTPUT_PATH, report_filename)
    
    # Create PDF document
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    story.append(Paragraph("Mask R-CNN Training Report", title_style))
    story.append(Spacer(1, 20))
    
    # Training Summary
    summary_style = ParagraphStyle(
        'Summary',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12,
        alignment=TA_LEFT
    )
    
    training_duration = training_end_time - training_start_time
    
    story.append(Paragraph("Training Summary", styles['Heading2']))
    story.append(Paragraph(f"<b>Training Started:</b> {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}", summary_style))
    story.append(Paragraph(f"<b>Training Completed:</b> {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}", summary_style))
    story.append(Paragraph(f"<b>Total Duration:</b> {str(training_duration).split('.')[0]}", summary_style))
    story.append(Paragraph(f"<b>Final Training Loss:</b> {train_losses[-1]:.4f}", summary_style))
    story.append(Paragraph(f"<b>Final Validation Loss:</b> {val_losses[-1]:.4f}", summary_style))
    story.append(Paragraph(f"<b>Best Validation Loss:</b> {min(val_losses):.4f}", summary_style))
    story.append(Paragraph(f"<b>Final Dice Coefficient:</b> {(final_dice):.4f}", summary_style))
    story.append(Spacer(1, 20))
    
    # Configuration Table - Automatic Global Variable Extraction
    story.append(Paragraph("Training Configuration", styles['Heading2']))

    def extract_global_config_variables():
        """
        Systematically extract all global configuration variables.
        Filters for uppercase constants (standard Python convention).
        """
        import types
        
        config_vars = {}
        
        # Get all global variables
        all_globals = globals()
        
        # Filter for configuration variables (uppercase constants)
        for name, value in all_globals.items():
            if (name.isupper() and 
                not name.startswith('_') and 
                not callable(value) and
                not isinstance(value, types.ModuleType) and
                name not in ['REPORTLAB_AVAILABLE', 'A4', 'TA_CENTER', 'TA_LEFT']):
                
                # Convert complex objects to readable strings
                if isinstance(value, (dict, list, tuple)):
                    config_vars[name] = str(value)
                elif hasattr(value, '__name__'):
                    config_vars[name] = str(value)
                else:
                    config_vars[name] = str(value)
        
        return config_vars


    # Extract all configuration variables automatically
    config_variables = extract_global_config_variables()

    # Sort by category for better organization
    dataset_vars = {k: v for k, v in config_variables.items() 
                   if any(keyword in k for keyword in ['DATASET', 'PATH', 'ANNOTATIONS'])}
    training_vars = {k: v for k, v in config_variables.items() 
                    if any(keyword in k for keyword in ['NUM_', 'BATCH', 'LEARNING', 'EPOCHS'])}
    model_vars = {k: v for k, v in config_variables.items() 
                 if any(keyword in k for keyword in ['IMG_', 'CONFIDENCE', 'DICE', 'USE_'])}
    rpn_vars = {k: v for k, v in config_variables.items() if k.startswith('RPN_')}
    output_vars = {k: v for k, v in config_variables.items() 
                  if any(keyword in k for keyword in ['REPORT', 'TEMP'])}
    class_vars = {k: v for k, v in config_variables.items() if 'CLASS' in k}
    other_vars = {k: v for k, v in config_variables.items() 
                 if k not in {**dataset_vars, **training_vars, **model_vars, 
                             **rpn_vars, **output_vars, **class_vars} and k != 'DEVICE'}

    # Build the configuration table data
    config_data = [['Parameter', 'Value']]

    # Add categorized sections
    def add_section(section_name, variables):
        if variables:
            config_data.append([f"=== {section_name} ===", ""])
            for name, value in sorted(variables.items()):
                config_data.append([name, value])

    add_section("Dataset Configuration", dataset_vars)
    add_section("Training Parameters", training_vars)
    add_section("Model Parameters", model_vars)
    add_section("RPN Parameters", rpn_vars)
    add_section("Output Configuration", output_vars)
    add_section("Class Configuration", class_vars)

    # Add device and other variables
    config_data.append(["=== System ===", ""])
    config_data.append(["DEVICE", str(config_variables.get('DEVICE', 'Not found'))])

    # Add any remaining variables
    if other_vars:
        add_section("Other Parameters", other_vars)

    config_table = Table(config_data, colWidths=[3*inch, 2*inch])
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    # Apply special styling to section headers
    for i, row in enumerate(config_data):
        if len(row) > 0 and row[0].startswith("==="):
            config_table.setStyle(TableStyle([
                ('BACKGROUND', (0, i), (-1, i), colors.darkgrey),
                ('TEXTCOLOR', (0, i), (-1, i), colors.white),
                ('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold'),
            ]))

    story.append(config_table)
    story.append(PageBreak())
    
    # Training Plots
    story.append(Paragraph("Training Progress", styles['Heading2']))
    
    # Add training plots
    training_plots_path = os.path.join(TEMP_FIGURES_PATH, "training_metrics.png")
    if os.path.exists(training_plots_path):
        story.append(RLImage(training_plots_path, width=7*inch, height=2.5*inch))
    story.append(PageBreak())
    
    # Test Set Inference Examples
    story.append(Paragraph("Test Set Inference Examples", styles['Heading2']))
    
    # Add all test inference images
    for idx, test_image_path in enumerate(test_image_paths):
        story.append(Paragraph(f"Test Image {idx+1}", styles['Heading3']))
        if test_image_path and os.path.exists(test_image_path):
            story.append(RLImage(test_image_path, width=6*inch, height=4*inch))
        story.append(Spacer(1, 20))

    
    # Build PDF
    doc.build(story)
    print(f"Training report generated: {report_path}")
    return report_path


# =============================================================================
# MAIN FUNCTION WITH ARGUMENT SUPPORT
# =============================================================================

def main():
    """
    Main function with enhanced training pipeline and argument support.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Update global variables with arguments
    update_global_variables(args)

    # Update MODEL_PATH global variable
    global MODEL_PATH
    MODEL_PATH = args.model_path  # Use the path provided by main.py
    
    print(f"Model will be saved to: {MODEL_PATH}")
    
    # Ensure the model directory exists
    model_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    # Continue with training
    training_start_time = datetime.now()
    print(f"Using device: {DEVICE}")
    print(f"Using class mapping: {CLASS_MAPPING}")
    print(f"Using class names: {CLASS_NAMES}")
    print(f"Training started at: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directories
    os.makedirs(REPORT_OUTPUT_PATH, exist_ok=True)
    os.makedirs(TEMP_FIGURES_PATH, exist_ok=True)

    # Create enhanced datasets
    print("Loading datasets with small object optimizations...")
    train_dataset = COCOInstanceDataset(
        TRAIN_IMAGES_PATH,
        TRAIN_ANNOTATIONS,
        transforms=get_train_transforms(),
        oversample_small_objects=OVERSAMPLE_SMALL_OBJECTS,
        use_copy_paste=USE_COPY_PASTE
    )

    val_dataset = COCOInstanceDataset(
        VAL_IMAGES_PATH,
        VAL_ANNOTATIONS,
        transforms=get_val_transforms(),
        oversample_small_objects=False,
        use_copy_paste=False
    )

    print(f"Training dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")

    def collate_fn(batch):
        if len(batch) == 1:
            # Handle single-item batch explicitly
            images, targets = batch[0]
            return [images], [targets]
        else:
            # Handle multi-item batches
            return tuple(zip(*batch))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Create enhanced model
    print("Initializing enhanced Mask R-CNN model...")
    model = get_model_instance_segmentation(NUM_CLASSES, MASK_RESOLUTION)
    model.to(DEVICE)

    # Enhanced optimizer and scheduler setup
    optimizer = optim.AdamW(model.parameters(), 
                        lr=LEARNING_RATE,
                        weight_decay=0.0005,
                        betas=(0.9, 0.999),
                        eps=1e-8)


    # # Learning rate scheduler
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
    #                                           milestones=[100, 150],
    #                                           gamma=0.1)

    # # Warmup scheduler for first few epochs
    # warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer,
    #                                               start_factor=0.1,
    #                                               total_iters=10)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=15,
        min_lr=1e-7
    )

    # Training metrics
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # Track best validation loss

    print("Starting enhanced training for small objects...")

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        # print(f"Memory before epoch: "
        #     f"{torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, "
        #     f"{torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
        
        # Training
        train_loss = train_one_epoch(
            model,
            optimizer,
            train_loader,
            DEVICE,
            epoch+1,
            dice_weight=DICE_WEIGHT,
            use_focal_dice=USE_FOCAL_DICE
        )
        train_losses.append(train_loss)
        del train_loss
        clear_gpu_memory()
        
        # # Update learning rate
        # if epoch < 10:
        #     warmup_scheduler.step()
        # else:
        #     scheduler.step()
        
        # Validation
        val_loss = calculate_validation_loss(model, val_loader, DEVICE)
        val_losses.append(val_loss)
        
        print(f"Training loss: {train_losses[-1]:.4f}")
        print(f"Validation loss: {val_losses[-1]:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        # print(f"Memory after epoch: "
        #     f"{torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, "
        #     f"{torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
        
        scheduler.step(val_losses[-1])

        # Save model when validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved (val_loss improved to {val_loss:.4f}): {MODEL_PATH}")

        # Save model every 25 epochs starting at epoch 100
        elif (epoch + 1) >= 100 and (epoch + 1) % 25 == 0:
            epoch_path = f"weights/epoch{epoch + 1:03d}.pth"
            os.makedirs(os.path.dirname(epoch_path), exist_ok=True)
            torch.save(model.state_dict(), epoch_path)
            print(f"Model saved at epoch {epoch + 1}: {epoch_path}")

        del val_loss
        clear_gpu_memory()

    # Record training end time
    training_end_time = datetime.now()
    print(f"Training completed at: {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Final evaluation
    print("\n=== FINAL EVALUATION ===")
    model.load_state_dict(torch.load(MODEL_PATH))
    final_dice = evaluate_model(model, val_loader, DEVICE)
    print(f"Final Dice coefficient on validation: {final_dice:.4f}")

    # Generate and save training plots (without displaying)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(val_losses, color='orange')
    plt.title('Validation Loss Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    training_plots_path = os.path.join(TEMP_FIGURES_PATH, "training_metrics.png")
    plt.savefig(training_plots_path, dpi=300, bbox_inches='tight')
    plt.close() # Close instead of show to avoid display

    # Generate inference examples
    print("Generating inference examples for report over test dataset.")
    test_inference_paths = save_inference_examples(model, TEMP_FIGURES_PATH)

    # Generate PDF report
    print("Generating PDF training report...")
    report_path = generate_training_report(
        train_losses, val_losses, final_dice,
        test_inference_paths,
        training_start_time, training_end_time
    )

    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    if report_path:
        print(f"Training report saved: {report_path}")

# =============================================================================
# EXECUTE SCRIPT
# =============================================================================

if __name__ == "__main__":
    main()