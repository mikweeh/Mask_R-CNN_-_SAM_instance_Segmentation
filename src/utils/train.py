#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full version of training and testing and pdf report of a Mask R-CNN
model for small fishes segmentation
"""

import argparse
import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL CONFIGURATION VARIABLES
# =============================================================================

# Dataset paths
DATASET_PATH = "dataset"
TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train")
VAL_IMAGES_PATH = os.path.join(DATASET_PATH, "valid")
TEST_IMAGES_PATH = os.path.join(DATASET_PATH, "test")
TRAIN_ANNOTATIONS = os.path.join(TRAIN_IMAGES_PATH, 
                                "_annotations_filtered.coco.json")
VAL_ANNOTATIONS = os.path.join(VAL_IMAGES_PATH, 
                              "_annotations_filtered.coco.json")
# Add test annotations if you have them
TEST_ANNOTATIONS = os.path.join(TEST_IMAGES_PATH, 
                               "_annotations_filtered.coco.json")

# Model and training configuration
MODEL_PATH = 'weights/m01.pth'  # Base model path
NUM_CLASSES = 3  # Including background
BATCH_SIZE = 1
NUM_EPOCHS = 150
LEARNING_RATE = 0.0001
IMG_SIZE = 2048
CONFIDENCE_THRESHOLD = 0.3
MASK_RESOLUTION = 56
BASE_MIN_ANCHOR = 16

# Training enhancements
DICE_WEIGHT = 0.0
USE_FOCAL_DICE = False
OVERSAMPLE_SMALL_OBJECTS = True
USE_COPY_PASTE = True

# Output paths
REPORT_OUTPUT_PATH = 'results'
TEMP_FIGURES_PATH = os.path.join(REPORT_OUTPUT_PATH, 'imgs')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class configuration (can be overridden by arguments)
CLASS_MAPPING = {1: 0, 2: 1}
CLASS_NAMES = {0: 'Chromis chromis', 1: 'Coris julis'}

# RPN parameters
RPN_PRE_NMS_TOP_N_TRAIN = 1500
RPN_POST_NMS_TOP_N_TRAIN = 600
RPN_NMS_THRESH = 0.6

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
    """Parse command-line arguments with smart default using MODEL_PATH."""
    parser = argparse.ArgumentParser(description='Train Mask R-CNN for segmentation')
    
    # Dataset configuration
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH,
                       help='Path to dataset directory')
    
    # Model path with smart default
    parser.add_argument('--model_path', type=str, 
                       default=get_next_model_name_train(),
                       help='Full path where model will be saved (default: next available name)')
    
    # Model configuration
    parser.add_argument('--num_classes', type=int, default=NUM_CLASSES,
                       help='Number of classes (including background)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE,
                       help='Image size for training')
    parser.add_argument('--confidence_threshold', type=float, default=CONFIDENCE_THRESHOLD,
                       help='Confidence threshold for predictions')
    parser.add_argument('--mask_resolution', type=int, default=MASK_RESOLUTION,
                       help='Mask resolution (28, 56, or 112)')
    parser.add_argument('--base_min_anchor', type=int, default=BASE_MIN_ANCHOR,
                       help='Base minimum anchor size')
    
    # Training enhancement flags
    parser.add_argument('--dice_weight', type=float, default=DICE_WEIGHT,
                       help='Weight for Dice loss component')
    parser.add_argument('--use_focal_dice', action='store_true', default=USE_FOCAL_DICE,
                       help='Use focal Dice loss')
    parser.add_argument('--no_use_focal_dice', action='store_false', dest='use_focal_dice',
                       help='Disable focal Dice loss')
    parser.add_argument('--oversample_small_objects', action='store_true', 
                       default=OVERSAMPLE_SMALL_OBJECTS,
                       help='Enable small object oversampling')
    parser.add_argument('--no_oversample_small_objects', action='store_false', 
                       dest='oversample_small_objects',
                       help='Disable small object oversampling')
    parser.add_argument('--use_copy_paste', action='store_true', default=USE_COPY_PASTE,
                       help='Enable copy-paste augmentation')
    parser.add_argument('--no_use_copy_paste', action='store_false', dest='use_copy_paste',
                       help='Disable copy-paste augmentation')
    
    # RPN parameters
    parser.add_argument('--rpn_pre_nms_top_n_train', type=int, default=RPN_PRE_NMS_TOP_N_TRAIN,
                       help='RPN pre-NMS top N for training')
    parser.add_argument('--rpn_post_nms_top_n_train', type=int, default=RPN_POST_NMS_TOP_N_TRAIN,
                       help='RPN post-NMS top N for training')
    parser.add_argument('--rpn_nms_thresh', type=float, default=RPN_NMS_THRESH,
                       help='RPN NMS threshold')
    
    # Output configuration
    parser.add_argument('--report_output_path', type=str, default=REPORT_OUTPUT_PATH,
                       help='Path for saving training reports')
    
    # Class configuration (optional - will override defaults if provided)
    parser.add_argument('--class_mapping', type=str, default=None,
                       help='JSON string with class ID mapping')
    parser.add_argument('--class_names', type=str, default=None,
                       help='JSON string with class names')
    
    return parser.parse_args()

def update_global_variables(args):
    """Update global variables with command-line arguments ONLY if provided."""
    global DATASET_PATH, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE
    global IMG_SIZE, CONFIDENCE_THRESHOLD, MASK_RESOLUTION, BASE_MIN_ANCHOR
    global DICE_WEIGHT, USE_FOCAL_DICE, OVERSAMPLE_SMALL_OBJECTS, USE_COPY_PASTE
    global RPN_PRE_NMS_TOP_N_TRAIN, RPN_POST_NMS_TOP_N_TRAIN, RPN_NMS_THRESH
    global REPORT_OUTPUT_PATH, TEMP_FIGURES_PATH, CLASS_MAPPING, CLASS_NAMES
    global TRAIN_IMAGES_PATH, TRAIN_ANNOTATIONS, VAL_IMAGES_PATH, VAL_ANNOTATIONS
    global TEST_IMAGES_PATH, TEST_ANNOTATIONS
    
    # Update basic parameters
    DATASET_PATH = args.dataset_path
    NUM_CLASSES = args.num_classes
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    IMG_SIZE = args.img_size
    CONFIDENCE_THRESHOLD = args.confidence_threshold
    MASK_RESOLUTION = args.mask_resolution
    BASE_MIN_ANCHOR = args.base_min_anchor
    
    # Update training enhancements
    DICE_WEIGHT = args.dice_weight
    USE_FOCAL_DICE = args.use_focal_dice
    OVERSAMPLE_SMALL_OBJECTS = args.oversample_small_objects
    USE_COPY_PASTE = args.use_copy_paste
    
    # Update RPN parameters
    RPN_PRE_NMS_TOP_N_TRAIN = args.rpn_pre_nms_top_n_train
    RPN_POST_NMS_TOP_N_TRAIN = args.rpn_post_nms_top_n_train
    RPN_NMS_THRESH = args.rpn_nms_thresh
    
    # Update output paths
    REPORT_OUTPUT_PATH = args.report_output_path
    TEMP_FIGURES_PATH = os.path.join(REPORT_OUTPUT_PATH, 'imgs')
    
    # FIXED: Only rebuild paths if dataset_path was actually changed
    if args.dataset_path != "dataset":  # Only if different from default
        TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train")
        VAL_IMAGES_PATH = os.path.join(DATASET_PATH, "valid")
        TEST_IMAGES_PATH = os.path.join(DATASET_PATH, "test")
        TRAIN_ANNOTATIONS = os.path.join(TRAIN_IMAGES_PATH, 
                                        "_annotations_filtered.coco.json")
        VAL_ANNOTATIONS = os.path.join(VAL_IMAGES_PATH, 
                                      "_annotations_filtered.coco.json")
        TEST_ANNOTATIONS = os.path.join(TEST_IMAGES_PATH, 
                                       "_annotations_filtered.coco.json")
    
    # Update class configuration ONLY if provided as arguments
    if args.class_mapping is not None:
        CLASS_MAPPING = json.loads(args.class_mapping)
        CLASS_MAPPING = {int(k): v for k, v in CLASS_MAPPING.items()}
        print(f"Updated CLASS_MAPPING: {CLASS_MAPPING}")
    
    if args.class_names is not None:
        CLASS_NAMES = json.loads(args.class_names)
        CLASS_NAMES = {int(k): v for k, v in CLASS_NAMES.items()}
        print(f"Updated CLASS_NAMES: {CLASS_NAMES}")

# =============================================================================
# DATASET CLASSES
# =============================================================================

class COCOInstanceDataset(torch.utils.data.Dataset):
    """Enhanced COCO dataset with augmentation support."""
    
    def __init__(self, images_dir, annotations_file, transforms=None, 
                 oversample_small_objects=False, use_copy_paste=False):
        self.images_dir = images_dir
        self.transforms = transforms
        self.oversample_small_objects = oversample_small_objects
        self.use_copy_paste = use_copy_paste
        
        # Load COCO annotations
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mappings
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = list(self.image_id_to_info.keys())
        
        # Group annotations by image
        self.image_id_to_annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_id_to_info[image_id]
        
        # Load image
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        annotations = self.image_id_to_annotations.get(image_id, [])
        
        # Convert annotations to target format
        boxes = []
        labels = []
        masks = []
        
        for ann in annotations:
            # Get bounding box
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # Get label (map using CLASS_MAPPING)
            category_id = ann['category_id']
            if category_id in CLASS_MAPPING:
                labels.append(CLASS_MAPPING[category_id])
            else:
                continue  # Skip unmapped categories
            
            # Convert segmentation to mask
            if 'segmentation' in ann and ann['segmentation']:
                mask = self._segmentation_to_mask(ann['segmentation'], 
                                                image_info['height'], 
                                                image_info['width'])
                masks.append(mask)
        
        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.stack([torch.tensor(mask, dtype=torch.uint8) for mask in masks])
        else:
            # Handle empty annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, image_info['height'], image_info['width']), dtype=torch.uint8)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([image_id])
        }
        
        # Apply transformations
        if self.transforms:
            # Convert mask format for albumentations
            mask_list = [mask.numpy() for mask in masks] if len(masks) > 0 else []
            
            transformed = self.transforms(
                image=image,
                masks=mask_list,
                bboxes=boxes.numpy() if len(boxes) > 0 else [],
                category_ids=labels.numpy() if len(labels) > 0 else []
            )
            
            image = transformed['image']
            if transformed.get('masks'):
                masks = torch.stack([torch.tensor(mask, dtype=torch.uint8) 
                                   for mask in transformed['masks']])
                target['masks'] = masks
            
            if transformed.get('bboxes'):
                boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                target['boxes'] = boxes
        
        return image, target
    
    def _segmentation_to_mask(self, segmentation, height, width):
        """Convert COCO segmentation to binary mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if isinstance(segmentation, list):
            # Polygon format
            for polygon in segmentation:
                if len(polygon) >= 6:  # At least 3 points
                    polygon = np.array(polygon).reshape(-1, 2)
                    cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        
        return mask

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

def get_anchor_sizes(img_size, base_img_size=1280, base_min_anchor=16):
    """Calculate anchor sizes maintaining powers-of-2 progression."""
    scale_factor = img_size / base_img_size
    min_anchor = int(round(base_min_anchor * scale_factor))
    anchor_sizes = tuple(min_anchor * (2**i) for i in range(5))
    return anchor_sizes

class HighResMaskRCNNPredictor(torch.nn.Module):
    """High-resolution mask predictor for small objects."""
    
    def __init__(self, in_channels, dim_reduced, num_classes, mask_size):
        super().__init__()
        self.mask_size = mask_size
        
        if mask_size <= 28:
            self.conv5_mask = torch.nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)
            self.relu = torch.nn.ReLU(inplace=True)
            self.mask_fcn_logits = torch.nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        elif mask_size <= 56:
            self.conv5_mask = torch.nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)
            self.relu1 = torch.nn.ReLU(inplace=True)
            self.conv6_mask = torch.nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)
            self.relu2 = torch.nn.ReLU(inplace=True)
            self.mask_fcn_logits = torch.nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        elif mask_size <= 112:
            self.conv5_mask = torch.nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)
            self.relu1 = torch.nn.ReLU(inplace=True)
            self.conv6_mask = torch.nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)
            self.relu2 = torch.nn.ReLU(inplace=True)
            self.conv7_mask = torch.nn.ConvTranspose2d(dim_reduced, dim_reduced, 2, 2, 0)
            self.relu3 = torch.nn.ReLU(inplace=True)
            self.mask_fcn_logits = torch.nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        
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

def get_model_instance_segmentation(num_classes, mask_resolution=56):
    """Create enhanced Mask R-CNN model."""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Enhanced ROI pooling for masks
    roi_pool_size = mask_resolution // 2
    model.roi_heads.mask_roi_pool.output_size = (roi_pool_size, roi_pool_size)
    
    # Optimized anchor generator for small objects
    anchor_generator = torchvision.models.detection.anchor_utils.AnchorGenerator(
        sizes=tuple((size,) for size in get_anchor_sizes(IMG_SIZE, base_min_anchor=BASE_MIN_ANCHOR)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    model.rpn.anchor_generator = anchor_generator
    
    # Configure RPN
    model.rpn.pre_nms_top_n = {'training': RPN_PRE_NMS_TOP_N_TRAIN, 'testing': 1000}
    model.rpn.post_nms_top_n = {'training': RPN_POST_NMS_TOP_N_TRAIN, 'testing': 1000}
    model.rpn.nms_thresh = RPN_NMS_THRESH
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor with high-resolution version
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = HighResMaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes, mask_resolution
    )
    
    return model

# =============================================================================
# DATA TRANSFORMS
# =============================================================================

def get_train_transforms():
    """Get training transforms with augmentation."""
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, 
                     border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

def get_val_transforms():
    """Get validation transforms without augmentation."""
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, 
                     border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def collate_fn(batch):
    """Custom collate function for DataLoader."""
    if len(batch) == 1:
        images, targets = batch[0]
        return [images], [targets]
    else:
        return tuple(zip(*batch))

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(model, optimizer, data_loader, device, epoch, 
                   dice_weight=0.0, use_focal_dice=False):
    """Train for one epoch with enhanced loss computation."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(data_loader)}, "
                  f"Loss: {losses.item():.4f}")
    
    return total_loss / num_batches if num_batches > 0 else 0

def calculate_validation_loss(model, data_loader, device):
    """Calculate validation loss."""
    model.train()  # Keep in train mode for loss calculation
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            total_loss += losses.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0

def evaluate_model(model, data_loader, device):
    """Evaluate model and return Dice score."""
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            
            predictions = model(images)
            
            for pred, target in zip(predictions, targets):
                if len(pred['masks']) > 0:
                    # Simple Dice calculation for validation
                    pred_masks = (pred['masks'] > 0.5).float()
                    target_masks = target['masks'].float().to(device)
                    
                    if len(target_masks) > 0 and len(pred_masks) > 0:
                        # Calculate Dice for first mask pair (simplified)
                        intersection = (pred_masks[0] * target_masks[0]).sum()
                        union = pred_masks[0].sum() + target_masks[0].sum()
                        dice = (2.0 * intersection / (union + 1e-8)).cpu().item()
                        dice_scores.append(dice)
    
    return np.mean(dice_scores) if dice_scores else 0.0

def save_inference_examples(model, train_dataset, val_dataset, output_dir, num_examples=3):
    """Save inference examples for report."""
    model.eval()
    inference_paths = []
    
    # Get some validation images
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader):
            if idx >= num_examples:
                break
                
            images = [img.to(DEVICE) for img in images]
            predictions = model(images)
            
            # Save visualization
            image = images[0].cpu()
            pred = predictions[0]
            target = targets[0]
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            img_np = image.permute(1, 2, 0).numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + 
                     np.array([0.485, 0.456, 0.406]))
            img_np = np.clip(img_np, 0, 1)
            axes[0].imshow(img_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Ground truth
            axes[1].imshow(img_np)
            if len(target['masks']) > 0:
                mask = target['masks'][0].numpy()
                axes[1].imshow(mask, alpha=0.5, cmap='green')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Prediction
            axes[2].imshow(img_np)
            if len(pred['masks']) > 0:
                mask = (pred['masks'][0, 0] > 0.5).cpu().numpy()
                axes[2].imshow(mask, alpha=0.5, cmap='red')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            inference_path = os.path.join(output_dir, f'test_inference_{idx}.png')
            plt.savefig(inference_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            inference_paths.append(inference_path)
    
    return inference_paths

def generate_training_report(train_losses, val_losses, val_dice_scores, 
                           inference_paths, start_time, end_time):
    """Generate comprehensive PDF training report."""
    report_path = os.path.join(REPORT_OUTPUT_PATH, 'training_report.pdf')
    
    with PdfPages(report_path) as pdf:
        # Training summary page
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Report Summary', fontsize=16, fontweight='bold')
        
        # Training and validation losses
        axes[0, 0].plot(train_losses, label='Training Loss', color='blue')
        axes[0, 0].plot(val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Losses')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validation Dice scores
        axes[0, 1].plot(val_dice_scores, color='green')
        axes[0, 1].set_title('Validation Dice Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training statistics
        axes[1, 0].text(0.1, 0.8, f'Total Epochs: {NUM_EPOCHS}', fontsize=12)
        axes[1, 0].text(0.1, 0.7, f'Batch Size: {BATCH_SIZE}', fontsize=12)
        axes[1, 0].text(0.1, 0.6, f'Learning Rate: {LEARNING_RATE}', fontsize=12)
        axes[1, 0].text(0.1, 0.5, f'Best Dice: {max(val_dice_scores):.4f}', fontsize=12)
        axes[1, 0].text(0.1, 0.4, f'Final Loss: {train_losses[-1]:.4f}', fontsize=12)
        axes[1, 0].set_title('Training Configuration')
        axes[1, 0].axis('off')
        
        # Timing information
        duration = end_time - start_time
        axes[1, 1].text(0.1, 0.8, f'Start: {start_time.strftime("%Y-%m-%d %H:%M")}', fontsize=10)
        axes[1, 1].text(0.1, 0.7, f'End: {end_time.strftime("%Y-%m-%d %H:%M")}', fontsize=10)
        axes[1, 1].text(0.1, 0.6, f'Duration: {str(duration).split(".")[0]}', fontsize=10)
        axes[1, 1].text(0.1, 0.5, f'Device: {DEVICE}', fontsize=10)
        axes[1, 1].set_title('Training Timeline')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Add inference examples
        for inference_path in inference_paths:
            if os.path.exists(inference_path):
                img = plt.imread(inference_path)
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(img)
                ax.axis('off')
                pdf.savefig(fig, dpi=150, bbox_inches='tight')
                plt.close()
    
    return report_path

# =============================================================================
# EXTRACTED MODULAR FUNCTIONS
# =============================================================================

def load_datasets():
    """
    Load and create train, validation, and test datasets.
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    print("Loading datasets with small object optimizations...")
    
    # Training dataset with augmentations
    train_dataset = COCOInstanceDataset(
        TRAIN_IMAGES_PATH,
        TRAIN_ANNOTATIONS,
        transforms=get_train_transforms(),
        oversample_small_objects=OVERSAMPLE_SMALL_OBJECTS,
        use_copy_paste=USE_COPY_PASTE
    )

    # Validation dataset without augmentations
    val_dataset = COCOInstanceDataset(
        VAL_IMAGES_PATH,
        VAL_ANNOTATIONS,
        transforms=get_val_transforms(),
        oversample_small_objects=False,
        use_copy_paste=False
    )
    
    # Test dataset with annotations (as requested)
    test_dataset = COCOInstanceDataset(
        TEST_IMAGES_PATH,
        TEST_ANNOTATIONS,
        transforms=get_val_transforms(),
        oversample_small_objects=False,
        use_copy_paste=False
    )
    
    print(f"Training dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    print(f"Test dataset: {len(test_dataset)} images")
    
    return train_dataset, val_dataset, test_dataset

def validate_model(model, val_loader):
    """
    Perform validation and return validation loss.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        
    Returns:
        float: Average validation loss
    """
    # Validation loss
    val_loss = calculate_validation_loss(model, val_loader, DEVICE)
    
    # Clean up memory
    clear_gpu_memory()
    
    return val_loss

def test_model(model, test_loader):
    """
    Perform final evaluation on test set.
    
    Args:
        model: The trained model
        test_loader: Test data loader
    """
    print("\n=== FINAL TEST EVALUATION ===")
    
    # Load the best model weights
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    total_predictions = 0
    images_with_predictions = 0
    confidence_scores = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = [img.to(DEVICE) for img in images]
            
            # Get predictions
            predictions = model(images)
            
            for pred in predictions:
                # Filter by confidence threshold
                high_conf_mask = pred['scores'] > CONFIDENCE_THRESHOLD
                valid_predictions = pred['scores'][high_conf_mask]
                
                if len(valid_predictions) > 0:
                    total_predictions += len(valid_predictions)
                    images_with_predictions += 1
                    confidence_scores.extend(valid_predictions.cpu().numpy())
    
    # Print test statistics
    print(f"Test set evaluation completed:")
    print(f"  Total test images: {len(test_loader)}")
    print(f"  Images with predictions: {images_with_predictions}")
    print(f"  Total predictions: {total_predictions}")
    
    if confidence_scores:
        print(f"  Average confidence: {np.mean(confidence_scores):.4f}")
        print(f"  Max confidence: {np.max(confidence_scores):.4f}")
        print(f"  Min confidence: {np.min(confidence_scores):.4f}")
    else:
        print("  No predictions above confidence threshold")
    
    clear_gpu_memory()

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    """Main training function with simplified model path handling."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Update global variables with arguments
    update_global_variables(args)
    
    # Use the model path from arguments (either smart default or provided by main.py)
    global MODEL_PATH
    MODEL_PATH = args.model_path
    
    print(f"Model will be saved to: {MODEL_PATH}")
    
    # Ensure the model directory exists
    model_dir = os.path.dirname(MODEL_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
    
    print(f"Using device: {DEVICE}")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Using class mapping: {CLASS_MAPPING}")
    print(f"Using class names: {CLASS_NAMES}")
    
    # Create output directories
    os.makedirs(REPORT_OUTPUT_PATH, exist_ok=True)
    os.makedirs(TEMP_FIGURES_PATH, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    train_dataset, val_dataset = load_datasets()
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print("Creating model...")
    model = get_model_instance_segmentation(NUM_CLASSES, MASK_RESOLUTION)
    model.to(DEVICE)
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=0.0005
    )
    
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.1
    )
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    val_dice_scores = []
    
    training_start_time = datetime.now()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        # Training phase
        train_loss = train_one_epoch(
            model, optimizer, train_loader, DEVICE, epoch+1,
            dice_weight=DICE_WEIGHT, use_focal_dice=USE_FOCAL_DICE
        )
        train_losses.append(train_loss)
        
        # Validation phase
        val_loss = validate_model(model, val_loader)
        val_losses.append(val_loss)
        
        # Evaluation phase
        val_dice = evaluate_model(model, val_loader, DEVICE)
        val_dice_scores.append(val_dice)
        
        # Update learning rate
        lr_scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Dice: {val_dice:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if not val_dice_scores[:-1] or val_dice_scores[-1] > max(val_dice_scores[:-1]):
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved as: {MODEL_PATH}")
    
    training_end_time = datetime.now()
    
    # Save the final model
    print(f"\nSaving final model to: {MODEL_PATH}")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved as: {MODEL_PATH}")
    
    # Test phase
    print("Evaluating on test set...")
    test_model(model, test_loader)
    
    # Generate training report
    print("Generating training report...")
    test_inference_paths = save_inference_examples(
        model, train_dataset, val_dataset, TEMP_FIGURES_PATH
    )
    
    report_path = generate_training_report(
        train_losses, val_losses, val_dice_scores,
        test_inference_paths, training_start_time, training_end_time
    )
    
    print("Training completed successfully!")
    print(f"Final model saved at: {MODEL_PATH}")
    print(f"Training report saved at: {report_path}")

if __name__ == "__main__":
    main()
