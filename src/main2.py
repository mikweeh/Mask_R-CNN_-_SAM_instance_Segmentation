#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL CONFIGURATION VARIABLES (ENHANCED FOR SMALL OBJECTS)
# =============================================================================

# Dataset paths
DATASET_PATH = "dataset"
TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train")
VAL_IMAGES_PATH = os.path.join(DATASET_PATH, "valid")
TRAIN_ANNOTATIONS = os.path.join(TRAIN_IMAGES_PATH, "_annotations_filtered.coco.json")
VAL_ANNOTATIONS = os.path.join(VAL_IMAGES_PATH, "_annotations_filtered.coco.json")

# Enhanced training parameters for small objects
NUM_CLASSES = 3
BATCH_SIZE = 1
NUM_EPOCHS = 200
LEARNING_RATE = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enhanced model parameters for small objects
IMG_SIZE = 2048
CONFIDENCE_THRESHOLD = 0.3  # Lower threshold for small objects
USE_FOCAL_DICE = True

# RPN parameters optimized for small objects
RPN_PRE_NMS_TOP_N_TRAIN = 3000
RPN_POST_NMS_TOP_N_TRAIN = 1000
RPN_NMS_THRESH = 0.6

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
        max_copies: Maximum number of copies per object (reduced from 3 to 2)
        
    Returns:
        Augmented image, masks, boxes, and labels
    """
    if len(boxes) == 0:
        return image, masks, boxes, labels
        
    # Limit augmentation for very small datasets or single-item batches
    if len(boxes) > 10:  # Skip copy-paste if too many objects already
        return image, masks, boxes, labels
        
    h, w = image.shape[:2]
    new_masks = masks.copy()
    new_boxes = boxes.copy()
    new_labels = labels.copy()
    
    # Reduce number of copies for memory efficiency
    max_copies = min(max_copies, 2)
    
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
            A.LongestMaxSize(max_size=800),
            A.LongestMaxSize(max_size=1024),
            A.LongestMaxSize(max_size=1280),
        ], p=1.0),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, 
                     border_mode=cv2.BORDER_CONSTANT, p=1.0),
        
        # Enhanced augmentations for small objects
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.3),
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

# =============================================================================
# Manage GPU memory
# =============================================================================
def clear_gpu_memory():
    """
    Comprehensive memory cleanup to prevent epoch-to-epoch accumulation.
    
    Returns:
        None
    """
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

# =============================================================================
# Extra losses calculation
# =============================================================================

def dice_loss_for_masks(pred_masks, target_masks, smooth=1e-6):
    """
    Compute dice loss for mask segmentation in Mask R-CNN.
    
    Args:
        pred_masks (torch.Tensor): Predicted masks from Mask R-CNN
                                  Shape: (N, C, H, W) where N is number of 
                                  detections
        target_masks (torch.Tensor): Ground truth masks 
                                    Shape: (N, H, W) or (N, C, H, W)
        smooth (float): Smoothing factor to avoid division by zero
        
    Returns:
        torch.Tensor: Dice loss value
    """
    if pred_masks.numel() == 0 or target_masks.numel() == 0:
        return torch.tensor(0.0, device=pred_masks.device, 
                          requires_grad=True)
    
    # Ensure pred_masks are in the right format and apply sigmoid
    if pred_masks.dim() == 4 and pred_masks.size(1) == 1:
        pred_masks = pred_masks.squeeze(1)  # Remove channel dimension if 1
    pred_masks = torch.sigmoid(pred_masks)
    
    # Ensure target_masks have the same shape as pred_masks
    if target_masks.dim() == 4 and target_masks.size(1) == 1:
        target_masks = target_masks.squeeze(1)
    
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
    Particularly useful for small objects where easy/hard example 
    imbalance is severe.
    
    Args:
        pred_masks (torch.Tensor): Predicted masks
        target_masks (torch.Tensor): Ground truth masks  
        alpha (float): Weighting factor for rare class
        gamma (float): Focusing parameter for hard examples
        smooth (float): Smoothing factor
        
    Returns:
        torch.Tensor: Focal dice loss value
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
    
    Args:
        mask1: First binary mask tensor
        mask2: Second binary mask tensor  
        threshold: Threshold for binarizing masks
        
    Returns:
        IoU score between the two masks
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
    
    Args:
        pred_masks: Predicted masks tensor [N, H, W]
        target_masks: Ground truth masks tensor [M, H, W]
        pred_boxes: Predicted bounding boxes [N, 4]
        target_boxes: Ground truth bounding boxes [M, 4]
        pred_scores: Prediction confidence scores [N]
        iou_threshold: Minimum IoU for valid match
        score_threshold: Minimum confidence score for predictions
        
    Returns:
        List of (pred_idx, target_idx) tuples for valid matches
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
                if ann['area'] < 1000:  # Adjust threshold for your data
                    has_small_objects = True
                    break
            
            if has_small_objects:
                small_object_images.append(image_id)
            else:
                regular_images.append(image_id)
        
        # Oversample small object images (repeat 3 times)
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

def get_model_instance_segmentation(num_classes):
    """
    Creates Mask R-CNN model with optimized configuration for small objects.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Optimize anchor generator for small objects with proper FPN configuration
    anchor_generator = (
        torchvision.models.detection.anchor_utils.AnchorGenerator(
            sizes=((24,), (48,), (96,), (192,), (384,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
    )
    
    model.rpn.anchor_generator = anchor_generator
    
    # Replace predictors
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
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
    
    Args:
        model: Mask R-CNN model
        optimizer: Model optimizer
        data_loader: Training data loader
        device: Training device (CPU/CUDA)
        epoch: Current epoch number
        dice_weight: Weight for dice loss component
        use_focal_dice: Whether to use focal dice loss variant
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    total_standard_loss = 0
    total_dice_loss = 0
    num_batches = 0
    num_matched_pairs = 0  # Track successful matches

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} 
                  for t in targets]

        optimizer.zero_grad(set_to_none=True)

        # Forward pass - get both losses and predictions
        loss_dict = model(images, targets)
        
        # Calculate standard Mask R-CNN losses with optimized weights
        standard_losses = (
            loss_dict['loss_classifier'] * 1.0 +      # Classification loss
            loss_dict['loss_box_reg'] * 1.0 +         # Box regression loss  
            loss_dict['loss_mask'] * 1.5 +            # Standard mask loss
            loss_dict['loss_objectness'] * 1.0 +      # RPN objectness loss
            loss_dict['loss_rpn_box_reg'] * 1.0       # RPN box regression
        )
        
        # Calculate dice loss for masks with IoU matching
        dice_loss_value = torch.tensor(0.0, device=device, 
                                     requires_grad=True)
        
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
                    score_threshold=0.1  # Minimum confidence for predictions
                )
                
                # Compute dice loss only for matched pairs
                if matches:
                    matched_dice_losses = []
                    
                    for pred_idx, target_idx in matches:
                        pred_mask = pred_masks[pred_idx:pred_idx+1]  # Keep batch dim
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
        del images, targets, loss_dict, predictions
        del standard_losses, dice_loss_value, total_batch_loss

        if batch_idx % 4 == 0:
            torch.cuda.empty_cache()

        # Print detailed loss information for first epoch
        if epoch == 1 and batch_idx < 5:
            print(f"Batch {batch_idx}: Total={total_loss/num_batches:.4f}, "
                  f"Standard={total_standard_loss/num_batches:.4f}, "
                  f"Dice={total_dice_loss/num_batches:.4f}, "
                  f"Matches={batch_matches}")

    avg_total_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_dice_loss = total_dice_loss / num_batches if num_batches > 0 else 0.0
    avg_matches_per_batch = num_matched_pairs / num_batches if num_batches > 0 else 0.0
    
    print(f"Epoch {epoch} - Total Loss: {avg_total_loss:.4f}, "
          f"Dice Loss Component: {avg_dice_loss:.4f}, "
          f"Avg Matches/Batch: {avg_matches_per_batch:.1f}")
    
    return avg_total_loss

# =============================================================================
# TEST TIME AUGMENTATION FOR INFERENCE
# =============================================================================

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
            transformed = get_val_transforms()(image=scaled_image)
            tensor_image = transformed['image'].unsqueeze(0).to(device)
            
            # Predict
            pred = model(tensor_image)[0]
            
            # Scale back predictions
            if len(pred['boxes']) > 0:
                pred['boxes'] /= scale
                pred['masks'] = F.interpolate(
                    pred['masks'], size=(h, w), mode='bilinear'
                )
            
            predictions.append(pred)
    
    # Simple merge - use the prediction from scale=1.0
    return predictions[1] if len(predictions) > 1 else predictions[0]

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_model(model, data_loader, device):
    """
    Evaluates the model and calculates metrics.
    """
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]

            predictions = model(images)

            for pred, target in zip(predictions, targets):
                if len(pred['masks']) > 0 and len(target['masks']) > 0:
                    pred_masks = pred['masks'].cpu()
                    true_masks = target['masks'].cpu()

                    image_dice_scores = []
                    for i in range(min(len(pred_masks), len(true_masks))):
                        dice = dice_coefficient(pred_masks[i], true_masks[i])
                        image_dice_scores.append(dice)

                    if image_dice_scores:
                        dice_scores.append(np.mean(image_dice_scores))

    return np.mean(dice_scores) if dice_scores else 0.0

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
            losses = sum(loss for loss in loss_dict.values())
            
            total_loss += losses.item()
            num_batches += 1
    
    model.eval()
    return total_loss / num_batches if num_batches > 0 else 0.0

# =============================================================================
# MAIN TRAINING FUNCTION WITH ENHANCED LEARNING RATE SCHEDULING
# =============================================================================

def main():
    """
    Main function with enhanced training pipeline for small objects.
    """
    print(f"Using device: {DEVICE}")

    # Create enhanced datasets
    print("Loading datasets with small object optimizations...")
    train_dataset = COCOInstanceDataset(
        TRAIN_IMAGES_PATH,
        TRAIN_ANNOTATIONS,
        transforms=get_train_transforms(),
        oversample_small_objects=True,
        use_copy_paste=True
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
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(DEVICE)

    # Enhanced optimizer and scheduler setup
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, 
                           weight_decay=0.0001)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                              milestones=[100, 150], 
                                              gamma=0.1)
    
    # Warmup scheduler for first few epochs
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, 
                                                  start_factor=0.1, 
                                                  total_iters=10)

    # Training metrics
    train_losses = []
    val_losses = []
    val_dice_scores = []

    print("Starting enhanced training for small objects...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        print(f"Memory before epoch: "
            f"{torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, "
            f"{torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")

        # Training
        train_loss = train_one_epoch(
            model, 
            optimizer, 
            train_loader, 
            DEVICE, 
            epoch+1,
            dice_weight=2.0,
            use_focal_dice=USE_FOCAL_DICE
        )
        train_losses.append(train_loss)

        del train_loss
        clear_gpu_memory()

        # Update learning rate
        if epoch < 10:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # Validation
        val_loss = calculate_validation_loss(model, val_loader, DEVICE)
        val_losses.append(val_loss)

        del val_loss
        clear_gpu_memory()

        # Evaluation
        val_dice = evaluate_model(model, val_loader, DEVICE)
        val_dice_scores.append(val_dice)

        del val_dice
        clear_gpu_memory()

        print(f"Training loss: {train_losses[-1]:.4f}")
        print(f"Validation loss: {val_losses[-1]:.4f}")
        print(f"Validation Dice: {val_dice_scores[-1]:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        print(f"Memory after epoch: "
            f"{torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, "
            f"{torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")

        # Save best model
        if (not val_dice_scores[:-1] or 
            val_dice_scores[-1] > max(val_dice_scores[:-1])):
            torch.save(model.state_dict(), 'best_mask_rcnn_model.pth')
            print("Model saved (best Dice score)")
            clear_gpu_memory()

    # Final evaluation
    print("\n=== FINAL EVALUATION ===")
    model.load_state_dict(torch.load('best_mask_rcnn_model.pth'))
    final_dice = evaluate_model(model, val_loader, DEVICE)
    print(f"Final Dice coefficient on validation: {final_dice:.4f}")

    # Training visualization
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(train_losses, color='blue')
    plt.title('Training Loss (Detailed)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(val_dice_scores, color='green')
    plt.title('Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('enhanced_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nEnhanced training completed. Best Dice Score: "
        f"{max(val_dice_scores) if val_dice_scores else 0.0:.4f}")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")

# =============================================================================
# EXECUTE SCRIPT
# =============================================================================

if __name__ == "__main__":
    main()
