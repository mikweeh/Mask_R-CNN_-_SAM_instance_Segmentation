#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 fine-tuning script using GROUND TRUTH MASKS as prompts.
Compatible with SAM2 installed via: pip install git+https://github.com/facebookresearch/sam2.git

IMPROVEMENTS APPLIED:
- Increased epochs to 150
- Lowered learning rate to 5e-6
- Image size kept at 1024 (SAM2 optimal, see note below)
- Improved LR scheduler (factor=0.6, patience=10)
- Unfrozen prompt encoder for small objects
- Added data augmentation with Albumentations
- Added gradient accumulation
- Increased Dice loss weight to 2.0
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
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from PIL import Image
import random

# ADDED: Albumentations for data augmentation
import albumentations as A

# SAM2 imports
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    print("WARNING: SAM2 not available. Install with:")
    print("  pip install git+https://github.com/facebookresearch/sam2.git")
    SAM2_AVAILABLE = False

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

# Model configuration
MODEL_PATH = "weights/sam2_model.pth"
SAM2_MODEL_ID = "facebook/sam2-hiera-large"

# Dataset paths
DATASET_PATH = "dataset"
TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train")
VAL_IMAGES_PATH = os.path.join(DATASET_PATH, "valid")
TRAIN_ANNOTATIONS = os.path.join(TRAIN_IMAGES_PATH,
                                  "_annotations_filtered.coco.json")
VAL_ANNOTATIONS = os.path.join(VAL_IMAGES_PATH,
                                "_annotations_filtered.coco.json")

# Training parameters - IMPROVED FOR SMALL OBJECTS
NUM_CLASSES = 1
BATCH_SIZE = 1
NUM_EPOCHS = 150  # CHANGED: Increased from 50 to 150 for better convergence
LEARNING_RATE = 5e-6  # CHANGED: Lowered from 1e-5 to 5e-6 for fine-grained learning
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ADDED: Gradient accumulation steps for effective larger batch size
GRADIENT_ACCUMULATION_STEPS = 4  # NEW: Effective batch size = 1 * 3 = 3

# Model parameters
# NOTE: Keeping IMG_SIZE=1024. While you have large images, SAM2 was trained on 1024x1024
# and works optimally at this resolution. Increasing to 2048 would:
# - Increase memory usage by 4x (2048²/1024² = 4)
# - Slow training significantly
# - Not necessarily improve accuracy (SAM2's architecture is optimized for 1024)
# - Risk out-of-memory errors
# RECOMMENDATION: Keep at 1024 and rely on other improvements (augmentation, longer training)
IMG_SIZE = 1024  # UNCHANGED: Optimal for SAM2
MASK_THRESHOLD = 0.5
MIN_MASK_AREA = 100

# ADDED: Loss weight for Dice loss (for small objects)
DICE_WEIGHT = 2.0  # NEW: Give more weight to Dice loss for better boundaries

# Max number of instances per image
MAX_INSTANCES_PER_IMAGE = 100

# Report configuration
REPORT_OUTPUT_PATH = "results"
TEMP_FIGURES_PATH = os.path.join(REPORT_OUTPUT_PATH, "imgs")

# Class configuration
CLASS_NAMES = {0: "Chromis chromis", 1: "Coris julis"}
TARGET_CLASS_INDEX = 0


# =============================================================================
# Helper Functions
# =============================================================================

def get_next_model_name_train():
    """Get the next available model name."""
    global MODEL_PATH
    weights_dir = os.path.dirname(MODEL_PATH)
    basename_with_ext = os.path.basename(MODEL_PATH)
    basename, ext = os.path.splitext(basename_with_ext)
    
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        print(f"Created weights directory: {weights_dir}")
    
    base_path = MODEL_PATH
    if not os.path.exists(base_path):
        return base_path
    
    counter = 1
    while True:
        new_path = os.path.join(weights_dir, f"{basename}_{counter}{ext}")
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def clear_gpu_memory():
    """Enhanced GPU memory clearing."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


# =============================================================================
# SAM2 Dataset Class - WITH DATA AUGMENTATION
# =============================================================================

class SAM2Dataset(Dataset):
    """
    Dataset class for SAM2 training with COCO annotations.
    Returns ONLY ground truth masks (no boxes).
    IMPROVED: Added data augmentation for training.
    """
    
    def __init__(self, images_path, annotations_path, img_size=1024, 
                 is_training=True):  # ADDED: is_training parameter
        self.images_path = images_path
        self.img_size = img_size
        self.is_training = is_training  # NEW
        
        # Load COCO annotations
        self.coco = COCO(annotations_path)
        self.image_ids = list(self.coco.imgs.keys())
        
        # Filter out images without annotations
        self.image_ids = [
            img_id for img_id in self.image_ids
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]
        
        # ADDED: Data augmentation pipeline (only for training)
        if self.is_training:
            self.transform = A.Compose([
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,  # Important for scale variations
                    rotate_limit=45,
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5),
            ])
        else:
            self.transform = None
        
        print(f"Loaded {len(self.image_ids)} images with annotations")
        if self.is_training:
            print("Data augmentation ENABLED for training")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Returns numpy arrays that DataLoader will convert to tensors.
        IMPROVED: Proper COCO mask handling with dimension validation.
        
        Returns:
            image: RGB image array (H, W, 3) - numpy uint8
            masks: Binary masks array (N, H, W) - numpy float32
        """
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_path, img_info['file_name'])
        
        # Load image as numpy array
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get EXPECTED dimensions from COCO metadata (most reliable)
        expected_height = img_info['height']
        expected_width = img_info['width']
        
        # Verify loaded image matches COCO metadata
        actual_height, actual_width = image.shape[:2]
        if (actual_height != expected_height or 
            actual_width != expected_width):
            print(f"Warning: Image {img_id} dimension mismatch. "
                f"Expected {expected_height}x{expected_width}, "
                f"got {actual_height}x{actual_width}. Resizing.")
            image = cv2.resize(image, (expected_width, expected_height))
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract masks with proper dimension handling
        masks = []
        
        for ann in anns:
            if 'segmentation' in ann:
                # PROFESSIONAL: Use COCO's built-in dimension info
                if isinstance(ann['segmentation'], list):
                    # Polygon format
                    mask = self.coco.annToMask(ann)
                else:
                    # RLE format - can have dimension issues
                    if isinstance(ann['segmentation'], dict):
                        # Compressed RLE
                        mask = coco_mask.decode(ann['segmentation'])
                    else:
                        # Uncompressed RLE
                        mask = self.coco.annToMask(ann)
                
                # Validate mask dimensions against COCO metadata
                mask_h, mask_w = mask.shape[:2]
                if (mask_h != expected_height or mask_w != expected_width):
                    # This is expected for some COCO annotations
                    # Resize to match COCO metadata dimensions
                    mask = cv2.resize(
                        mask, 
                        (expected_width, expected_height),
                        interpolation=cv2.INTER_NEAREST
                    )
                
                masks.append(mask.astype(np.float32))
        
        # Skip images without valid masks
        if len(masks) == 0:
            masks = np.zeros((1, expected_height, expected_width), 
                            dtype=np.float32)
        else:
            masks = np.array(masks, dtype=np.float32)
        
        # Apply augmentation if training
        if self.is_training and self.transform is not None and masks.shape[0] > 0:
            try:
                # Convert masks to list format for Albumentations
                masks_list = [masks[i] for i in range(masks.shape[0])]
                
                # Apply augmentation
                augmented = self.transform(image=image, masks=masks_list)
                image = augmented['image']
                masks_augmented = augmented['masks']
                
                # Convert back to numpy array
                if len(masks_augmented) > 0:
                    masks = np.array(masks_augmented, dtype=np.float32)
                
            except Exception as e:
                # Fallback to original if augmentation fails
                print(f"Warning: Augmentation failed for image {img_id}: {e}")
                # Use original image and masks
        
        return image, masks


# =============================================================================
# SAM2 Model Functions - IMPROVED
# =============================================================================

def load_sam2_predictor(model_id, device, fine_tuned_weights_path=None):
    """
    Load SAM2ImagePredictor from Hugging Face for fine-tuning.
    IMPROVED: Unfreezes prompt encoder for better small object detection.
    
    Args:
        model_id: Hugging Face model ID
        device: Device to load model on
        fine_tuned_weights_path: Optional path to load fine-tuned weights
    
    Returns:
        SAM2ImagePredictor with model loaded
    """
    if not SAM2_AVAILABLE:
        raise ImportError("SAM2 is not installed.")
    
    print(f"Loading SAM2 from Hugging Face: {model_id}")
    print("This will automatically download the model on first use...")
    
    try:
        predictor = SAM2ImagePredictor.from_pretrained(model_id)
        sam2_model = predictor.model
        sam2_model = sam2_model.to(device)
        
        # Load fine-tuned weights if provided
        if fine_tuned_weights_path and os.path.exists(
            fine_tuned_weights_path
        ):
            print(f"Loading fine-tuned weights from {fine_tuned_weights_path}")
            state_dict = torch.load(fine_tuned_weights_path,
                                    map_location=device)
            sam2_model.load_state_dict(state_dict)
            print("Fine-tuned weights loaded successfully")
        
        sam2_model.train()
        
        # Freeze image encoder (keep frozen)
        for param in sam2_model.image_encoder.parameters():
            param.requires_grad = False
        
        # CHANGED: Unfreeze mask decoder (was already unfrozen)
        for param in sam2_model.sam_mask_decoder.parameters():
            param.requires_grad = True
        
        # ADDED: Unfreeze prompt encoder for better small object detection
        for param in sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = True
        
        print("SAM2 model loaded successfully from Hugging Face")
        print("Image encoder: FROZEN")
        print("Mask decoder: TRAINABLE")
        print("Prompt encoder: TRAINABLE (NEW - for small objects)")
        
        return predictor
        
    except Exception as e:
        print(f"Error loading SAM2 model: {e}")
        raise


# =============================================================================
# Training Functions - WITH GRADIENT ACCUMULATION
# =============================================================================

def train_one_epoch(predictor, optimizer, dataloader, device, epoch):
    """
    Train SAM2 for one epoch using ground truth MASKS as prompts.
    IMPROVED: Added gradient accumulation for effective larger batch size.
    """
    predictor.model.train()
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)
    
    total_loss = 0.0
    num_batches = 0
    
    # ADDED: For gradient accumulation
    optimizer.zero_grad()
    accumulation_counter = 0
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        try:
            batch_loss = 0.0
            
            for img_tensor, mask_gt_tensor in zip(images, masks):
                if mask_gt_tensor.shape[0] == 0:
                    continue
                
                img_np = img_tensor.cpu().numpy()
                mask_gt_tensor = mask_gt_tensor.to(device)
                
                predictor.set_image(img_np)
                image_embeddings = predictor._features["image_embed"]
                
                num_instances = mask_gt_tensor.shape[0]
                H, W = img_np.shape[:2]
                
                for i in range(min(num_instances, MAX_INSTANCES_PER_IMAGE)):
                    gt_mask_single = mask_gt_tensor[i]
                    gt_mask_4d = gt_mask_single.unsqueeze(0).unsqueeze(0)
                    
                    mask_input_size = (
                        predictor.model.sam_prompt_encoder.mask_input_size
                    )
                    
                    gt_mask_prompt = F.interpolate(
                        gt_mask_4d,
                        size=mask_input_size,
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    mask_prompt = (gt_mask_prompt - 0.5) * 20
                    mask_prompt = mask_prompt.detach()
                    
                    try:
                        point_coords = torch.zeros(1, 1, 2, device=device)
                        point_labels = -torch.ones(
                            1, 1, dtype=torch.int32, device=device
                        )
                        
                        sparse_embeddings, dense_embeddings = (
                            predictor.model.sam_prompt_encoder(
                                points=(point_coords, point_labels),
                                boxes=None,
                                masks=mask_prompt,
                            )
                        )
                        
                        high_res_features = None
                        if "high_res_feats" in predictor._features:
                            high_res_features = [
                                feat_level[-1].unsqueeze(0) 
                                for feat_level in 
                                predictor._features["high_res_feats"]
                            ]
                        
                        low_res_masks, iou_predictions, _, _ = (
                            predictor.model.sam_mask_decoder(
                                image_embeddings=image_embeddings,
                                image_pe=(
                                    predictor.model.sam_prompt_encoder.get_dense_pe()
                                ),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=False,
                                repeat_image=False,
                                high_res_features=high_res_features,
                            )
                        )
                        
                        pred_mask_resized = F.interpolate(
                            low_res_masks,
                            size=(H, W),
                            mode='bilinear',
                            align_corners=False
                        )
                        
                        # Compute loss
                        bce_loss = F.binary_cross_entropy_with_logits(
                            pred_mask_resized, gt_mask_4d
                        )
                        
                        pred_sigmoid = torch.sigmoid(pred_mask_resized)
                        intersection = (pred_sigmoid * gt_mask_4d).sum()
                        union = pred_sigmoid.sum() + gt_mask_4d.sum()
                        dice_loss = 1 - (2 * intersection + 1) / (union + 1)
                        
                        # CHANGED: Weighted loss with increased Dice weight
                        loss = bce_loss + DICE_WEIGHT * dice_loss
                        
                        # CHANGED: Scale loss for gradient accumulation
                        loss = loss / GRADIENT_ACCUMULATION_STEPS
                        batch_loss += loss
                        
                    except Exception as pred_error:
                        print(f"Prediction error for instance {i}: {pred_error}")
                        continue
            
            # CHANGED: Gradient accumulation logic
            if batch_loss > 0 and isinstance(batch_loss, torch.Tensor):
                batch_loss.backward()
                accumulation_counter += 1
                
                # Update weights every GRADIENT_ACCUMULATION_STEPS
                if accumulation_counter % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(
                        predictor.model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += batch_loss.item() * GRADIENT_ACCUMULATION_STEPS
                num_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / max(num_batches, 1)
                print(f"  Batch [{batch_idx+1}/{len(dataloader)}] - "
                      f"Avg Loss: {avg_loss:.4f}")
                clear_gpu_memory()
        
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    # ADDED: Final update if there are remaining gradients
    if accumulation_counter % GRADIENT_ACCUMULATION_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(
            predictor.model.parameters(), max_norm=1.0
        )
        optimizer.step()
        optimizer.zero_grad()
    
    avg_epoch_loss = total_loss / max(num_batches, 1)
    return avg_epoch_loss


def calculate_validation_loss(predictor, dataloader, device):
    """Calculate validation loss using masks as prompts."""
    predictor.model.eval()
    predictor.model.sam_mask_decoder.eval()
    predictor.model.sam_prompt_encoder.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            try:
                batch_loss = 0.0
                
                for img_tensor, mask_gt_tensor in zip(images, masks):
                    if mask_gt_tensor.shape[0] == 0:
                        continue
                    
                    img_np = img_tensor.cpu().numpy()
                    mask_gt_tensor = mask_gt_tensor.to(device)
                    
                    predictor.set_image(img_np)
                    image_embeddings = predictor._features["image_embed"]
                    
                    num_instances = mask_gt_tensor.shape[0]
                    H, W = img_np.shape[:2]
                    
                    for i in range(min(num_instances, MAX_INSTANCES_PER_IMAGE)):
                        gt_mask_single = mask_gt_tensor[i]
                        gt_mask_4d = gt_mask_single.unsqueeze(0).unsqueeze(0)
                        
                        mask_input_size = (
                            predictor.model.sam_prompt_encoder.mask_input_size
                        )
                        gt_mask_prompt = F.interpolate(
                            gt_mask_4d,
                            size=mask_input_size,
                            mode='bilinear',
                            align_corners=False
                        )
                        
                        mask_prompt = (gt_mask_prompt - 0.5) * 20
                        
                        try:
                            point_coords = torch.zeros(1, 1, 2, device=device)
                            point_labels = -torch.ones(
                                1, 1, dtype=torch.int32, device=device
                            )
                            
                            sparse_embeddings, dense_embeddings = (
                                predictor.model.sam_prompt_encoder(
                                    points=(point_coords, point_labels),
                                    boxes=None,
                                    masks=mask_prompt,
                                )
                            )
                            
                            high_res_features = None
                            if "high_res_feats" in predictor._features:
                                high_res_features = [
                                    feat_level[-1].unsqueeze(0) 
                                    for feat_level in 
                                    predictor._features["high_res_feats"]
                                ]
                            
                            low_res_masks, _, _, _ = (
                                predictor.model.sam_mask_decoder(
                                    image_embeddings=image_embeddings,
                                    image_pe=(
                                        predictor.model.sam_prompt_encoder.get_dense_pe()
                                    ),
                                    sparse_prompt_embeddings=sparse_embeddings,
                                    dense_prompt_embeddings=dense_embeddings,
                                    multimask_output=False,
                                    repeat_image=False,
                                    high_res_features=high_res_features,
                                )
                            )
                            
                            pred_mask_resized = F.interpolate(
                                low_res_masks,
                                size=(H, W),
                                mode='bilinear',
                                align_corners=False
                            )
                            
                            bce_loss = F.binary_cross_entropy_with_logits(
                                pred_mask_resized, gt_mask_4d
                            )
                            
                            pred_sigmoid = torch.sigmoid(pred_mask_resized)
                            intersection = (pred_sigmoid * gt_mask_4d).sum()
                            union = pred_sigmoid.sum() + gt_mask_4d.sum()
                            dice_loss = 1 - (
                                2 * intersection + 1
                            ) / (union + 1)
                            
                            # CHANGED: Use weighted loss
                            loss = bce_loss + DICE_WEIGHT * dice_loss
                            batch_loss += loss
                            
                        except Exception:
                            continue
                
                if batch_loss > 0 and isinstance(batch_loss, torch.Tensor):
                    total_loss += batch_loss.item()
                    num_batches += 1
            
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


# =============================================================================
# Test Inference for PDF Report
# =============================================================================

def generate_test_inference_examples(predictor, output_path, num_samples=5):
    """Generate inference examples on test images for the PDF report."""
    predictor.model.eval()
    processed_paths = []
    
    test_images_path = os.path.join(DATASET_PATH, "test")
    
    if not os.path.exists(test_images_path):
        print(f"Test images folder not found: {test_images_path}")
        return processed_paths
    
    test_image_files = [
        f for f in os.listdir(test_images_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    if len(test_image_files) == 0:
        print("No test images found.")
        return processed_paths
    
    sampled_files = random.sample(
        test_image_files, 
        min(num_samples, len(test_image_files))
    )
    
    print(f"\nGenerating {len(sampled_files)} test inference examples...")
    
    for idx, image_file in enumerate(sampled_files):
        try:
            image_path = os.path.join(test_images_path, image_file)
            print(f"  Processing {idx+1}/{len(sampled_files)}: {image_file}")
            
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            predictor.set_image(image_rgb)
            image_embeddings = predictor._features["image_embed"]
            
            H, W = image_rgb.shape[:2]
            num_points = 64  # TODO: This should be passed as a parameter
            y_points = np.linspace(0, H-1, num_points, dtype=np.float32)
            x_points = np.linspace(0, W-1, num_points, dtype=np.float32)
            
            masks = []
            
            for y in y_points:
                for x in x_points:
                    point_coords = torch.tensor(
                        [[[x, y]]], dtype=torch.float32, device=DEVICE
                    )
                    point_labels = torch.ones(
                        (1, 1), dtype=torch.int32, device=DEVICE
                    )
                    
                    try:
                        sparse_embeddings, dense_embeddings = (
                            predictor.model.sam_prompt_encoder(
                                points=(point_coords, point_labels),
                                boxes=None,
                                masks=None,
                            )
                        )
                        
                        high_res_features = None
                        if "high_res_feats" in predictor._features:
                            high_res_features = [
                                feat_level[-1].unsqueeze(0) 
                                for feat_level in 
                                predictor._features["high_res_feats"]
                            ]
                        
                        with torch.no_grad():
                            low_res_masks, _, _, _ = (
                                predictor.model.sam_mask_decoder(
                                    image_embeddings=image_embeddings,
                                    image_pe=(
                                        predictor.model.sam_prompt_encoder.get_dense_pe()
                                    ),
                                    sparse_prompt_embeddings=sparse_embeddings,
                                    dense_prompt_embeddings=dense_embeddings,
                                    multimask_output=False,
                                    repeat_image=False,
                                    high_res_features=high_res_features,
                                )
                            )
                        
                        pred_mask = F.interpolate(
                            low_res_masks,
                            size=(H, W),
                            mode='bilinear',
                            align_corners=False
                        )
                        
                        mask_np = (
                            torch.sigmoid(pred_mask) > 0.5
                        ).cpu().numpy()[0, 0]
                        
                        if mask_np.sum() >= MIN_MASK_AREA:
                            masks.append(mask_np.astype(np.uint8))
                    
                    except Exception:
                        continue
            
            result_image = image.copy()
            class_colors = {0: (0, 255, 0), 1: (255, 0, 0)}
            color = class_colors.get(TARGET_CLASS_INDEX, (255, 255, 255))
            alpha = 0.4
            
            for mask in masks:
                colored_mask = np.zeros_like(result_image)
                colored_mask[mask > 0] = color
                mask_area = mask > 0
                result_image[mask_area] = cv2.addWeighted(
                    result_image[mask_area], 1.0 - alpha,
                    colored_mask[mask_area], alpha, 0
                )
            
            output_filename = f"test_inference_{idx+1}.jpg"
            output_filepath = os.path.join(output_path, output_filename)
            cv2.imwrite(output_filepath, result_image)
            
            processed_paths.append(output_filepath)
            print(f"    Saved: {output_filename} ({len(masks)} masks)")
            
        except Exception as e:
            print(f"    Error processing {image_file}: {e}")
            continue
    
    return processed_paths


# =============================================================================
# PDF Report Generation
# =============================================================================

def generate_training_report(train_losses, val_losses, training_start,
                             training_end, test_image_paths=None):
    """Generate PDF training report with test inference examples."""
    if not REPORTLAB_AVAILABLE:
        print("ReportLab not available, skipping PDF generation")
        return None
    
    os.makedirs(TEMP_FIGURES_PATH, exist_ok=True)
    
    timestamp = training_start.strftime("%Y%m%d_%H%M%S")
    class_name = CLASS_NAMES.get(TARGET_CLASS_INDEX, "unknown")
    report_filename = (f"sam2_training_report_{class_name}_{timestamp}.pdf")
    report_path = os.path.join(REPORT_OUTPUT_PATH, report_filename)
    
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2E4053'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph(f"SAM2 Training Report - {class_name}",
                           title_style))
    story.append(Spacer(1, 20))
    
    # ADDED: Show new configuration parameters
    info_data = [
        ["Training Start", training_start.strftime("%Y-%m-%d %H:%M:%S")],
        ["Training End", training_end.strftime("%Y-%m-%d %H:%M:%S")],
        ["Duration", str(training_end - training_start)],
        ["Target Class", class_name],
        ["Number of Epochs", str(NUM_EPOCHS)],
        ["Batch Size", f"{BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})"],  # CHANGED
        ["Learning Rate", str(LEARNING_RATE)],
        ["Dice Loss Weight", str(DICE_WEIGHT)],  # ADDED
        ["Data Augmentation", "ENABLED"],  # ADDED
        ["Prompt Encoder", "TRAINABLE"],  # ADDED
        ["Device", str(DEVICE)],
    ]
    
    info_table = Table(info_data, colWidths=[2.5*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(info_table)
    story.append(Spacer(1, 30))
    
    # Training metrics plot
    plt.figure(figsize=(12, 5))
    
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
    training_plots_path = os.path.join(TEMP_FIGURES_PATH,
                                        "sam2_training_metrics.png")
    plt.savefig(training_plots_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    story.append(Paragraph("Training Metrics", styles['Heading2']))
    story.append(Spacer(1, 10))
    story.append(RLImage(training_plots_path, width=7*inch, height=2.5*inch))
    
    # Add test inference images
    if test_image_paths and len(test_image_paths) > 0:
        story.append(Spacer(1, 30))
        story.append(Paragraph("Test Inference Examples", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        for idx, img_path in enumerate(test_image_paths):
            if img_path and os.path.exists(img_path):
                story.append(Paragraph(f"Test Image {idx+1}", styles['Heading3']))
                story.append(RLImage(img_path, width=6*inch, height=4*inch))
                story.append(Spacer(1, 20))
    
    doc.build(story)
    print(f"Training report generated: {report_path}")
    
    return report_path


# =============================================================================
# Main Training Function
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train SAM2 model for binary fish segmentation'
    )
    
    parser.add_argument('--dataset_path', type=str, default=DATASET_PATH)
    parser.add_argument('--model_path', type=str,
                        default=get_next_model_name_train())
    parser.add_argument('--sam2_model_id', type=str, default=SAM2_MODEL_ID)
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float,
                        default=LEARNING_RATE)
    parser.add_argument('--img_size', type=int, default=IMG_SIZE)
    parser.add_argument('--report_output_path', type=str,
                        default=REPORT_OUTPUT_PATH)
    parser.add_argument('--class_names', type=str, default=None)
    parser.add_argument('--target_class_index', type=int,
                        default=TARGET_CLASS_INDEX)
    
    return parser.parse_args()


def update_global_variables(args):
    """Update global variables with command-line arguments."""
    global MODEL_PATH, DATASET_PATH, TRAIN_IMAGES_PATH, VAL_IMAGES_PATH, \
           TRAIN_ANNOTATIONS, VAL_ANNOTATIONS, NUM_EPOCHS, BATCH_SIZE, \
           LEARNING_RATE, IMG_SIZE, REPORT_OUTPUT_PATH, TEMP_FIGURES_PATH, \
           CLASS_NAMES, TARGET_CLASS_INDEX, SAM2_MODEL_ID
    
    MODEL_PATH = args.model_path
    DATASET_PATH = args.dataset_path
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    IMG_SIZE = args.img_size
    REPORT_OUTPUT_PATH = args.report_output_path
    SAM2_MODEL_ID = args.sam2_model_id
    TARGET_CLASS_INDEX = args.target_class_index
    
    TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train")
    VAL_IMAGES_PATH = os.path.join(DATASET_PATH, "valid")
    TRAIN_ANNOTATIONS = os.path.join(TRAIN_IMAGES_PATH,
                                      "_annotations_filtered.coco.json")
    VAL_ANNOTATIONS = os.path.join(VAL_IMAGES_PATH,
                                    "_annotations_filtered.coco.json")
    TEMP_FIGURES_PATH = os.path.join(REPORT_OUTPUT_PATH, "imgs")
    
    if args.class_names is not None:
        CLASS_NAMES = json.loads(args.class_names)
        CLASS_NAMES = {int(k): v for k, v in CLASS_NAMES.items()}


def main():
    """Main training function."""
    args = parse_arguments()
    update_global_variables(args)
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(REPORT_OUTPUT_PATH, exist_ok=True)
    os.makedirs(TEMP_FIGURES_PATH, exist_ok=True)
    
    training_start_time = datetime.now()
    print("\n" + "="*60)
    print("SAM2 TRAINING CONFIGURATION - IMPROVED FOR SMALL OBJECTS")
    print("="*60)
    print(f"Using device: {DEVICE}")
    print(f"Target class: {CLASS_NAMES.get(TARGET_CLASS_INDEX, 'unknown')}")
    print(f"SAM2 Model ID: {SAM2_MODEL_ID}")
    print(f"Number of epochs: {NUM_EPOCHS} (INCREASED)")
    print(f"Learning rate: {LEARNING_RATE} (LOWERED)")
    print(f"Dice loss weight: {DICE_WEIGHT} (INCREASED)")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Data augmentation: ENABLED")
    print("="*60 + "\n")
    
    # Load datasets with augmentation
    print("Loading datasets...")
    train_dataset = SAM2Dataset(
        TRAIN_IMAGES_PATH, TRAIN_ANNOTATIONS, IMG_SIZE, is_training=True
    )
    val_dataset = SAM2Dataset(
        VAL_IMAGES_PATH, VAL_ANNOTATIONS, IMG_SIZE, is_training=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)
    
    # Load SAM2 predictor with unfrozen prompt encoder
    print("\nLoading SAM2 predictor...")
    predictor = load_sam2_predictor(SAM2_MODEL_ID, DEVICE)
    
    # Setup optimizer (only trainable parameters)
    trainable_params = [
        p for p in predictor.model.parameters() if p.requires_grad
    ]
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE,
                            weight_decay=0.01)
    
    # CHANGED: Improved learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.6,      # Reduce by 40%
        patience=10,     # Wait longer before reducing
        min_lr=1e-7,
    )
    
    # Training loop
    print("\nStarting SAM2 training with improvements...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch [{epoch+1}/{NUM_EPOCHS}] ---")
        
        train_loss = train_one_epoch(predictor, optimizer, train_loader,
                                      DEVICE, epoch + 1)
        train_losses.append(train_loss)
        
        clear_gpu_memory()
        
        val_loss = calculate_validation_loss(predictor, val_loader, DEVICE)
        val_losses.append(val_loss)
        
        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(predictor.model.state_dict(), MODEL_PATH)
            print(f"Model saved (val_loss improved to {val_loss:.4f}): "
                  f"{MODEL_PATH}")
        
        if (epoch + 1) % 10 == 0:
            epoch_path = (f"{os.path.splitext(MODEL_PATH)[0]}_"
                         f"epoch{epoch+1:03d}.pth")
            torch.save(predictor.model.state_dict(), epoch_path)
            print(f"Checkpoint saved at epoch {epoch+1}: {epoch_path}")
        
        scheduler.step(val_loss)
        clear_gpu_memory()
    
    training_end_time = datetime.now()
    print(f"\nTraining completed!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    # Generate test inference examples
    print("\nGenerating test inference examples for report...")
    test_inference_paths = generate_test_inference_examples(
        predictor, TEMP_FIGURES_PATH, num_samples=5
    )
    
    # Generate PDF report with test images
    print("\nGenerating PDF training report...")
    generate_training_report(
        train_losses, val_losses,
        training_start_time, training_end_time,
        test_inference_paths
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
