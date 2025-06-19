#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import torch
import torch.nn as nn
import torch.optim as optim
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
# GLOBAL CONFIGURATION VARIABLES
# =============================================================================

# Dataset paths
DATASET_PATH = "dataset"
TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train")
VAL_IMAGES_PATH = os.path.join(DATASET_PATH, "valid")
TRAIN_ANNOTATIONS = os.path.join(TRAIN_IMAGES_PATH, "_annotations_filtered.coco.json")
VAL_ANNOTATIONS = os.path.join(VAL_IMAGES_PATH, "_annotations_filtered.coco.json")

# Training parameters
NUM_CLASSES = 3  # background + Chromis chromis + Coris julis
BATCH_SIZE = 2   # Small due to reduced dataset
NUM_EPOCHS = 200  # More epochs for small dataset
LEARNING_RATE = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
IMG_SIZE = 1024
CONFIDENCE_THRESHOLD = 0.5

# =============================================================================
# AUGMENTATION CONFIGURATION WITH ALBUMENTATIONS
# =============================================================================

def get_train_transforms():
    """
    Defines training augmentations.
    This version preserves aspect ratio.
    """
    return A.Compose([
        # Step 1: Resize the longest side to IMG_SIZE while keeping aspect ratio.
        A.LongestMaxSize(max_size=IMG_SIZE), 
        # Step 2: Pad the shorter side with zeros to make it a square.
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, 
                 value=0),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.Blur(blur_limit=3, p=0.2),
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
        # Step 1: Resize the longest side to IMG_SIZE while keeping aspect ratio.
        A.LongestMaxSize(max_size=IMG_SIZE),
        # Step 2: Pad the shorter side with zeros to make it a square.
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']),
    additional_targets={'masks': 'masks'})

def clear_gpu_memory():
    """
    Comprehensive memory cleanup to prevent epoch-to-epoch accumulation.
    
    Returns:
        None
    """
    # Force garbage collection first
    gc.collect()
    
    # Clear PyTorch cache
    torch.cuda.empty_cache()
    
    # Additional cleanup for fragmentation
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all operations to complete
        torch.cuda.empty_cache()  # Clear cache again after sync

# =============================================================================
# CUSTOM DATASET FOR MASK R-CNN
# =============================================================================

class COCOInstanceDataset(Dataset):
    """
    Custom dataset to load data in COCO format for instance segmentation.
    """

    def __init__(self, images_path, annotations_file, transforms=None):
        self.images_path = images_path
        self.transforms = transforms
        self.coco = COCO(annotations_file)
        self.image_ids = list(self.coco.imgs.keys())

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
            # Convert segmentation to mask
            if 'segmentation' in ann:
                mask = self.coco.annToMask(ann)
                masks.append(mask)

                # Get bounding box
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])

        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.FloatTensor(boxes)
            labels = torch.LongTensor(labels)
            masks = torch.FloatTensor(np.array(masks))
        else:
            # Image without annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, image.shape[0], image.shape[1]), dtype=torch.float32)

        # Apply transformations if they exist
        if self.transforms:
            # Prepare data for albumentations
            bboxes_coco = boxes.clone()
            if len(bboxes_coco) > 0:
                bboxes_coco[:, 2] = bboxes_coco[:, 2] - bboxes_coco[:, 0]  # width
                bboxes_coco[:, 3] = bboxes_coco[:, 3] - bboxes_coco[:, 1]  # height

            transformed = self.transforms(
                image=image,
                bboxes=bboxes_coco.tolist() if len(bboxes_coco) > 0 else [],
                labels=labels.tolist() if len(labels) > 0 else [],
                masks=masks.numpy() if len(masks) > 0 else []
            )

            image = transformed['image']
            if transformed['bboxes']:
                # Convert back to xyxy format
                new_boxes = torch.FloatTensor(transformed['bboxes'])
                new_boxes[:, 2] = new_boxes[:, 0] + new_boxes[:, 2]  # x2 = x1 + width
                new_boxes[:, 3] = new_boxes[:, 1] + new_boxes[:, 3]  # y2 = y1 + height
                boxes = new_boxes
                labels = torch.LongTensor(transformed['labels'])
                masks = torch.FloatTensor(transformed['masks'])

        # Create target dictionary for Mask R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([image_id])
        }

        return image, target

# =============================================================================
# FUNCTION TO CREATE THE MASK R-CNN MODEL
# =============================================================================

def get_model_instance_segmentation(num_classes):
    """
    Creates and configures the pretrained Mask R-CNN model.
    """
    # Load pretrained model on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the box classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

# =============================================================================
# FUNCTION TO CALCULATE DICE COEFFICIENT
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

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        num_batches += 1

        # Delete variables within the loop
        del images, targets, loss_dict, losses

        # Clear cache every few batches to prevent fragmentation
        if batch_idx % 4 == 0:
            torch.cuda.empty_cache()

        if epoch == 1 and batch_idx < 5:  # Debug first epoch
            print(f"Batch {batch_idx}: "
                  f"{torch.cuda.memory_allocated()/1024**3:.2f}GB")

    avg_loss = (total_loss / num_batches) if num_batches > 0 else 0.0
    return avg_loss

# =============================================================================
# EVALUATION FUNCTION
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

            # Prediction
            predictions = model(images)

            # Calculate Dice for each image
            for pred, target in zip(predictions, targets):
                if len(pred['masks']) > 0 and len(target['masks']) > 0:
                    pred_masks = pred['masks'].cpu()
                    true_masks = target['masks'].cpu()

                    # Calculate average Dice for all masks in the image
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
    
    Args:
        model: The trained model
        data_loader: Validation data loader
        device: Device to run calculations on
        
    Returns:
        float: Average validation loss
    """
    model.train()  # Set to train mode to get loss calculations
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():  # No gradient calculation needed
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} 
                      for t in targets]
            
            # Forward pass to get losses
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            total_loss += losses.item()
            num_batches += 1
    
    model.eval()  # Set back to eval mode
    return total_loss / num_batches if num_batches > 0 else 0.0


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    """
    Main function that executes the entire training pipeline.
    """
    print(f"Using device: {DEVICE}")

    # Create datasets
    print("Loading datasets...")
    train_dataset = COCOInstanceDataset(
        TRAIN_IMAGES_PATH,
        TRAIN_ANNOTATIONS,
        transforms=get_train_transforms()
    )

    val_dataset = COCOInstanceDataset(
        VAL_IMAGES_PATH,
        VAL_ANNOTATIONS,
        transforms=get_val_transforms()
    )

    print(f"Training dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")

    # Create data loaders
    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Create model
    print("Initializing Mask R-CNN model...")
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(DEVICE)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Lists to store metrics
    train_losses = []
    val_losses = []
    val_dice_scores = []

    # Training loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # Print memory before epoch starts
        print(f"Memory before epoch: "
            f"{torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, "
            f"{torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")

        # Training
        train_loss = train_one_epoch(model, optimizer, train_loader, 
                                    DEVICE, epoch+1)
        train_losses.append(train_loss)

        # Clear everything before validation
        del train_loss  # Delete the returned value
        clear_gpu_memory()

        # Validation loss calculation
        val_loss = calculate_validation_loss(model, val_loader, DEVICE)
        val_losses.append(val_loss)

        # Clear validation variables
        del val_loss
        clear_gpu_memory()

        # Evaluation (Dice score)
        val_dice = evaluate_model(model, val_loader, DEVICE)
        val_dice_scores.append(val_dice)

        # Clear evaluation variables
        del val_dice
        clear_gpu_memory()

        print(f"Training loss: {train_losses[-1]:.4f}")
        print(f"Validation loss: {val_losses[-1]:.4f}")
        print(f"Validation Dice: {val_dice_scores[-1]:.4f}")
        
        # Print memory after epoch
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
    print(f"Final Dice coefficient on test: {final_dice:.4f}")

    # Show training graphs
    plt.figure(figsize=(15, 5))

    # Plot 1: Training and Validation Losses
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Training Loss only (for detailed view)
    plt.subplot(1, 3, 2)
    plt.plot(train_losses, color='blue')
    plt.title('Training Loss (Detailed)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    # Plot 3: Validation Dice Score
    plt.subplot(1, 3, 3)
    plt.plot(val_dice_scores, color='green')
    plt.title('Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nTraining completed. Best Dice Score: "
        f"{max(val_dice_scores) if val_dice_scores else 0.0:.4f}")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")

# =============================================================================
# EXECUTE SCRIPT
# =============================================================================

if __name__ == "__main__":
    main()
