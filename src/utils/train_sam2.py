#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2 fine-tuning script using GROUND TRUTH MASKS as prompts.
Compatible with SAM2 installed via: pip install git+https://github.com/facebookresearch/sam2.git
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

MODEL_PATH = "weights/sam2_model.pth"
SAM2_MODEL_ID = "facebook/sam2-hiera-large"

DATASET_PATH = "dataset"
TRAIN_IMAGES_PATH = os.path.join(DATASET_PATH, "train")
VAL_IMAGES_PATH = os.path.join(DATASET_PATH, "valid")
TRAIN_ANNOTATIONS = os.path.join(TRAIN_IMAGES_PATH,
                                  "_annotations_filtered.coco.json")
VAL_ANNOTATIONS = os.path.join(VAL_IMAGES_PATH,
                                "_annotations_filtered.coco.json")

NUM_CLASSES = 1
BATCH_SIZE = 1
NUM_EPOCHS = 50
LEARNING_RATE = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_SIZE = 1024
MASK_THRESHOLD = 0.5

REPORT_OUTPUT_PATH = "results"
TEMP_FIGURES_PATH = os.path.join(REPORT_OUTPUT_PATH, "imgs")

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
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


# =============================================================================
# SAM2 Dataset Class
# =============================================================================

class SAM2Dataset(Dataset):
    """
    Dataset class for SAM2 training with COCO annotations.
    Returns ONLY ground truth masks (no boxes).
    """
    
    def __init__(self, images_path, annotations_path, img_size=1024):
        self.images_path = images_path
        self.img_size = img_size
        
        # Load COCO annotations
        self.coco = COCO(annotations_path)
        self.image_ids = list(self.coco.imgs.keys())
        
        # Filter out images without annotations
        self.image_ids = [
            img_id for img_id in self.image_ids
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]
        
        print(f"Loaded {len(self.image_ids)} images with annotations")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Returns numpy arrays that DataLoader will convert to tensors.
        
        Returns:
            image: RGB image array (H, W, 3) - numpy uint8
            masks: Binary masks array (N, H, W) - numpy float32
        """
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_path, img_info['file_name'])
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract masks only
        masks = []
        
        for ann in anns:
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    mask = self.coco.annToMask(ann)
                else:
                    mask = coco_mask.decode(ann['segmentation'])
                
                masks.append(mask.astype(np.float32))
        
        masks = np.array(masks, dtype=np.float32)
        
        return image, masks


# =============================================================================
# SAM2 Model Functions
# =============================================================================

def load_sam2_predictor(model_id, device):
    """Load SAM2ImagePredictor from Hugging Face."""
    if not SAM2_AVAILABLE:
        raise ImportError("SAM2 is not installed.")
    
    print(f"Loading SAM2 from Hugging Face: {model_id}")
    
    try:
        predictor = SAM2ImagePredictor.from_pretrained(model_id)
        predictor.model.to(device)
        predictor.model.train()
        
        # Freeze image encoder
        for param in predictor.model.image_encoder.parameters():
            param.requires_grad = False
        
        print("SAM2ImagePredictor loaded successfully")
        print("Image encoder frozen, mask decoder trainable")
        
        return predictor
        
    except Exception as e:
        print(f"Error loading SAM2: {e}")
        raise


# =============================================================================
# Training Functions - CORRECTED TO USE MASKS AS PROMPTS
# =============================================================================

def train_one_epoch(predictor, optimizer, dataloader, device, epoch):
    """
    Train SAM2 for one epoch - FOLLOWING OFFICIAL EXAMPLES.
    Uses predictor.set_image() which handles all preprocessing internally.
    """
    predictor.model.train()
    
    # Enable training of mask decoder and prompt encoder
    predictor.model.sam_mask_decoder.train(True)
    predictor.model.sam_prompt_encoder.train(True)
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        try:
            batch_loss = 0.0
            
            for img_tensor, mask_gt_tensor in zip(images, masks):
                # Skip if no masks
                if mask_gt_tensor.shape[0] == 0:
                    continue
                
                # img_tensor: (H, W, 3) - uint8 tensor from dataloader
                # mask_gt_tensor: (N, H, W) - float tensor
                
                # Convert to numpy for predictor.set_image
                # (This is the standard SAM2 API)
                img_np = img_tensor.cpu().numpy()
                
                # Move masks to device
                mask_gt_tensor = mask_gt_tensor.to(device)
                
                # CRITICAL: Use predictor.set_image() which handles ALL 
                # preprocessing internally
                predictor.set_image(img_np)
                
                # Get the preprocessed features from predictor
                # predictor._features contains properly formatted features
                image_embeddings = predictor._features["image_embed"]
                
                # Process each instance mask
                num_instances = mask_gt_tensor.shape[0]
                H, W = img_np.shape[:2]
                
                for i in range(num_instances):
                    gt_mask_single = mask_gt_tensor[i]  # Shape: (H, W)
                    
                    # Prepare ground truth mask (4D tensor)
                    gt_mask_4d = gt_mask_single.unsqueeze(0).unsqueeze(0)
                    
                    # Resize to prompt encoder's expected mask size
                    mask_input_size = (
                        predictor.model.sam_prompt_encoder.mask_input_size
                    )
                    
                    gt_mask_prompt = F.interpolate(
                        gt_mask_4d,
                        size=mask_input_size,
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    # Scale to logit range and detach
                    mask_prompt = (gt_mask_prompt - 0.5) * 20
                    mask_prompt = mask_prompt.detach()
                    
                    try:
                        # Prepare dummy point prompts
                        point_coords = torch.zeros(1, 1, 2, device=device)
                        point_labels = -torch.ones(
                            1, 1, dtype=torch.int32, device=device
                        )
                        
                        # Forward through prompt encoder (WITH gradients)
                        sparse_embeddings, dense_embeddings = (
                            predictor.model.sam_prompt_encoder(
                                points=(point_coords, point_labels),
                                boxes=None,
                                masks=mask_prompt,
                            )
                        )
                        
                        # Get high-res features if available
                        high_res_features = None
                        if "high_res_feats" in predictor._features:
                            high_res_features = [
                                feat_level[-1].unsqueeze(0) 
                                for feat_level in 
                                predictor._features["high_res_feats"]
                            ]
                        
                        # Forward through mask decoder (WITH gradients)
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
                        
                        # Resize to original image size
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
                        
                        # Combined loss
                        loss = bce_loss + dice_loss
                        batch_loss += loss
                        
                    except Exception as pred_error:
                        print(f"Prediction error for instance {i}: "
                              f"{pred_error}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            # Backward pass
            if batch_loss > 0 and isinstance(batch_loss, torch.Tensor):
                optimizer.zero_grad()
                batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    predictor.model.parameters(), max_norm=1.0
                )
                
                optimizer.step()
                
                total_loss += batch_loss.item()
                num_batches += 1
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / max(num_batches, 1)
                print(f"  Batch [{batch_idx+1}/{len(dataloader)}] - "
                      f"Avg Loss: {avg_loss:.4f}")
        
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    avg_epoch_loss = total_loss / max(num_batches, 1)
    return avg_epoch_loss


def calculate_validation_loss(predictor, dataloader, device):
    """
    Calculate validation loss using masks as prompts.
    Uses the SAME approach as training: predictor.set_image().
    """
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
                    # Skip if no masks
                    if mask_gt_tensor.shape[0] == 0:
                        continue
                    
                    # Convert to numpy for predictor API
                    img_np = img_tensor.cpu().numpy()
                    
                    # Move masks to device
                    mask_gt_tensor = mask_gt_tensor.to(device)
                    
                    # CRITICAL: Use predictor.set_image() same as training
                    predictor.set_image(img_np)
                    
                    # Get preprocessed features from predictor
                    image_embeddings = predictor._features["image_embed"]
                    
                    num_instances = mask_gt_tensor.shape[0]
                    H, W = img_np.shape[:2]
                    
                    for i in range(num_instances):
                        gt_mask_single = mask_gt_tensor[i]
                        gt_mask_4d = gt_mask_single.unsqueeze(0).unsqueeze(0)
                        
                        # Prepare mask prompt
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
                            # Dummy points
                            point_coords = torch.zeros(1, 1, 2, device=device)
                            point_labels = -torch.ones(
                                1, 1, dtype=torch.int32, device=device
                            )
                            
                            # Get embeddings
                            sparse_embeddings, dense_embeddings = (
                                predictor.model.sam_prompt_encoder(
                                    points=(point_coords, point_labels),
                                    boxes=None,
                                    masks=mask_prompt,
                                )
                            )
                            
                            # Get high-res features if available
                            high_res_features = None
                            if "high_res_feats" in predictor._features:
                                high_res_features = [
                                    feat_level[-1].unsqueeze(0) 
                                    for feat_level in 
                                    predictor._features["high_res_feats"]
                                ]
                            
                            # Decode - SAME AS TRAINING
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
                            
                            # Resize
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
                            dice_loss = 1 - (
                                2 * intersection + 1
                            ) / (union + 1)
                            
                            loss = bce_loss + dice_loss
                            batch_loss += loss
                            
                        except Exception as pred_error:
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
# PDF Report Generation (UNCHANGED)
# =============================================================================

def generate_training_report(train_losses, val_losses, training_start,
                             training_end):
    """Generate PDF training report."""
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
    
    info_data = [
        ["Training Start", training_start.strftime("%Y-%m-%d %H:%M:%S")],
        ["Training End", training_end.strftime("%Y-%m-%d %H:%M:%S")],
        ["Duration", str(training_end - training_start)],
        ["Target Class", class_name],
        ["Number of Epochs", str(NUM_EPOCHS)],
        ["Batch Size", str(BATCH_SIZE)],
        ["Learning Rate", str(LEARNING_RATE)],
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
    print("SAM2 TRAINING CONFIGURATION")
    print("="*60)
    print(f"Using device: {DEVICE}")
    print(f"Target class: {CLASS_NAMES.get(TARGET_CLASS_INDEX, 'unknown')}")
    print(f"SAM2 Model ID: {SAM2_MODEL_ID}")
    print(f"Prompt type: Ground truth MASKS")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print("="*60 + "\n")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = SAM2Dataset(TRAIN_IMAGES_PATH, TRAIN_ANNOTATIONS,
                                 IMG_SIZE)
    val_dataset = SAM2Dataset(VAL_IMAGES_PATH, VAL_ANNOTATIONS, IMG_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)
    
    # Load SAM2 predictor
    print("\nLoading SAM2 predictor...")
    predictor = load_sam2_predictor(SAM2_MODEL_ID, DEVICE)
    
    # Setup optimizer
    trainable_params = [
        p for p in predictor.model.parameters() if p.requires_grad
    ]
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE,
                            weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )
    
    # Training loop
    print("\nStarting SAM2 training with mask prompts...")
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(predictor.model.state_dict(), MODEL_PATH)
            print(f"Model saved: {MODEL_PATH}")
        
        if (epoch + 1) % 10 == 0:
            epoch_path = (f"{os.path.splitext(MODEL_PATH)[0]}_"
                         f"epoch{epoch+1:03d}.pth")
            torch.save(predictor.model.state_dict(), epoch_path)
        
        scheduler.step(val_loss)
        clear_gpu_memory()
    
    training_end_time = datetime.now()
    print(f"\nTraining completed!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    # Generate report
    generate_training_report(train_losses, val_losses,
                            training_start_time, training_end_time)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
