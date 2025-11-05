#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask R-CNN Model Loading Utilities

Shared utilities for loading trained Mask R-CNN models with custom
high-resolution mask predictor. Ensures consistency across all inference scripts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class HighResMaskRCNNPredictor(nn.Module):
    """
    Custom high-resolution mask predictor.
    Supports mask_size: 28, 56, or 112.
    
    This class MUST match the architecture used during training in train.py.
    """
    
    def __init__(self, in_channels, dim_reduced, num_classes, mask_size):
        """
        Initialize high-resolution mask predictor.
        
        Args:
            in_channels: Number of input channels from ROI features
            dim_reduced: Reduced dimension (typically 256)
            num_classes: Number of classes including background
            mask_size: Target mask resolution (28, 56, or 112)
        """
        super().__init__()
        self.mask_size = mask_size
        
        # CRITICAL: Reduce hidden dimension for high-resolution masks
        # This MUST match train.py logic
        if mask_size <= 56:
            hidden_dim = dim_reduced
        else:
            hidden_dim = dim_reduced // 2  # 256 -> 128 for memory efficiency
        
        # Build architecture based on mask_size
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
            raise ValueError(
                f"Mask resolution {mask_size} not supported. Use 28, 56, or 112."
            )
        
        # Initialize weights
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif "bias" in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        """Forward pass through the mask predictor."""
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
            x = self.relu2(self.conv6_mask(x))
            x = self.relu3(self.conv7_mask(x))
            x = self.mask_fcn_logits(x)
        
        # Crop to exact size if needed
        if x.shape[-1] != self.mask_size:
            x = F.interpolate(
                x, 
                size=(self.mask_size, self.mask_size),
                mode='bilinear', 
                align_corners=False
            )
        return x


def load_maskrcnn_model(model_path, num_classes, mask_resolution=112, device='cuda'):
    """
    Load trained Mask R-CNN model with custom high-resolution predictor.
    
    This function ensures the loaded model architecture EXACTLY matches
    the architecture used during training.
    
    Args:
        model_path: Path to trained model weights (.pth file)
        num_classes: Number of classes INCLUDING background
        mask_resolution: Mask resolution used during training (28, 56, or 112)
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        Loaded Mask R-CNN model in eval mode
        
    Example:
        >>> model = load_maskrcnn_model(
        ...     'weights/model.pth',
        ...     num_classes=3,
        ...     mask_resolution=112,
        ...     device='cuda'
        ... )
    """
    print(f"Loading Mask R-CNN model from: {model_path}")
    
    # Load base Mask R-CNN model (weights=None to match training)
    model = maskrcnn_resnet50_fpn(weights=None)
    
    # CRITICAL: Set ROI pool size to match training configuration
    if mask_resolution <= 56:
        roi_pool_size = mask_resolution // 2
    else:
        # Cap ROI pooling size at 28 for higher resolutions to save memory
        roi_pool_size = 28
    
    model.roi_heads.mask_roi_pool.output_size = (roi_pool_size, roi_pool_size)
    print(f"  - Mask ROI pool size: {roi_pool_size}×{roi_pool_size}")
    
    # Replace box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor with custom high-res version
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = HighResMaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes, mask_size=mask_resolution
    )
    
    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    print(f"✓ Mask R-CNN model loaded successfully")
    print(f"  - Mask resolution: {mask_resolution}×{mask_resolution}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Device: {device}")
    
    return model


def get_anchor_sizes(img_size, base_min_anchor=16):
    """
    Helper function to calculate anchor sizes for different image resolutions.
    Used during training to match the anchor configuration.
    
    Args:
        img_size: Input image size (typically 2048)
        base_min_anchor: Minimum anchor size for the finest feature map
        
    Returns:
        List of anchor sizes for each feature pyramid level
    """
    scales = [2**i for i in range(5)]  # [1, 2, 4, 8, 16] for 5 FPN levels
    return [base_min_anchor * scale * (img_size / 2048) for scale in scales]
