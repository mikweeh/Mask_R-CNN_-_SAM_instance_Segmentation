#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared visualization utilities for inference scripts.
Provides consistent visualization across Mask R-CNN and hybrid inference.
"""

import cv2
import numpy as np


# Define class colors (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 255, 0),    # Green for Chromis chromis
    1: (255, 0, 0),    # Blue for Coris julis
    2: (0, 0, 255)     # Red (if you add more classes)
}


def draw_polygon_outline(image, mask, class_id, class_names, 
                         min_mask_area=10, line_thickness=1, 
                         draw_label=False):
    """
    Draw polygon outline on image for a single detection.
    
    Args:
        image: Image to draw on (BGR format)
        mask: Binary mask (2D numpy array)
        class_id: Class ID
        class_names: Dict mapping class_id to class_name
        min_mask_area: Minimum contour area to draw
        line_thickness: Line thickness
        draw_label: Whether to draw class label
        
    Returns:
        Modified image
    """
    # Binarize mask
    binary_mask = (mask > 0.5).astype(np.uint8)
    if not np.any(binary_mask):
        return image
    
    # Get color for this class
    color = CLASS_COLORS.get(class_id, (255, 255, 255))
    
    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if not contours:
        return image
    
    # Draw all contours
    for contour in contours:
        if cv2.contourArea(contour) > min_mask_area:
            # Draw polygon outline with anti-aliasing
            cv2.drawContours(
                image, [contour], -1, color, 
                thickness=line_thickness, lineType=cv2.LINE_AA
            )
    
    # Draw label if requested
    if draw_label:
        x, y, w, h = cv2.boundingRect(contours[0])
        label = class_names.get(class_id, f"Class {class_id}")
        
        # Draw label background
        label_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )[0]
        cv2.rectangle(
            image, 
            (x, y - label_size[1] - 5), 
            (x + label_size[0], y),
            color, -1
        )
        
        # Draw label text
        cv2.putText(
            image, label, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    
    return image


def draw_filled_mask(image, mask, class_id, alpha=0.4):
    """
    Draw filled semi-transparent mask on image.
    
    Args:
        image: Image to draw on (BGR format)
        mask: Binary mask (2D numpy array)
        class_id: Class ID
        alpha: Transparency (0=transparent, 1=opaque)
        
    Returns:
        Modified image
    """
    # Binarize mask
    binary_mask = (mask > 0.5).astype(np.uint8)
    if not np.any(binary_mask):
        return image
    
    # Get color for this class
    color = CLASS_COLORS.get(class_id, (255, 255, 255))
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[binary_mask == 1] = color
    
    # Blend with original image
    mask_area = binary_mask == 1
    if np.any(mask_area):
        image[mask_area] = cv2.addWeighted(
            image[mask_area], 1.0 - alpha,
            colored_mask[mask_area], alpha, 0
        )
    
    return image


def create_detection_visualization(image_rgb, detections, class_names, 
                                   output_path, mode='outline', 
                                   line_thickness=1, alpha=0.4,
                                   min_mask_area=10):
    """
    Create visualization of detections (unified for all inference types).
    
    Args:
        image_rgb: Image in RGB format
        detections: List of detection dicts with 'mask' and 'class' keys
        class_names: Dict mapping class_id to class_name
        output_path: Where to save visualization
        mode: 'outline' for polygon outlines, 'filled' for filled masks
        line_thickness: Line thickness for outline mode
        alpha: Transparency for filled mode
        min_mask_area: Minimum mask area to visualize
    """
    # Convert to BGR for OpenCV
    image = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    
    for det in detections:
        mask = det['mask']
        class_id = det['class']
        
        if mode == 'outline':
            image = draw_polygon_outline(
                image, mask, class_id, class_names,
                min_mask_area=min_mask_area,
                line_thickness=line_thickness,
                draw_label=False
            )
        else:  # filled
            image = draw_filled_mask(image, mask, class_id, alpha=alpha)
    
    cv2.imwrite(output_path, image)
