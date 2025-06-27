#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script creates images with polygon outlines drawn in yellow.
It works with raw masks, YOLO format, and COCO format.
"""

##########################
# Imports and libraries
##########################

import cv2
import numpy as np
import os
import json
from PIL import Image, ImageDraw

#########################
# Global variables
#########################

# Define your directories
# img_directory = 'dataset/original_yolo/train/images'
# msk_directory = 'dataset/original_yolo/train/labels'
# out_directory = 'dataset/original_yolo/train/both'
img_directory = 'dataset/inference/images'
msk_directory = 'dataset/inference/labels_full'
out_directory = 'dataset/inference/both'


# Format selection options
MASK_FORMAT = "YOLO"  # Options: "PNG", "YOLO", "COCO"

# For COCO format, specify the JSON file path
COCO_JSON_PATH = 'annotations_filtered.coco.json'

# NEW: Line drawing parameters
LINE_COLOR = (0, 255, 255)  # Yellow color in BGR format
LINE_THICKNESS = 2           # Line thickness in pixels

# Color mapping for different classes (if you want different colors per class)
CLASS_COLORS = {
    0: (0, 255, 255),    # Yellow for class 0
    1: (255, 0, 255),    # Magenta for class 1  
    2: (255, 255, 0),    # Cyan for class 2
    # Add more colors as needed
}

# Set to True if you want different colors per class, False for all yellow
USE_CLASS_COLORS = False

###########################
# Helper functions for different formats
###########################

def parse_yolo_polygon(txt_path, image_width, image_height):
    """
    Parse YOLO format TXT file and convert to polygon coordinates.
    
    Args:
        txt_path (str): Path to YOLO TXT file
        image_width (int): Width of the image
        image_height (int): Height of the image
    
    Returns:
        list: List of (polygon_points, class_id) tuples
    """
    polygons = []
    
    if not os.path.exists(txt_path):
        return polygons
    
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:  # Need at least class + 3 points (6 coords)
                    continue
                
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                
                # Convert normalized coordinates to pixel coordinates
                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * image_width)
                    y = int(coords[i + 1] * image_height)
                    points.append([x, y])  # Format for cv2.polylines
                
                if len(points) >= 3:  # Need at least 3 points for polygon
                    polygons.append((np.array(points, np.int32), class_id))
    
    except Exception as e:
        print(f"Error parsing YOLO file {txt_path}: {e}")
    
    return polygons

def parse_coco_annotations(coco_json_path, image_filename):
    """
    Parse COCO format JSON file and extract polygons for specific image.
    
    Args:
        coco_json_path (str): Path to COCO JSON file
        image_filename (str): Filename of the image to get annotations for
    
    Returns:
        list: List of (polygon_points, class_id) tuples
    """
    polygons = []
    
    if not os.path.exists(coco_json_path):
        print(f"COCO JSON file not found: {coco_json_path}")
        return polygons
    
    try:
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        # Find image ID for the given filename
        image_id = None
        
        for img_info in coco_data['images']:
            if img_info['file_name'] == image_filename:
                image_id = img_info['id']
                break
        
        if image_id is None:
            print(f"Image {image_filename} not found in COCO annotations")
            return polygons
        
        # Find all annotations for this image
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_id:
                segmentation = annotation['segmentation']
                category_id = annotation['category_id']
                
                # Handle polygon segmentation
                if isinstance(segmentation, list) and len(segmentation) > 0:
                    for polygon in segmentation:
                        if len(polygon) >= 6:  # At least 3 points
                            # Convert to (x, y) pairs
                            points = []
                            for i in range(0, len(polygon), 2):
                                x = int(polygon[i])
                                y = int(polygon[i + 1])
                                points.append([x, y])
                            
                            if len(points) >= 3:
                                polygons.append((np.array(points, np.int32), 
                                               category_id))
    
    except Exception as e:
        print(f"Error parsing COCO file {coco_json_path}: {e}")
    
    return polygons

def extract_contours_from_mask(mask_path):
    """
    Extract contours from PNG mask file.
    
    Args:
        mask_path (str): Path to the mask file
    
    Returns:
        list: List of contours
    """
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    except Exception as e:
        print(f"Error extracting contours from {mask_path}: {e}")
        return []

def draw_polygon_outlines(image, polygons, use_class_colors=False):
    """
    Draw polygon outlines on image.
    
    Args:
        image (numpy.ndarray): Input image
        polygons (list): List of (polygon_points, class_id) tuples
        use_class_colors (bool): Whether to use different colors per class
    
    Returns:
        numpy.ndarray: Image with drawn polygons
    """
    result_image = image.copy()
    
    for polygon_points, class_id in polygons:
        # Choose color
        if use_class_colors:
            color = CLASS_COLORS.get(class_id, LINE_COLOR)
        else:
            color = LINE_COLOR
        
        # Draw polygon outline
        cv2.polylines(result_image, [polygon_points], isClosed=True, 
                     color=color, thickness=LINE_THICKNESS)
    
    return result_image

def draw_contour_outlines(image, contours):
    """
    Draw contour outlines on image.
    
    Args:
        image (numpy.ndarray): Input image
        contours (list): List of contours
    
    Returns:
        numpy.ndarray: Image with drawn contours
    """
    result_image = image.copy()
    
    # Draw all contours
    cv2.drawContours(result_image, contours, -1, LINE_COLOR, LINE_THICKNESS)
    
    return result_image

def draw_outlines_png(image_path, mask_path, output_path):
    """
    Draw outlines from PNG masks.
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not open image: {image_path}")

        # Extract contours from mask
        contours = extract_contours_from_mask(mask_path)
        
        if not contours:
            print(f"No contours found in {mask_path}")
            return
        
        # Draw contour outlines
        result_image = draw_contour_outlines(image, contours)
        
        # Save the result
        cv2.imwrite(output_path, result_image)
        
        print(f"Successfully drew PNG outlines ({len(contours)} contours). "
              f"Output: {output_path}")

    except Exception as e:
        print(f"Error with PNG outline drawing: {e}")

def draw_outlines_yolo(image_path, txt_path, output_path):
    """
    Draw outlines from YOLO TXT format.
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not open image: {image_path}")

        image_height, image_width = image.shape[:2]
        
        # Parse YOLO annotations
        polygons = parse_yolo_polygon(txt_path, image_width, image_height)
        
        if not polygons:
            print(f"No valid polygons found in {txt_path}")
            return
        
        # Draw polygon outlines
        result_image = draw_polygon_outlines(image, polygons, USE_CLASS_COLORS)
        
        # Save the result
        cv2.imwrite(output_path, result_image)
        
        print(f"Successfully drew YOLO outlines ({len(polygons)} objects). "
              f"Output: {output_path}")

    except Exception as e:
        print(f"Error with YOLO outline drawing: {e}")

def draw_outlines_coco(image_path, coco_json_path, output_path):
    """
    Draw outlines from COCO JSON format.
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not open image: {image_path}")

        # Get image filename for COCO lookup
        image_filename = os.path.basename(image_path)
        
        # Parse COCO annotations
        polygons = parse_coco_annotations(coco_json_path, image_filename)
        
        if not polygons:
            print(f"No polygons found for {image_filename} in COCO data")
            return
        
        # Draw polygon outlines
        result_image = draw_polygon_outlines(image, polygons, USE_CLASS_COLORS)
        
        # Save the result
        cv2.imwrite(output_path, result_image)
        
        print(f"Successfully drew COCO outlines ({len(polygons)} objects). "
              f"Output: {output_path}")

    except Exception as e:
        print(f"Error with COCO outline drawing: {e}")

def process_images(image_dir, mask_dir, output_dir, mask_format="PNG", 
                  coco_json_path=None):
    """
    Enhanced function to process images with polygon outline drawing.
    
    Args:
        image_dir (str): Directory containing the images
        mask_dir (str): Directory containing the masks (for PNG/YOLO formats)
        output_dir (str): Directory to save the images with outlines
        mask_format (str): Format of masks ("PNG", "YOLO", "COCO")
        coco_json_path (str): Path to COCO JSON file (for COCO format)
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing images with {mask_format} format masks...")
    print(f"Line color: {LINE_COLOR} (BGR)")
    print(f"Line thickness: {LINE_THICKNESS}")
    print(f"Use class colors: {USE_CLASS_COLORS}")
    
    processed_count = 0
    
    # Loop through all files in the image directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_name = os.path.splitext(filename)[0]
            image_path = os.path.join(image_dir, filename)
            
            # Create output filename
            output_filename = (f"{image_name}_outline_"
                             f"{mask_format.lower()}{os.path.splitext(filename)[1]}")
            output_path = os.path.join(output_dir, output_filename)
            
            if mask_format == "PNG":
                mask_path = os.path.join(mask_dir, image_name + ".png")
                if os.path.exists(mask_path):
                    draw_outlines_png(image_path, mask_path, output_path)
                    processed_count += 1
                else:
                    print(f"PNG mask not found for image: {filename}")
            
            elif mask_format == "YOLO":
                txt_path = os.path.join(mask_dir, image_name + ".txt")
                if os.path.exists(txt_path):
                    draw_outlines_yolo(image_path, txt_path, output_path)
                    processed_count += 1
                else:
                    print(f"YOLO TXT file not found for image: {filename}")
            
            elif mask_format == "COCO":
                if coco_json_path and os.path.exists(coco_json_path):
                    draw_outlines_coco(image_path, coco_json_path, output_path)
                    processed_count += 1
                else:
                    print(f"COCO JSON file not found: {coco_json_path}")
                    break  # No point continuing if JSON is missing
    
    print(f"Processed {processed_count} images")

#################
# Main
#################

def main():
    """Main function with polygon outline drawing."""
    print("="*60)
    print("POLYGON OUTLINE DRAWING SCRIPT")
    print("="*60)
    print(f"Selected format: {MASK_FORMAT}")
    print(f"Image directory: {img_directory}")
    print(f"Mask directory: {msk_directory}")
    print(f"Output directory: {out_directory}")
    print(f"Line color (BGR): {LINE_COLOR}")
    print(f"Line thickness: {LINE_THICKNESS}")
    
    if MASK_FORMAT == "COCO":
        print(f"COCO JSON path: {COCO_JSON_PATH}")
    
    print("="*60)
    
    # Process the images based on selected format
    if MASK_FORMAT == "PNG":
        process_images(img_directory, msk_directory, out_directory, "PNG")
    elif MASK_FORMAT == "YOLO":
        process_images(img_directory, msk_directory, out_directory, "YOLO")
    elif MASK_FORMAT == "COCO":
        process_images(img_directory, msk_directory, out_directory, "COCO", 
                      COCO_JSON_PATH)
    else:
        print(f"Error: Unsupported mask format '{MASK_FORMAT}'")
        print("Supported formats: PNG, YOLO, COCO")
        return
    
    print("="*60)
    print("Processing completed!")
    print("="*60)

if __name__ == "__main__":
    main()
