#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script combines inference labels (classes 0 and 1) with original YOLO 
labels (other classes), creating complete annotation files that replace the 
original classes 0 and 1 with the new inferred ones while preserving all 
other class annotations.
"""

import os
import argparse
import json
import random
import shutil
from pathlib import Path
from PIL import Image

# =============================================================================
# DEFAULT GLOBAL CONFIGURATION VARIABLES
# =============================================================================

# Default paths
INFERENCE_FOLDER = "dataset/inference/labels"
ORIGINAL_FOLDER = "dataset/original_yolo/test/labels"
ORIGINAL_IMG_FOLDER = "dataset/test"
OUTPUT_FOLDER = "dataset/inference/labels_full"
ORIGINAL_YOLO_DATA_YAML = "dataset/original_yolo/data.yaml"

# Default processing configuration - can be overridden by arguments from main.py
TARGET_CLASSES = {0, 1}  # Default classes to replace with inference results

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Combine inference labels with original YOLO labels"
    )
    
    parser.add_argument(
        '--inference_folder',
        default=INFERENCE_FOLDER,
        help='Path to folder containing inference labels'
    )
    
    parser.add_argument(
        '--original_folder', 
        default=ORIGINAL_FOLDER,
        help='Path to folder containing original YOLO labels (all classes)'
    )
    
    parser.add_argument(
        '--output_folder',
        default=OUTPUT_FOLDER,
        help='Output folder path for combined labels'
    )
    
    # Class configuration argument (optional - will override default if provided)
    parser.add_argument(
        '--target_classes',
        type=str,
        default=None,
        help='JSON string with list of target classes to replace'
    )
    
    # Folder used to upload the results to roboflow (containing original images and infered labels)
    parser.add_argument(
        '--upload_folder',
        type=str,
        default=None,
        help='Optional folder for upload files (copy images, labels, yaml)'
    )

    # Folder with original images
    parser.add_argument(
        '--original_img_folder', 
        default=ORIGINAL_IMG_FOLDER,
        help='Path to folder containing original images'
    )

    return parser.parse_args()

def update_global_variables(args):
    """Update global variables with command-line arguments."""
    global INFERENCE_FOLDER, ORIGINAL_FOLDER, OUTPUT_FOLDER, TARGET_CLASSES, ORIGINAL_IMG_FOLDER
    
    # Update standard global variables
    INFERENCE_FOLDER = args.inference_folder
    ORIGINAL_FOLDER = args.original_folder
    OUTPUT_FOLDER = args.output_folder
    ORIGINAL_IMG_FOLDER = args.original_img_folder
    
    # Update class configuration ONLY if provided as argument
    if args.target_classes is not None:
        target_classes_list = json.loads(args.target_classes)
        TARGET_CLASSES = set(target_classes_list)
        print(f"Updated TARGET_CLASSES from arguments: {TARGET_CLASSES}")
    else:
        print(f"Using default TARGET_CLASSES: {TARGET_CLASSES}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_roboflow_filename(filename):
    """
    Remove Roboflow-generated text from filename.
    
    Args:
        filename: Original filename that may contain .rf. pattern
        
    Returns:
        str: Cleaned filename with .rf. pattern removed
    """
    if ".rf." in filename:
        # Split by .rf. and take only the first part
        base_part = filename.split(".rf.")[0]
        # Get the original extension
        original_ext = os.path.splitext(filename)[1]
        return base_part + original_ext
    return filename


def read_yolo_labels(label_path):
    """
    Reads YOLO format labels from a file.
    
    Args:
        label_path: Path to the label file
        
    Returns:
        List of label lines (strings)
    """
    if not os.path.exists(label_path):
        return []
    
    with open(label_path, 'r') as f:
        lines = f.read().strip().split('\n')
    
    # Filter out empty lines
    return [line for line in lines if line.strip()]

def get_class_from_line(line):
    """
    Extracts class ID from a YOLO format line.
    
    Args:
        line: YOLO format annotation line
        
    Returns:
        int: Class ID, or None if invalid line
    """
    try:
        parts = line.strip().split()
        if parts:
            return int(parts[0])
    except (ValueError, IndexError):
        pass
    return None

def filter_classes(lines, classes_to_exclude):
    """
    Filters out lines containing specific classes.
    
    Args:
        lines: List of YOLO format lines
        classes_to_exclude: Set of class IDs to exclude
        
    Returns:
        List of filtered lines
    """
    filtered_lines = []
    
    for line in lines:
        class_id = get_class_from_line(line)
        if class_id is not None and class_id not in classes_to_exclude:
            filtered_lines.append(line)
    
    return filtered_lines

def combine_labels(inference_lines, original_lines, target_classes={0, 1}):
    """
    Combines inference labels with original labels, replacing target classes.
    
    Args:
        inference_lines: Lines from inference labels (classes 0 and 1)
        original_lines: Lines from original labels (all classes)
        target_classes: Set of classes to replace (default: {0, 1})
        
    Returns:
        List of combined label lines
    """
    # Filter out target classes from original labels
    filtered_original = filter_classes(original_lines, target_classes)
    
    # Combine filtered original with all inference labels
    combined_lines = filtered_original + inference_lines
    
    return combined_lines

def process_label_files(inference_folder, original_folder, output_folder):
    """
    Processes all label files and creates combined versions.
    
    Args:
        inference_folder: Path to inference labels folder
        original_folder: Path to original YOLO labels folder  
        output_folder: Path to output folder for combined labels
        
    Returns:
        Dictionary with processing statistics
    """
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Statistics tracking
    stats = {
        'total_files': 0,
        'processed_files': 0,
        'missing_original': 0,
        'empty_inference': 0,
        'errors': 0
    }
    
    # Get all .txt files from inference folder
    inference_files = [f for f in os.listdir(inference_folder) 
                      if f.endswith('.txt')]
    
    stats['total_files'] = len(inference_files)
    
    print(f"Found {len(inference_files)} files to process")
    
    for filename in inference_files:
        try:
            inference_path = os.path.join(inference_folder, filename)
            original_path = os.path.join(original_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"Processing: {filename}")
            
            # Read inference labels (classes 0 and 1)
            inference_lines = read_yolo_labels(inference_path)
            
            if not inference_lines:
                print(f"  - Warning: Empty inference file")
                stats['empty_inference'] += 1
            
            # Read original labels (all classes)
            original_lines = read_yolo_labels(original_path)
            
            if not os.path.exists(original_path):
                print(f"  - Warning: Original file not found, using only "
                      f"inference labels")
                stats['missing_original'] += 1
                combined_lines = inference_lines
            else:
                # Combine labels (replace classes 0 and 1)
                combined_lines = combine_labels(inference_lines, 
                                              original_lines)
            
            # Write combined labels to output
            with open(output_path, 'w') as f:
                if combined_lines:
                    f.write('\n'.join(combined_lines))
                else:
                    # Create empty file if no annotations
                    f.write('')
            
            # Count classes in final output for verification
            class_counts = {}
            for line in combined_lines:
                class_id = get_class_from_line(line)
                if class_id is not None:
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
            
            print(f"  - Original annotations: {len(original_lines)}")
            print(f"  - Inference annotations: {len(inference_lines)}")
            print(f"  - Combined annotations: {len(combined_lines)}")
            print(f"  - Class distribution: {class_counts}")
            
            stats['processed_files'] += 1
            
        except Exception as e:
            print(f"  - Error processing {filename}: {str(e)}")
            stats['errors'] += 1
    
    return stats

def copy_images_to_upload(src_folder, dest_folder):
    """
    Copy image files from source folder to destination folder.
    
    Args:
        src_folder: Path to source folder containing images
        dest_folder: Path to destination folder
    
    Returns:
        List of copied filenames
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    copied_files = []
    if os.path.exists(src_folder):
        for filename in os.listdir(src_folder):
            if os.path.splitext(filename)[1].lower() in image_extensions:
                src_path = os.path.join(src_folder, filename)
                if os.path.isfile(src_path):
                    # Clean the filename before copying
                    clean_filename = clean_roboflow_filename(filename)
                    dest_path = os.path.join(dest_folder, clean_filename)
                    shutil.copy2(src_path, dest_path)
                    copied_files.append(clean_filename)
    return copied_files


def copy_labels_to_upload(src_folder, dest_folder):
    """
    Copy label files from source folder to destination folder.
    
    Args:
        src_folder: Path to source folder containing label files
        dest_folder: Path to destination folder
    
    Returns:
        List of copied filenames
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    copied_files = []
    if os.path.exists(src_folder):
        for filename in os.listdir(src_folder):
            if filename.endswith('.txt'):
                src_path = os.path.join(src_folder, filename)
                if os.path.isfile(src_path):
                    # Clean the filename before copying
                    clean_filename = clean_roboflow_filename(filename)
                    dest_path = os.path.join(dest_folder, clean_filename)
                    shutil.copy2(src_path, dest_path)
                    copied_files.append(clean_filename)
    return copied_files


def copy_data_yaml(src_yaml_path, dest_folder):
    """
    Copy data.yaml file to destination folder.
    
    Args:
        src_yaml_path: Path to source data.yaml file
        dest_folder: Path to destination folder
    
    Returns:
        bool: True if file was copied, False otherwise
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    if os.path.exists(src_yaml_path) and os.path.isfile(src_yaml_path):
        shutil.copy2(src_yaml_path, dest_folder)
        return True
    return False

def modify_pixel_rgb(image_path):
    """
    Modify the RGB value of pixel at position (0,0) to a random value.
    
    Args:
        image_path: Path to the image file to modify
    
    Returns:
        bool: True if pixel was modified successfully, False otherwise
    """
    try:
        # Open the image
        image = Image.open(image_path)
        
        # Ensure the image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Load pixel data
        pixels = image.load()
        
        # Generate random RGB values
        new_rgb = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Modify the pixel at position (0,0)
        pixels[0, 0] = new_rgb
        
        # Save the modified image
        image.save(image_path)
        
        return True
    except Exception as e:
        print(f"  - Error modifying pixel for {image_path}: {str(e)}")
        return False

def modify_all_copied_images(upload_folder, copied_images):
    """
    Modify the RGB value of pixel (0,0) for all copied images.
    
    Args:
        upload_folder: Path to folder containing copied images
        copied_images: List of copied image filenames
    
    Returns:
        int: Number of successfully modified images
    """
    print("\n" + "=" * 50)
    print("MODIFYING PIXEL RGB VALUES")
    print("=" * 50)
    
    modified_count = 0
    
    for filename in copied_images:
        image_path = os.path.join(upload_folder, filename)
        # print(f"Modifying pixel (0,0) for: {filename}")
        
        if modify_pixel_rgb(image_path):
            modified_count += 1
        #     print(f"  - Successfully modified")
        # else:
        #     print(f"  - Failed to modify")
    
    print(f"\nTotal images with modified pixels: {modified_count}/{len(copied_images)}")
    print("=" * 50)
    
    return modified_count

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function that handles label combination."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Update global variables with arguments
    update_global_variables(args)
    
    print(f"Using target classes for replacement: {TARGET_CLASSES}")
    print(f"Inference folder: {INFERENCE_FOLDER}")
    print(f"Original folder: {ORIGINAL_FOLDER}")  
    print(f"Output folder: {OUTPUT_FOLDER}")
    
    # Validate input folders using the updated global variables
    if not os.path.exists(INFERENCE_FOLDER):
        print(f"Error: Inference folder does not exist: {INFERENCE_FOLDER}")
        return
    
    if not os.path.exists(ORIGINAL_FOLDER):
        print(f"Error: Original folder does not exist: {ORIGINAL_FOLDER}")
        return
    
    print("-" * 50)
    
    # Process all label files using the updated global variables
    stats = process_label_files(
        INFERENCE_FOLDER,
        ORIGINAL_FOLDER, 
        OUTPUT_FOLDER
    )
    
    # Print final statistics
    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total files found: {stats['total_files']}")
    print(f"Successfully processed: {stats['processed_files']}")
    print(f"Missing original files: {stats['missing_original']}")
    print(f"Empty inference files: {stats['empty_inference']}")
    print(f"Errors encountered: {stats['errors']}")
    print(f"\nOutput saved to: {OUTPUT_FOLDER}")


    # Setup upload folder
    if args.upload_folder:
        print("\n" + "=" * 50)
        print("COPYING FILES TO UPLOAD FOLDER")
        print("=" * 50)
        print(f"Upload folder: {args.upload_folder}")
        
        # Define source paths
        test_images_folder = ORIGINAL_IMG_FOLDER
        inference_labels_full_folder = OUTPUT_FOLDER
        original_yolo_data_yaml = ORIGINAL_YOLO_DATA_YAML
        
        # Copy files to upload folder
        copied_images = copy_images_to_upload(test_images_folder, 
                                              args.upload_folder)
        copied_labels = copy_labels_to_upload(inference_labels_full_folder, 
                                              args.upload_folder)
        copied_yaml = copy_data_yaml(original_yolo_data_yaml, 
                                     args.upload_folder)
        
        # Print summary
        print(f"Copied {len(copied_images)} image files")
        print(f"Copied {len(copied_labels)} label files")
        print(f"Copied data.yaml file: {copied_yaml}")
        print(f"Upload folder created at: {args.upload_folder}")
        print("=" * 50)
        
        # Modify pixel RGB values for all copied images
        if copied_images:
            modify_all_copied_images(args.upload_folder, copied_images)


if __name__ == "__main__":
    main()
    # datasets/inference/labels datasets/original_yolo/test/labels --output_folder datasets/inference/labels_full

