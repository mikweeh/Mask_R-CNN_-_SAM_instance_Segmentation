#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script filters the main COCO annotation file for a SINGLE target
class (for SAM2 binary segmentation training).

Modified from original to support single-class filtering for SAM2.


 The dataset folder structure is expected to be:

 dataset/
   train/               # Folder containing training images
   valid/               # Folder containing validation images
   test/                # Folder containing test images
   original_coco/       # Folder containing the original COCO annotation file and original images
   original_yolo/       # (Not used in this script)
   train.txt            # Text file listing image filenames for training
   valid.txt            # Text file listing image filenames for validation
   test.txt             # Text file listing image filenames for test

 The text files can contain filenames with or without extensions, with or without
 quotes, commas, and comments starting with '#'. The script will parse these files
 to extract clean lists of image filename prefixes.
"""

import argparse
import json
import os
import shutil
from typing import Dict, List, Set

# =============================================================================
# DEFAULT GLOBAL CONFIGURATION VARIABLES
# =============================================================================

# Default class configuration - can be overridden by arguments from main.py
CLASSES_TO_KEEP = [
    'Chromis chromis',
    'Coris julis',
]

# New parameter for single-class filtering
TARGET_CLASS_INDEX = 0  # Which class to filter (0 or 1)

# =============================================================================
# Functions
# =============================================================================

def load_image_list_from_txt(file_path: str) -> List[str]:
    """
    Loads a list of image filename prefixes from a text file.
    Ignores comments and empty lines. Strips quotes, commas, whitespace,
    and common image extensions to ensure proper prefix matching.
    """
    image_list = []
    # Common image extensions to remove
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                        '.webp', '.gif')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Strip whitespace first
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Remove trailing commas
                while line.endswith(','):
                    line = line[:-1].strip()
                
                # Remove surrounding quotes
                while ((line.startswith('"') and line.endswith('"')) or
                       (line.startswith("'") and line.endswith("'"))):
                    line = line[1:-1].strip()
                
                # Remove common image extensions (case insensitive)
                line_lower = line.lower()
                for ext in image_extensions:
                    if line_lower.endswith(ext):
                        line = line[:-len(ext)]
                        break
                
                # Add the cleaned filename if it's not empty
                if line:
                    image_list.append(line)
                    
    except FileNotFoundError:
        print(f"WARNING: File {file_path} not found. Returning empty list.")
    except Exception as e:
        print(f"ERROR: Could not read file {file_path}: {e}")
    
    return image_list


def find_coco_annotation_file(original_coco_dir: str) -> str:
    """
    Finds the unique JSON annotation file in the original_coco directory.
    
    Args:
        original_coco_dir: Path to the original_coco directory
    
    Returns:
        Full path to the annotation file
    
    Raises:
        FileNotFoundError: If no JSON file or multiple JSON files are found
    """
    json_files = [f for f in os.listdir(original_coco_dir)
                  if f.endswith('.json')]
    
    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON annotation file found in "
                                f"{original_coco_dir}")
    elif len(json_files) > 1:
        raise FileNotFoundError(f"Multiple JSON files found in "
                                f"{original_coco_dir}. Expected only one: "
                                f"{json_files}")
    
    return os.path.join(original_coco_dir, json_files[0])


def create_list_file_if_missing(txt_path: str, images_dir: str) -> None:
    """
    Create a text file listing image filenames with a trailing comma if
    missing.
    
    Args:
        txt_path: Path to the .txt file to create
        images_dir: Directory containing the image files to list
    """
    if not os.path.exists(txt_path):
        print(f"Creating missing file: {txt_path}")
        
        # List image files in images_dir
        try:
            files = os.listdir(images_dir)
        except FileNotFoundError:
            print(f"WARNING: Directory {images_dir} not found, "
                  f"cannot create {txt_path}")
            return
        
        # Filter to common image extensions
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff',
                            '.tif', '.webp', '.gif')
        image_files = [f for f in files
                       if f.lower().endswith(image_extensions)]
        
        # Write filenames (with extension) one per line ending with a comma
        with open(txt_path, 'w', encoding='utf-8') as f:
            for filename in sorted(image_files):
                f.write(filename + ',\n')
        
        print(f"Created {txt_path} with {len(image_files)} image filenames "
              f"from {images_dir}")
    else:
        print(f"File {txt_path} already exists, skipping creation.")


def filter_coco_dataset(images_to_keep: List[str],
                        new_annotation_file: str,
                        dataset_name: str,
                        original_annotation_file: str,
                        target_class_name: str) -> None:
    """
    Filters a COCO dataset to keep only specified images and ONE target
    class for SAM2 binary segmentation.
    Also copies the corresponding image files to the output directory.
    
    Args:
        images_to_keep: List of image filename prefixes to keep
        new_annotation_file: Path where to save the filtered annotation file
        dataset_name: Name of the dataset (train/valid/test) for logging
        original_annotation_file: Path to the original annotation file
        target_class_name: Single class name to keep (binary segmentation)
    """
    print(f"\n=== Processing {dataset_name.upper()} dataset ===")
    print(f"Target class for binary segmentation: {target_class_name}")
    print(f"Loading original annotations from: {original_annotation_file}")
    
    try:
        with open(original_annotation_file, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: The file {original_annotation_file} was not found.")
        return
    
    # Create a new dictionary to hold the filtered data.
    filtered_coco = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # 1. Find the target category and its ID (only ONE class for SAM2)
    target_category_id = None
    original_categories = coco_data.get('categories', [])
    
    for category in original_categories:
        if category['name'] == target_class_name:
            target_category_id = category['id']
            # For SAM2, we use category ID = 1 (binary: target vs background)
            filtered_coco['categories'].append({
                'id': 1,
                'name': category['name'],
                'supercategory': category.get('supercategory', '')
            })
            break
    
    if target_category_id is None:
        print(f"ERROR: Target class '{target_class_name}' not found in "
              f"the original categories.")
        return
    
    print(f"Found target class: {target_class_name} "
          f"(original ID: {target_category_id})")
    
    # 2. Get the image IDs for the specified filenames.
    image_ids_to_keep: Set[int] = set()
    for image_info in coco_data.get('images', []):
        if any(image_info['file_name'].startswith(prefix)
               for prefix in images_to_keep):
            image_ids_to_keep.add(image_info['id'])
    
    if not image_ids_to_keep:
        print(f"WARNING: None of the specified images were found in the "
              f"annotation file for {dataset_name} dataset.")
    else:
        print(f"Found {len(image_ids_to_keep)} of the specified images to "
              f"process for {dataset_name} dataset.")
    
    # 3. Filter annotations for target class only (SAM2 binary segmentation)
    kept_image_ids_from_annotations: Set[int] = set()
    for annotation in coco_data.get('annotations', []):
        # Check if annotation belongs to target images AND target class
        if (annotation['image_id'] in image_ids_to_keep and
                annotation['category_id'] == target_category_id):
            # Set category ID to 1 for SAM2 binary segmentation
            annotation['category_id'] = 1
            filtered_coco['annotations'].append(annotation)
            kept_image_ids_from_annotations.add(annotation['image_id'])
    
    print(f"Filtered to {len(filtered_coco['annotations'])} annotations "
          f"for class '{target_class_name}' in {dataset_name} dataset.")
    
    # 4. Filter the images list to include only those that have annotations.
    for image_info in coco_data.get('images', []):
        if image_info['id'] in kept_image_ids_from_annotations:
            filtered_coco['images'].append(image_info)
    
    print(f"Final {dataset_name} dataset contains "
          f"{len(filtered_coco['images'])} images with relevant "
          f"annotations.")
    
    # 5. Save the new filtered annotation file.
    print(f"Saving filtered annotations to: {new_annotation_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(new_annotation_file), exist_ok=True)
    
    with open(new_annotation_file, 'w') as f:
        json.dump(filtered_coco, f, indent=4)
    
    # 6. Copy image files to output directory
    print(f"Copying image files for {dataset_name} dataset...")
    original_images_dir = os.path.dirname(original_annotation_file)
    output_dir = os.path.dirname(new_annotation_file)
    
    copied_files = []
    for image_prefix in images_to_keep:
        # Find files that start with the given prefix
        try:
            for file_name in os.listdir(original_images_dir):
                if file_name.startswith(image_prefix):
                    src_path = os.path.join(original_images_dir, file_name)
                    dst_path = os.path.join(output_dir, file_name)
                    shutil.copy2(src_path, dst_path)
                    copied_files.append(file_name)
                    break  # Only copy the first matching file for each prefix
        except FileNotFoundError:
            print(f"WARNING: Could not access directory "
                  f"{original_images_dir}")
            break
    
    print(f"Copied {len(copied_files)} images to {output_dir}")
    print(f"=== {dataset_name.upper()} dataset processing complete! ===")


def main() -> None:
    """
    Main function that processes all three datasets (train, valid, test)
    for a SINGLE target class (SAM2 binary segmentation).
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Filter COCO dataset for single class '
                    '(SAM2 binary segmentation)'
    )
    
    parser.add_argument(
        'dataset_path',
        nargs='?',
        default='./dataset',
        help='Path to the dataset directory (default: ./dataset)'
    )
    
    # Add classes_to_keep argument
    parser.add_argument('--classes_to_keep', type=str, default=None,
                        help='JSON string with list of class names')
    
    # Add target_class_index argument for single-class filtering
    parser.add_argument('--target_class_index', type=int, default=0,
                        help='Index of target class to filter (0 or 1)')
    
    args = parser.parse_args()
    dataset_path = args.dataset_path
    
    print(f"Using dataset path: {dataset_path}")
    
    # Create train.txt, valid.txt, test.txt if missing
    print("Checking for missing .txt files and creating if necessary...")
    create_list_file_if_missing(
        os.path.join(dataset_path, 'train.txt'),
        os.path.join(dataset_path, 'original_coco', 'train')
    )
    create_list_file_if_missing(
        os.path.join(dataset_path, 'valid.txt'),
        os.path.join(dataset_path, 'original_coco', 'valid')
    )
    create_list_file_if_missing(
        os.path.join(dataset_path, 'test.txt'),
        os.path.join(dataset_path, 'original_coco', 'test')
    )
    
    # Update CLASSES_TO_KEEP from arguments if provided
    global CLASSES_TO_KEEP, TARGET_CLASS_INDEX
    if args.classes_to_keep is not None:
        CLASSES_TO_KEEP = json.loads(args.classes_to_keep)
    
    TARGET_CLASS_INDEX = args.target_class_index
    
    # Select the single target class for this run
    if TARGET_CLASS_INDEX >= len(CLASSES_TO_KEEP):
        print(f"ERROR: target_class_index {TARGET_CLASS_INDEX} is out of "
              f"range. Available classes: {CLASSES_TO_KEEP}")
        return
    
    target_class_name = CLASSES_TO_KEEP[TARGET_CLASS_INDEX]
    print(f"\n{'='*60}")
    print(f"FILTERING FOR SINGLE CLASS: {target_class_name}")
    print(f"Class index: {TARGET_CLASS_INDEX}")
    print(f"{'='*60}\n")
    
    # --- Input Files
    original_coco_dir = os.path.join(dataset_path, 'original_coco')
    
    # Define paths for each split's annotation file
    train_coco_dir = os.path.join(original_coco_dir, 'train')
    valid_coco_dir = os.path.join(original_coco_dir, 'valid')
    test_coco_dir = os.path.join(original_coco_dir, 'test')
    
    # Find annotation files for each split
    try:
        train_annotation_file = find_coco_annotation_file(train_coco_dir)
        print(f"Found train annotation file: {train_annotation_file}")
    except FileNotFoundError as e:
        print(f"ERROR: Could not find train annotation file: {e}")
        return
    
    try:
        valid_annotation_file = find_coco_annotation_file(valid_coco_dir)
        print(f"Found valid annotation file: {valid_annotation_file}")
    except FileNotFoundError as e:
        print(f"ERROR: Could not find valid annotation file: {e}")
        return
    
    try:
        test_annotation_file = find_coco_annotation_file(test_coco_dir)
        print(f"Found test annotation file: {test_annotation_file}")
    except FileNotFoundError as e:
        print(f"ERROR: Could not find test annotation file: {e}")
        return
    
    # --- Output Files ---
    # Paths where the new filtered annotation files will be saved.
    new_annotation_file_train = os.path.join(
        dataset_path, 'train', '_annotations_filtered.coco.json'
    )
    new_annotation_file_valid = os.path.join(
        dataset_path, 'valid', '_annotations_filtered.coco.json'
    )
    new_annotation_file_test = os.path.join(
        dataset_path, 'test', '_annotations_filtered.coco.json'
    )
    
    # --- Filtering Criteria ---
    # Load image filename prefixes from text files
    images_to_keep_train = load_image_list_from_txt(
        os.path.join(dataset_path, 'train.txt')
    )
    images_to_keep_valid = load_image_list_from_txt(
        os.path.join(dataset_path, 'valid.txt')
    )
    images_to_keep_test = load_image_list_from_txt(
        os.path.join(dataset_path, 'test.txt')
    )
    
    print("Starting COCO dataset filtering for single class...")
    
    # Process train dataset with single target class
    filter_coco_dataset(images_to_keep_train, new_annotation_file_train,
                        "train", train_annotation_file, target_class_name)
    
    # Process validation dataset with single target class
    filter_coco_dataset(images_to_keep_valid, new_annotation_file_valid,
                        "valid", valid_annotation_file, target_class_name)
    
    # Process test dataset with single target class
    filter_coco_dataset(images_to_keep_test, new_annotation_file_test,
                        "test", test_annotation_file, target_class_name)
    
    print("\n" + "="*60)
    print("ALL DATASETS PROCESSED SUCCESSFULLY!")
    print(f"Filtered for class: {target_class_name} (index "
          f"{TARGET_CLASS_INDEX})")
    print("Created directories:")
    print(f"  - {dataset_path}/train/ with {len(images_to_keep_train)} "
          f"image prefixes")
    print(f"  - {dataset_path}/valid/ with {len(images_to_keep_valid)} "
          f"image prefixes")
    print(f"  - {dataset_path}/test/ with {len(images_to_keep_test)} "
          f"image prefixes")
    print("="*60)


if __name__ == '__main__':
    main()
