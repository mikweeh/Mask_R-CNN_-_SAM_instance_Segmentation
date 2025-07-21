#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 This script filters the main COCO annotation file based on two criteria:
 - a specific list of image files for train, valid, and test sets
 - and a list of target classes


 The output will be three COCO annotation files that only contain the image
 files and the classes specified for each set. It also copies the corresponding
 image files to their respective output directories.


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


def filter_coco_dataset(images_to_keep: List[str], new_annotation_file: str, 
                        dataset_name: str, original_annotation_file: str) -> None:
    """
    Filters a COCO dataset to keep only specified images and target classes.
    Also copies the corresponding image files to the output directory.
    
    Args:
        images_to_keep: List of image filename prefixes to keep
        new_annotation_file: Path where to save the filtered annotation file
        dataset_name: Name of the dataset (train/valid/test) for logging
        original_annotation_file: Path to the original annotation file
    """
    print(f"\n=== Processing {dataset_name.upper()} dataset ===")
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


    # 1. Find the target categories and their IDs.
    target_category_ids: Set[int] = set()
    category_id_mapping: Dict[int, int] = {}
    original_categories = coco_data.get('categories', [])
    
    new_category_id = 1
    for category in original_categories:
        if category['name'] in CLASSES_TO_KEEP:
            original_id = category['id']
            target_category_ids.add(original_id)
            category_id_mapping[original_id] = new_category_id
            
            # Add the category to filtered data with new ID
            filtered_coco['categories'].append({
                'id': new_category_id,
                'name': category['name'],
                'supercategory': category.get('supercategory', '')
            })
            new_category_id += 1


    if not target_category_ids:
        print(f"ERROR: None of the classes {CLASSES_TO_KEEP} were found in "
              f"the original categories.")
        return


    print(f"Found {len(target_category_ids)} target classes: "
          f"{[cat['name'] for cat in filtered_coco['categories']]}")


    # 2. Get the image IDs for the specified filenames.
    image_ids_to_keep: Set[int] = set()
    for image_info in coco_data.get('images', []):
        if any(image_info['file_name'].startswith(prefix) for prefix in images_to_keep):
            image_ids_to_keep.add(image_info['id'])
    
    if not image_ids_to_keep:
        print(f"WARNING: None of the specified images were found in the "
              f"annotation file for {dataset_name} dataset.")
    else:
        print(f"Found {len(image_ids_to_keep)} of the specified images to "
              f"process for {dataset_name} dataset.")


    # 3. Filter annotations based on image IDs and target category IDs.
    kept_image_ids_from_annotations: Set[int] = set()
    for annotation in coco_data.get('annotations', []):
        # Check if annotation belongs to target images AND target classes
        if (annotation['image_id'] in image_ids_to_keep and 
            annotation['category_id'] in target_category_ids):
            # Re-index the category ID using the mapping
            annotation['category_id'] = category_id_mapping[
                annotation['category_id']
            ]
            filtered_coco['annotations'].append(annotation)
            kept_image_ids_from_annotations.add(annotation['image_id'])


    print(f"Filtered to {len(filtered_coco['annotations'])} annotations "
          f"for classes {CLASSES_TO_KEEP} in {dataset_name} dataset.")


    # 4. Filter the images list to include only those that have annotations.
    for image_info in coco_data.get('images', []):
        if image_info['id'] in kept_image_ids_from_annotations:
            filtered_coco['images'].append(image_info)
    
    print(f"Final {dataset_name} dataset contains "
          f"{len(filtered_coco['images'])} images with relevant annotations.")


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
            print(f"WARNING: Could not access directory {original_images_dir}")
            break
    
    print(f"Copied {len(copied_files)} images to {output_dir}")
    print(f"=== {dataset_name.upper()} dataset processing complete! ===")


def main() -> None:
    """
    Main function that processes all three datasets (train, valid, test).
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Filter COCO dataset by images and classes, creating '
                   'train/valid/test splits'
    )
    parser.add_argument(
        'dataset_path',
        nargs='?',
        default='./dataset',
        help='Path to the dataset directory (default: ./dataset)'
    )
    
    # Add classes_to_keep argument
    parser.add_argument('--classes_to_keep', type=str, default=None,
                       help='JSON string with list of class names to keep')

    args = parser.parse_args()
    dataset_path = args.dataset_path
    
    print(f"Using dataset path: {dataset_path}")
    
    # Update CLASSES_TO_KEEP from arguments if provided
    global CLASSES_TO_KEEP
    if args.classes_to_keep is not None:
        CLASSES_TO_KEEP = json.loads(args.classes_to_keep)

    # --- Input Files ---
    # Path to the original COCO annotation file (automatically detected)
    original_coco_dir = os.path.join(dataset_path, 'original_coco')
    try:
        try:
            # Try to find annotation file in original_coco_dir
            original_annotation_file = find_coco_annotation_file(
                original_coco_dir
            )
        except FileNotFoundError:
            # Try to find annotation file in original_coco/train/
            alt_coco_dir = os.path.join(original_coco_dir, 'train')
            original_annotation_file = find_coco_annotation_file(
                alt_coco_dir
            )
            original_coco_dir = alt_coco_dir  # Update to new working dir
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    # --- Output Files ---
    # Paths where the new filtered annotation files will be saved.
    new_annotation_file_train = os.path.join(dataset_path, 'train', 
                                           '_annotations_filtered.coco.json')
    new_annotation_file_valid = os.path.join(dataset_path, 'valid', 
                                           '_annotations_filtered.coco.json')
    new_annotation_file_test = os.path.join(dataset_path, 'test', 
                                          '_annotations_filtered.coco.json')
    
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
    
    print("Starting COCO dataset filtering for all splits...")
    
    # Process train dataset
    filter_coco_dataset(images_to_keep_train, new_annotation_file_train, 
                       "train", original_annotation_file)
    
    # Process validation dataset
    filter_coco_dataset(images_to_keep_valid, new_annotation_file_valid, 
                       "valid", original_annotation_file)
    
    # Process test dataset
    filter_coco_dataset(images_to_keep_test, new_annotation_file_test, 
                       "test", original_annotation_file)
    
    print("\n" + "="*60)
    print("ALL DATASETS PROCESSED SUCCESSFULLY!")
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
