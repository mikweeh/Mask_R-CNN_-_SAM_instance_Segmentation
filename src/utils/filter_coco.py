#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 This script filters the main COCO annotation file based on two criteria:
 - a specific list of image files
 - and a list of target classes

 The output will be a COCO annotation file that only contains the image
 files and the classes specified.
"""

import json
import os
from typing import Dict, List, Set

# ====================================================================
# CONFIGURATION - MODIFY THESE VALUES
# ====================================================================

# --- Input Files ---
# Path to the original COCO annotation file you downloaded from Roboflow.
# This script should be run separately for your 'train' and 'valid' sets.
ORIGINAL_ANNOTATION_FILE = 'dataset/_annotations.coco.json'

TRAIN = False  # When True it filters the train dataset,
              # otherwise the validation dataset

# --- Output Files ---
# Path where the new, filtered annotation file will be saved.
NEW_ANNOTATION_FILE = 'dataset/train/_annotations_filtered.coco.json' if TRAIN else \
    'dataset/valid/_annotations_filtered.coco.json'

# --- Filtering Criteria ---

# List of the exact filenames of the images you want to keep.
IMAGES_TO_KEEP_TRAIN = [
    '2023-10-16_Mero_Morena_frame186_jpg',
    '2023-10-16_Mero_frame9_jpg',
    '2023-10-16_Morena_frame38_jpg',
    '20240322-165801-IPC608_8BC7_166_jpg',
]
IMAGES_TO_KEEP_VALID = [
    # Validation
    '2023-10-16_Mero_Morena_frame180_jpg',
]

IMAGES_TO_KEEP = IMAGES_TO_KEEP_TRAIN if TRAIN else IMAGES_TO_KEEP_VALID

# List of class names you are interested in.
CLASSES_TO_KEEP = [
    'Chromis chromis',
    'Coris julis',
]

# ====================================================================
# SCRIPT LOGIC - DO NOT MODIFY BELOW THIS LINE
# ====================================================================

def filter_coco_dataset() -> None:
    """
    Filters a COCO dataset to keep only specified images and target classes.
    
    This function reads the original COCO annotation file, filters it based
    on the specified image files and class names, and saves the filtered
    result to a new file.
    """
    print(f"Loading original annotations from: {ORIGINAL_ANNOTATION_FILE}")
    try:
        with open(ORIGINAL_ANNOTATION_FILE, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: The file {ORIGINAL_ANNOTATION_FILE} was not found.")
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
        if any(image_info['file_name'].startswith(prefix) for prefix in IMAGES_TO_KEEP):
            image_ids_to_keep.add(image_info['id'])
    
    if not image_ids_to_keep:
        print(f"WARNING: None of the specified images were found in the "
              f"annotation file.")
    else:
        print(f"Found {len(image_ids_to_keep)} of the specified images to "
              f"process.")

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
          f"for classes {CLASSES_TO_KEEP}.")

    # 4. Filter the images list to include only those that have annotations.
    for image_info in coco_data.get('images', []):
        if image_info['id'] in kept_image_ids_from_annotations:
            filtered_coco['images'].append(image_info)
    
    print(f"Final dataset contains {len(filtered_coco['images'])} images "
          f"with relevant annotations.")

    # 5. Save the new filtered annotation file.
    print(f"Saving filtered annotations to: {NEW_ANNOTATION_FILE}")
    with open(NEW_ANNOTATION_FILE, 'w') as f:
        json.dump(filtered_coco, f, indent=4)

    print("Filtering complete!")


if __name__ == '__main__':
    filter_coco_dataset()
