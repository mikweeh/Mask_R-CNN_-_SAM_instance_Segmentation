#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter COCO dataset - Creates multiple annotation files in the same folder.

Structure:
dataset/
├── train/
│   ├── images...
│   ├── _annotations.coco.json          # Multi-class (for Mask R-CNN)
│   ├── _annotations_class0.coco.json   # Class 0 only (for SAM2)
│   └── _annotations_class1.coco.json   # Class 1 only (for SAM2)
├── valid/  # Same structure
└── test/   # Same structure
"""

import argparse
import json
import os
import shutil
from typing import Dict, List, Set

# Default classes
CLASSES_TO_KEEP = ['Chromis chromis', 'Coris julis']


def load_image_list_from_txt(file_path: str) -> List[str]:
    """Load image filename prefixes from text file."""
    image_list = []
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', 
                       '.webp', '.gif')
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                while line.endswith(','):
                    line = line[:-1].strip()
                
                while ((line.startswith('"') and line.endswith('"')) or 
                       (line.startswith("'") and line.endswith("'"))):
                    line = line[1:-1].strip()
                
                line_lower = line.lower()
                for ext in image_extensions:
                    if line_lower.endswith(ext):
                        line = line[:-len(ext)]
                        break
                
                if line:
                    image_list.append(line)
                    
    except FileNotFoundError:
        print(f"WARNING: File {file_path} not found.")
    except Exception as e:
        print(f"ERROR: Could not read file {file_path}: {e}")
    
    return image_list


def find_coco_annotation_file(original_coco_dir: str) -> str:
    """Find unique JSON annotation file in directory."""
    json_files = [f for f in os.listdir(original_coco_dir) 
                  if f.endswith('.json')]
    
    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON file in {original_coco_dir}")
    elif len(json_files) > 1:
        raise FileNotFoundError(f"Multiple JSON files in {original_coco_dir}: {json_files}")
    
    return os.path.join(original_coco_dir, json_files[0])


def create_list_file_if_missing(txt_path: str, images_dir: str) -> None:
    """Create text file listing images if missing."""
    if not os.path.exists(txt_path):
        print(f"Creating {txt_path}")
        try:
            files = os.listdir(images_dir)
        except FileNotFoundError:
            print(f"WARNING: Directory {images_dir} not found")
            return
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', 
                          '.tif', '.webp', '.gif')
        image_files = [f for f in files 
                      if f.lower().endswith(image_extensions)]
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            for filename in sorted(image_files):
                f.write(filename + ',\n')
        
        print(f"Created {txt_path} with {len(image_files)} images")


def filter_coco_annotations(images_to_keep: List[str], output_file: str,
                           dataset_name: str, original_annotation_file: str,
                           target_classes: List[str], class_filter_mode: str = 'multi'):
    """
    Filter COCO annotations.
    
    Args:
        images_to_keep: List of image filename prefixes
        output_file: Where to save filtered annotations
        dataset_name: Dataset name (train/valid/test)
        original_annotation_file: Original COCO annotations
        target_classes: List of class names to keep
        class_filter_mode: 'multi' for all classes, 'class0', 'class1', etc. for single class
    """
    print(f"\n=== Processing {dataset_name.upper()} - {class_filter_mode} ===")
    print(f"Loading: {original_annotation_file}")
    
    try:
        with open(original_annotation_file, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {original_annotation_file}")
        return

    filtered_coco = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'images': [],
        'annotations': [],
        'categories': []
    }

    # Determine which classes to include
    if class_filter_mode == 'multi':
        classes_to_include = target_classes
    else:
        # Extract class index from mode (e.g., 'class0' -> 0)
        class_idx = int(class_filter_mode.replace('class', ''))
        if class_idx < len(target_classes):
            classes_to_include = [target_classes[class_idx]]
        else:
            print(f"ERROR: Invalid class index {class_idx}")
            return

    # Find target categories
    target_category_ids: Set[int] = set()
    category_id_mapping: Dict[int, int] = {}
    original_categories = coco_data.get('categories', [])
    
    new_category_id = 1
    for category in original_categories:
        if category['name'] in classes_to_include:
            original_id = category['id']
            target_category_ids.add(original_id)
            category_id_mapping[original_id] = new_category_id
            
            filtered_coco['categories'].append({
                'id': new_category_id,
                'name': category['name'],
                'supercategory': category.get('supercategory', '')
            })
            new_category_id += 1

    if not target_category_ids:
        print(f"ERROR: No classes found: {classes_to_include}")
        return

    print(f"Found {len(target_category_ids)} classes: "
          f"{[cat['name'] for cat in filtered_coco['categories']]}")

    # Get image IDs for specified filenames
    image_ids_to_keep: Set[int] = set()
    for image_info in coco_data.get('images', []):
        if any(image_info['file_name'].startswith(prefix) for prefix in images_to_keep):
            image_ids_to_keep.add(image_info['id'])
    
    if not image_ids_to_keep:
        print(f"WARNING: No images found for {dataset_name}")
    else:
        print(f"Found {len(image_ids_to_keep)} images")

    # Filter annotations
    kept_image_ids: Set[int] = set()
    for annotation in coco_data.get('annotations', []):
        if (annotation['image_id'] in image_ids_to_keep and 
            annotation['category_id'] in target_category_ids):
            annotation['category_id'] = category_id_mapping[annotation['category_id']]
            filtered_coco['annotations'].append(annotation)
            kept_image_ids.add(annotation['image_id'])

    print(f"Filtered to {len(filtered_coco['annotations'])} annotations")

    # Filter images list
    for image_info in coco_data.get('images', []):
        if image_info['id'] in kept_image_ids:
            filtered_coco['images'].append(image_info)
    
    print(f"Final: {len(filtered_coco['images'])} images with annotations")

    # Save
    print(f"Saving to: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(filtered_coco, f, indent=4)
    
    print(f"=== {dataset_name.upper()} - {class_filter_mode} complete ===")


def copy_images_to_folder(images_to_keep: List[str], original_images_dir: str,
                         output_dir: str):
    """Copy images to output folder."""
    print(f"\nCopying images to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    copied_files = []
    try:
        for image_prefix in images_to_keep:
            for file_name in os.listdir(original_images_dir):
                if file_name.startswith(image_prefix):
                    src_path = os.path.join(original_images_dir, file_name)
                    dst_path = os.path.join(output_dir, file_name)
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
                        copied_files.append(file_name)
                    break
    except FileNotFoundError:
        print(f"WARNING: Could not access {original_images_dir}")
    
    print(f"Copied {len(copied_files)} images")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Filter COCO dataset - creates multiple annotation files'
    )
    
    parser.add_argument(
        'dataset_path',
        nargs='?',
        default='./dataset',
        help='Path to dataset directory'
    )
    
    parser.add_argument('--classes_to_keep', type=str, default=None,
                       help='JSON list of class names')
    
    args = parser.parse_args()
    dataset_path = args.dataset_path
    print(f"Dataset path: {dataset_path}")

    # Create .txt files if missing
    print("Checking for .txt files...")
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

    # Update classes
    global CLASSES_TO_KEEP
    if args.classes_to_keep is not None:
        CLASSES_TO_KEEP = json.loads(args.classes_to_keep)
    
    print(f"\nTarget classes: {CLASSES_TO_KEEP}")
    
    # Find annotation files
    original_coco_dir = os.path.join(dataset_path, 'original_coco')
    
    try:
        train_annotation = find_coco_annotation_file(os.path.join(original_coco_dir, 'train'))
        valid_annotation = find_coco_annotation_file(os.path.join(original_coco_dir, 'valid'))
        test_annotation = find_coco_annotation_file(os.path.join(original_coco_dir, 'test'))
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # Load image lists
    images_train = load_image_list_from_txt(os.path.join(dataset_path, 'train.txt'))
    images_valid = load_image_list_from_txt(os.path.join(dataset_path, 'valid.txt'))
    images_test = load_image_list_from_txt(os.path.join(dataset_path, 'test.txt'))
    
    print("\nStarting dataset filtering...")
    
    # Process each split
    for split_name, images_list, original_ann, original_img_dir in [
        ('train', images_train, train_annotation, os.path.join(original_coco_dir, 'train')),
        ('valid', images_valid, valid_annotation, os.path.join(original_coco_dir, 'valid')),
        ('test', images_test, test_annotation, os.path.join(original_coco_dir, 'test'))
    ]:
        output_dir = os.path.join(dataset_path, split_name)
        
        # Copy images (only once)
        copy_images_to_folder(images_list, original_img_dir, output_dir)
        
        # Create multi-class annotation (for Mask R-CNN)
        filter_coco_annotations(
            images_list,
            os.path.join(output_dir, '_annotations.coco.json'),
            split_name,
            original_ann,
            CLASSES_TO_KEEP,
            class_filter_mode='multi'
        )
        
        # Create single-class annotations (for SAM2)
        for class_idx in range(len(CLASSES_TO_KEEP)):
            filter_coco_annotations(
                images_list,
                os.path.join(output_dir, f'_annotations_class{class_idx}.coco.json'),
                split_name,
                original_ann,
                CLASSES_TO_KEEP,
                class_filter_mode=f'class{class_idx}'
            )
    
    print("\n" + "="*80)
    print("ALL DATASETS PROCESSED SUCCESSFULLY!")
    print(f"Created directories with multiple annotation files:")
    print(f" - {dataset_path}/train/")
    print(f" - {dataset_path}/valid/")
    print(f" - {dataset_path}/test/")
    print("\nEach folder contains:")
    print(f" - _annotations.coco.json (multi-class for Mask R-CNN)")
    for i in range(len(CLASSES_TO_KEEP)):
        print(f" - _annotations_class{i}.coco.json (class {i} for SAM2)")
    print("="*80)


if __name__ == '__main__':
    main()

