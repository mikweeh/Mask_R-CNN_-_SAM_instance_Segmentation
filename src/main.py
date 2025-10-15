#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script that runs the complete SAM2 fish segmentation pipeline:
1. Filter COCO dataset for single class (filter_coco.py)
2. Train SAM2 model for that class (train_sam2.py)
3. Inference with SAM2 automatic mask generation (infer_sam2.py)
4. Combine inference labels with original YOLO labels (adapt2rbf.py)

For two-class problem, this pipeline runs twice (once per class).
"""

import argparse
import subprocess
import sys
import os
import json
import shutil
from datetime import datetime

# =============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# =============================================================================

# Dataset configuration
DATASET_PATH = "dataset"

# =============================================================================
# CENTRALIZED CLASS CONFIGURATION - DEFINE ONLY ONCE HERE
# =============================================================================

# Class names as they appear in COCO annotations
CLASSES_TO_KEEP = ["Chromis chromis", "Coris julis"]

# YOLO class IDs that will be replaced by inference results (adapt2rbf.py)
TARGET_CLASSES_FOR_REPLACEMENT = [0, 1]

# =============================================================================
# SAM2 MODEL CONFIGURATION - USING HUGGING FACE
# =============================================================================

# Hugging Face model ID (no need for manual checkpoint downloads)
SAM2_MODEL_ID = "facebook/sam2-hiera-large"  # Options:
# - "facebook/sam2-hiera-tiny"
# - "facebook/sam2-hiera-small" 
# - "facebook/sam2-hiera-base-plus"
# - "facebook/sam2-hiera-large"
# - "facebook/sam2.1-hiera-large" (newest version)

# Base model names for each class
BASE_MODEL_NAMES = {
    0: "sam2_chromis",  # Chromis chromis
    1: "sam2_coris"     # Coris julis
}

# Training parameters (simpler than Mask R-CNN)
NUM_CLASSES = 1  # Binary segmentation per model
BATCH_SIZE = 1
NUM_EPOCHS = 50  # Fewer epochs than Mask R-CNN
LEARNING_RATE = 1e-5
IMG_SIZE = 1024  # SAM2 default input size

# SAM2 Automatic Mask Generator parameters
POINTS_PER_SIDE = 32
PRED_IOU_THRESH = 0.88
STABILITY_SCORE_THRESH = 0.95
MIN_MASK_REGION_AREA = 100

# Output configuration
REPORT_OUTPUT_PATH = "results"
INFERENCE_FOLDER_PATH = os.path.join(DATASET_PATH, "inference")

# Inference configuration
INPUT_IMAGES_FOLDER = os.path.join(DATASET_PATH, "test")
OUTPUT_LABELS_FOLDER = os.path.join(INFERENCE_FOLDER_PATH, "labels")
OUTPUT_IMAGES_FOLDER = os.path.join(INFERENCE_FOLDER_PATH, "images")

# =============================================================================
# LABEL COMBINATION CONFIGURATION
# =============================================================================

# Label combination configuration (for adapt2rbf.py)
INFERENCE_LABELS_FOLDER = os.path.join(INFERENCE_FOLDER_PATH, "labels")
ORIGINAL_YOLO_LABELS_FOLDER = os.path.join(DATASET_PATH,
                                            "original_yolo/test/labels")
COMBINED_LABELS_OUTPUT_FOLDER = os.path.join(INFERENCE_FOLDER_PATH,
                                              "labels_full")
UPLOAD_FOLDER = os.path.join(DATASET_PATH, "upload")

# Generate class names mapping for YOLO
CLASS_NAMES_MAPPING = {i: name for i, name in enumerate(CLASSES_TO_KEEP)}

# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

parser = argparse.ArgumentParser(
    description='SAM2 Fish segmentation pipeline'
)
parser.add_argument('--mode', type=str,
                    choices=['full', 'inference', 'single_class'],
                    default='full',
                    help='Execution mode: "full" runs both classes, '
                         '"inference" only inference for both classes, '
                         '"single_class" runs one class only')
parser.add_argument('--target_class', type=int, default=None,
                    choices=[0, 1],
                    help='Target class index for single_class mode (0 or 1)')
args = parser.parse_args()

# =============================================================================
# MODEL NAMING LOGIC
# =============================================================================

def get_next_model_name(base_name):
    """
    Get the next available model name to avoid overwriting existing models.
    
    Args:
        base_name: Base name for the model (e.g., "sam2_chromis")
    
    Returns:
        Full path to the model file (e.g., "weights/sam2_chromis.pth")
    """
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        print(f"Created weights directory: {weights_dir}")
    
    # Check if base model exists
    base_path = f"{weights_dir}/{base_name}.pth"
    if not os.path.exists(base_path):
        return base_path
    
    # Find next available number
    counter = 1
    while True:
        new_path = f"{weights_dir}/{base_name}_{counter}.pth"
        if not os.path.exists(new_path):
            return new_path
        counter += 1


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================

def cleanup_folder(folder_path):
    """Remove the folder to clean up previous results."""
    if os.path.exists(folder_path):
        print(f"Removing existing folder: {folder_path}")
        try:
            shutil.rmtree(folder_path)
            print("Folder removed successfully.")
        except Exception as e:
            print(f"Warning: Could not remove folder: {e}")
    else:
        print(f"No existing {folder_path} folder found.")


def print_step_header(step_num, step_name):
    """Print a formatted header for each pipeline step."""
    print("\n" + "="*80)
    print(f"STEP {step_num}: {step_name}")
    print("="*80)


def print_step_footer(step_name):
    """Print a formatted footer for each pipeline step."""
    print("-"*80)
    print(f"{step_name} completed successfully!")
    print("-"*80)


def run_filter_coco(target_class_index):
    """
    Run the COCO dataset filtering script for a single class.
    
    Args:
        target_class_index: Index of class to filter (0 or 1)
    """
    print_step_header(1, f"FILTERING COCO DATASET - CLASS "
                      f"{CLASSES_TO_KEEP[target_class_index]}")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Target class: {CLASSES_TO_KEEP[target_class_index]}")
    print(f"Class index: {target_class_index}")
    
    try:
        # Run filter_coco.py with single class filtering
        cmd = [
            sys.executable, "src/utils/filter_coco.py", DATASET_PATH,
            "--classes_to_keep", json.dumps(CLASSES_TO_KEEP),
            "--target_class_index", str(target_class_index)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        print_step_footer("COCO Dataset Filtering")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: COCO filtering failed with return code "
              f"{e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: src/utils/filter_coco.py not found in current "
              "directory")
        sys.exit(1)


def run_training_sam2(model_path, target_class_index):
    """
    Run the SAM2 training script.
    
    Args:
        model_path: Full path where the model will be saved
        target_class_index: Index of class being trained (0 or 1)
    """
    print_step_header(2, f"TRAINING SAM2 MODEL - CLASS "
                      f"{CLASSES_TO_KEEP[target_class_index]}")
    print(f"Model will be saved to: {model_path}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Target class: {CLASSES_TO_KEEP[target_class_index]}")
    
    try:
        # Build command with Hugging Face model ID (no checkpoint files)
        cmd = [
            sys.executable, "src/utils/train_sam2.py",
            "--dataset_path", DATASET_PATH,
            "--model_path", model_path,
            "--sam2_model_id", SAM2_MODEL_ID,  # Changed from checkpoint/config
            "--num_epochs", str(NUM_EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--learning_rate", str(LEARNING_RATE),
            "--img_size", str(IMG_SIZE),
            "--report_output_path", REPORT_OUTPUT_PATH,
            "--class_names", json.dumps(CLASS_NAMES_MAPPING),
            "--target_class_index", str(target_class_index)
        ]
        
        print(f"Running SAM2 training with {len(cmd)} parameters...")
        result = subprocess.run(cmd, check=True)
        print_step_footer("SAM2 Model Training")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed with return code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: src/utils/train_sam2.py not found in current "
              "directory")
        sys.exit(1)


def run_inference_sam2(model_path, target_class_index):
    """
    Run the SAM2 inference script.
    
    Args:
        model_path: Path to the trained SAM2 model
        target_class_index: Index of class for inference (0 or 1)
    """
    print_step_header(3, f"SAM2 INFERENCE - CLASS "
                      f"{CLASSES_TO_KEEP[target_class_index]}")
    print(f"Using model: {model_path}")
    print(f"Target class: {CLASSES_TO_KEEP[target_class_index]}")
    
    try:
        # Build command with Hugging Face model ID (no checkpoint files)
        cmd = [
            sys.executable, "src/utils/infer_sam2.py",
            "--model_path", model_path,
            "--sam2_model_id", SAM2_MODEL_ID,  # Changed from checkpoint/config
            "--input_images_folder", INPUT_IMAGES_FOLDER,
            "--output_labels_folder", OUTPUT_LABELS_FOLDER,
            "--output_images_folder", OUTPUT_IMAGES_FOLDER,
            "--points_per_side", str(POINTS_PER_SIDE),
            "--pred_iou_thresh", str(PRED_IOU_THRESH),
            "--stability_score_thresh", str(STABILITY_SCORE_THRESH),
            "--min_mask_region_area", str(MIN_MASK_REGION_AREA),
            "--target_class_index", str(target_class_index),
            "--class_names", json.dumps(CLASS_NAMES_MAPPING)
        ]
        
        print(f"Running SAM2 inference with {len(cmd)} parameters...")
        result = subprocess.run(cmd, check=True)
        print_step_footer("SAM2 Inference")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Inference failed with return code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: src/utils/infer_sam2.py not found in current "
              "directory")
        sys.exit(1)


def run_adapt2rbf():
    """Run the label combination script."""
    print_step_header(4, "COMBINING INFERENCE WITH ORIGINAL LABELS")
    print(f"Target classes for replacement: "
          f"{TARGET_CLASSES_FOR_REPLACEMENT}")
    
    try:
        # Build command with label combination parameters
        cmd = [
            sys.executable, "src/utils/adapt2rbf.py",
            "--inference_folder", INFERENCE_LABELS_FOLDER,
            "--original_folder", ORIGINAL_YOLO_LABELS_FOLDER,
            "--output_folder", COMBINED_LABELS_OUTPUT_FOLDER,
            "--target_classes", json.dumps(TARGET_CLASSES_FOR_REPLACEMENT),
            "--upload_folder", UPLOAD_FOLDER,
            "--original_img_folder", INPUT_IMAGES_FOLDER,
        ]
        
        print(f"Running label combination with {len(cmd)} parameters...")
        result = subprocess.run(cmd, check=True)
        print_step_footer("Label Combination")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Label combination failed with return code "
              f"{e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: src/utils/adapt2rbf.py not found in current "
              "directory")
        sys.exit(1)


def merge_yolo_labels(labels_dir_class0, labels_dir_class1, output_dir):
    """
    Merge YOLO labels from two class-specific inference runs.
    
    Args:
        labels_dir_class0: Directory with class 0 labels
        labels_dir_class1: Directory with class 1 labels
        output_dir: Directory to save merged labels
    """
    print("\n" + "-"*80)
    print("MERGING LABELS FROM BOTH CLASSES")
    print("-"*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all label files
    files_class0 = set(os.listdir(labels_dir_class0)) if \
        os.path.exists(labels_dir_class0) else set()
    files_class1 = set(os.listdir(labels_dir_class1)) if \
        os.path.exists(labels_dir_class1) else set()
    
    all_files = files_class0.union(files_class1)
    
    for filename in all_files:
        if not filename.endswith('.txt'):
            continue
        
        merged_lines = []
        
        # Read class 0 labels
        path0 = os.path.join(labels_dir_class0, filename)
        if os.path.exists(path0):
            with open(path0, 'r') as f:
                merged_lines.extend(f.readlines())
        
        # Read class 1 labels
        path1 = os.path.join(labels_dir_class1, filename)
        if os.path.exists(path1):
            with open(path1, 'r') as f:
                merged_lines.extend(f.readlines())
        
        # Write merged labels
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            f.writelines(merged_lines)
    
    print(f"Merged {len(all_files)} label files")
    print("-"*80)


def check_prerequisites():
    """Check if all required scripts and directories exist."""
    print("Checking prerequisites...")
    
    required_scripts = [
        "src/utils/filter_coco.py",
        "src/utils/train_sam2.py",
        "src/utils/infer_sam2.py",
        "src/utils/adapt2rbf.py"
    ]
    
    for script in required_scripts:
        if not os.path.exists(script):
            print(f"ERROR: Required script '{script}' not found")
            sys.exit(1)
    
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset directory '{DATASET_PATH}' not found")
        sys.exit(1)
    
    # No need to check for checkpoint files anymore - Hugging Face handles it
    print("Prerequisites check completed.")
    print(f"SAM2 model '{SAM2_MODEL_ID}' will be downloaded automatically "
          f"from Hugging Face on first use.")


def print_pipeline_summary(class_index=None):
    """
    Print a summary of the pipeline configuration.
    
    Args:
        class_index: If specified, show config for single class only
    """
    print("\n" + "="*80)
    print("SAM2 FISH SEGMENTATION PIPELINE CONFIGURATION")
    print("="*80)
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Classes: {CLASSES_TO_KEEP}")
    print(f"Class Names Mapping: {CLASS_NAMES_MAPPING}")
    print(f"SAM2 Model (Hugging Face): {SAM2_MODEL_ID}")
    
    if class_index is not None:
        print(f"\nTarget Class: {CLASSES_TO_KEEP[class_index]} "
              f"(index {class_index})")
        model_path = get_next_model_name(BASE_MODEL_NAMES[class_index])
        print(f"Model Path: {model_path}")
    else:
        print(f"\nRunning for BOTH classes:")
        for idx in [0, 1]:
            model_path = get_next_model_name(BASE_MODEL_NAMES[idx])
            print(f"  Class {idx} ({CLASSES_TO_KEEP[idx]}): {model_path}")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Image Size: {IMG_SIZE}")
    print("="*80)


def run_single_class_pipeline(target_class_index, mode='full'):
    """
    Run the complete pipeline for a single class.
    
    Args:
        target_class_index: Index of class to process (0 or 1)
        mode: 'full' for complete pipeline, 'inference' for inference only
    """
    print("\n" + "#"*80)
    print(f"# PROCESSING CLASS: {CLASSES_TO_KEEP[target_class_index]} "
          f"(INDEX {target_class_index})")
    print("#"*80)
    
    # Get model path for this class
    base_name = BASE_MODEL_NAMES[target_class_index]
    model_path = get_next_model_name(base_name) if mode == 'full' else \
        f"weights/{base_name}.pth"
    
    # Create temporary output directories for this class
    temp_labels_dir = os.path.join(INFERENCE_FOLDER_PATH,
                                    f"labels_class{target_class_index}")
    temp_images_dir = os.path.join(INFERENCE_FOLDER_PATH,
                                    f"images_class{target_class_index}")
    
    # Update global variables temporarily
    global OUTPUT_LABELS_FOLDER, OUTPUT_IMAGES_FOLDER
    original_labels_folder = OUTPUT_LABELS_FOLDER
    original_images_folder = OUTPUT_IMAGES_FOLDER
    OUTPUT_LABELS_FOLDER = temp_labels_dir
    OUTPUT_IMAGES_FOLDER = temp_images_dir
    
    try:
        if mode == 'full':
            # Run complete pipeline for this class
            run_filter_coco(target_class_index)
            run_training_sam2(model_path, target_class_index)
        
        # Run inference (both 'full' and 'inference' modes)
        run_inference_sam2(model_path, target_class_index)
        
    finally:
        # Restore global variables
        OUTPUT_LABELS_FOLDER = original_labels_folder
        OUTPUT_IMAGES_FOLDER = original_images_folder
    
    return temp_labels_dir, temp_images_dir


def main():
    """Main pipeline execution function."""
    pipeline_start_time = datetime.now()
    
    print("\n" + "="*80)
    print("SAM2 FISH SEGMENTATION COMPLETE PIPELINE")
    print("="*80)
    print(f"Started at: {pipeline_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Execution mode: {args.mode}")
    
    check_prerequisites()
    
    try:
        if args.mode == 'single_class':
            # Single class mode
            if args.target_class is None:
                print("ERROR: --target_class required for single_class mode")
                sys.exit(1)
            
            print_pipeline_summary(class_index=args.target_class)
            run_single_class_pipeline(args.target_class, mode='full')
            
            # No merging needed for single class
            print("\nSingle class pipeline completed. Skipping merge step.")
            
        elif args.mode in ['full', 'inference']:
            # Process both classes
            print_pipeline_summary()
            
            if args.mode == 'full':
                print("\nRunning FULL pipeline for BOTH classes...")
            else:
                print("\nRunning INFERENCE for BOTH classes...")
            
            # Clean up previous results
            cleanup_folder(INFERENCE_FOLDER_PATH)
            cleanup_folder(UPLOAD_FOLDER)
            
            # Process class 0 (Chromis chromis)
            labels_dir_0, images_dir_0 = run_single_class_pipeline(
                0, mode=args.mode
            )
            
            # Process class 1 (Coris julis)
            labels_dir_1, images_dir_1 = run_single_class_pipeline(
                1, mode=args.mode
            )
            
            # Merge labels from both classes
            merge_yolo_labels(labels_dir_0, labels_dir_1,
                              OUTPUT_LABELS_FOLDER)
            
            # Merge visualizations (copy both to main output folder)
            print("\nCopying visualizations...")
            os.makedirs(OUTPUT_IMAGES_FOLDER, exist_ok=True)
            for img_dir in [images_dir_0, images_dir_1]:
                if os.path.exists(img_dir):
                    for filename in os.listdir(img_dir):
                        src = os.path.join(img_dir, filename)
                        dst = os.path.join(OUTPUT_IMAGES_FOLDER, filename)
                        shutil.copy2(src, dst)
            
            # Run adapt2rbf to combine with original labels
            run_adapt2rbf()
        
        pipeline_end_time = datetime.now()
        duration = pipeline_end_time - pipeline_start_time
        
        print("\n" + "="*80)
        print("!!! PIPELINE FINISHED SUCCESSFULLY !!!")
        print("="*80)
        print(f"Total duration: {str(duration).split('.')[0]}")
        print(f"Combined labels: {OUTPUT_LABELS_FOLDER}")
        print(f"Upload folder: {UPLOAD_FOLDER}")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
