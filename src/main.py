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

# =============================================================================
# IMPORTS
# =============================================================================

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

# Using SAM 2.1
SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"

# Base model names for each class
BASE_MODEL_NAMES = {
    0: "sam2_chromis",  # Chromis chromis
    1: "sam2_coris"     # Coris julis
}

# =============================================================================
# NEW: HYBRID PIPELINE CONFIGURATION
# =============================================================================

# Master switch: Enable hybrid mode (Mask R-CNN + SAM2)
USE_HYBRID_MODE = True  # Set to True to enable hybrid pipeline

# Mask R-CNN model path (your existing trained model)
MASKRCNN_MODEL_PATH = "weights/m09_1.pth"

# Unified SAM2 model name (trained on both classes)
UNIFIED_SAM2_MODEL = "sam2_fish_unified.pth"

# Training parameters (simpler than Mask R-CNN)
NUM_CLASSES = 1  # Binary segmentation per model
BATCH_SIZE = 1
NUM_EPOCHS = 300
LEARNING_RATE = 5e-6
IMG_SIZE = 1024  # SAM2 default input size
GRADIENT_ACCUMULATION_STEPS = 4
DICE_WEIGHT = 2.0
MAX_INSTANCES_PER_IMAGE = 100

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
    """Run the SAM2 training script."""
    print_step_header(2, f"TRAINING SAM2 MODEL - CLASS "
                      f"{CLASSES_TO_KEEP[target_class_index]}")
    print(f"Model will be saved to: {model_path}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Target class: {CLASSES_TO_KEEP[target_class_index]}")
    
    try:
        # Build command with Hugging Face model ID
        cmd = [
            sys.executable, "src/utils/train_sam2.py",
            "--dataset_path", DATASET_PATH,
            "--model_path", model_path,
            "--sam2_model_id", SAM2_MODEL_ID,
            "--num_epochs", str(NUM_EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--learning_rate", str(LEARNING_RATE),
            "--img_size", str(IMG_SIZE),
            "--report_output_path", REPORT_OUTPUT_PATH,
            "--class_names", json.dumps(CLASS_NAMES_MAPPING),
            "--target_class_index", str(target_class_index),
            "--gradient_accumulation_steps", str(GRADIENT_ACCUMULATION_STEPS),
            "--dice_weight", str(DICE_WEIGHT),
            "--max_instances_per_image", str(MAX_INSTANCES_PER_IMAGE)
        ]
        
        print(f"Running SAM2 training with {len(cmd)} parameters...")
        result = subprocess.run(cmd, check=True)
        print_step_footer("SAM2 Model Training")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed with return code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: src/utils/train_sam2.py not found in current directory")
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
        # Build command with Hugging Face model ID
        cmd = [
            sys.executable, "src/utils/infer_sam2.py",
            "--model_path", model_path,
            "--sam2_model_id", SAM2_MODEL_ID,
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
    
    # ADD THESE DEBUG COUNTERS:
    total_class0_instances = 0
    total_class1_instances = 0
    
    for filename in all_files:
        if not filename.endswith('.txt'):
            continue
        
        merged_lines = []
        
        # Read class 0 labels
        path0 = os.path.join(labels_dir_class0, filename)
        if os.path.exists(path0):
            with open(path0, 'r') as f:
                lines = f.readlines()
                merged_lines.extend(lines)
                total_class0_instances += len(lines)
        
        # Read class 1 labels
        path1 = os.path.join(labels_dir_class1, filename)
        if os.path.exists(path1):
            with open(path1, 'r') as f:
                lines = f.readlines()
                merged_lines.extend(lines)
                total_class1_instances += len(lines)
        
        # Write merged labels
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            f.writelines(merged_lines)
    
    print(f"Merged {len(all_files)} label files")
    print(f"Total class 0 instances: {total_class0_instances}")  # NEW
    print(f"Total class 1 instances: {total_class1_instances}")  # NEW
    print(f"Total combined instances: {total_class0_instances + total_class1_instances}")  # NEW
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

def run_hybrid_inference():
    """
    Run hybrid Mask R-CNN + SAM2 inference.
    Combines class predictions from Mask R-CNN with refined masks from SAM2.
    """
    print_step_header("HYBRID", "MASK R-CNN + SAM2 INFERENCE")
    
    print(f"Mask R-CNN model: {MASKRCNN_MODEL_PATH}")
    print(f"SAM2 model: {UNIFIED_SAM2_MODEL}")
    
    try:
        cmd = [
            sys.executable, "src/utils/infer_hybrid.py",
            "--maskrcnn_model", MASKRCNN_MODEL_PATH,
            "--sam2_model", os.path.join("weights", UNIFIED_SAM2_MODEL),
            "--sam2_model_id", SAM2_MODEL_ID,
            "--input_folder", os.path.join(DATASET_PATH, "test"),
            "--output_labels", os.path.join(DATASET_PATH, "inference",
                                             "labels_hybrid"),
            "--output_images", os.path.join(DATASET_PATH, "inference",
                                             "images_hybrid"),
            "--detection_threshold", "0.5"
        ]
        
        print(f"Running hybrid inference with {len(cmd)} parameters...")
        result = subprocess.run(cmd, check=True)
        print_step_footer("Hybrid Inference")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Hybrid inference failed with return code "
              f"{e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: src/utils/infer_hybrid.py not found")
        sys.exit(1)


def train_unified_sam2():
    """
    Train a single unified SAM2 model on ALL fish classes.
    This model will be used in hybrid pipeline.
    """
    print_step_header("UNIFIED", "TRAINING UNIFIED SAM2 MODEL")
    print("Training on ALL classes for hybrid pipeline")
    
    model_path = os.path.join("weights", UNIFIED_SAM2_MODEL)
    
    try:
        cmd = [
            sys.executable, "src/utils/train_sam2.py",
            "--dataset_path", DATASET_PATH,
            "--model_path", model_path,
            "--sam2_model_id", SAM2_MODEL_ID,
            "--num_epochs", str(NUM_EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--learning_rate", str(LEARNING_RATE),
            "--img_size", str(IMG_SIZE),
            "--report_output_path", REPORT_OUTPUT_PATH,
            "--class_names", json.dumps(CLASS_NAMES_MAPPING),
            "--target_class_index", "0",  # Doesn't matter for unified
            "--gradient_accumulation_steps",
                str(GRADIENT_ACCUMULATION_STEPS),
            "--dice_weight", str(DICE_WEIGHT),
            "--max_instances_per_image", str(MAX_INSTANCES_PER_IMAGE),
            "--train_unified"  # NEW: Enable unified training
        ]
        
        print(f"Running unified SAM2 training with {len(cmd)} parameters...")
        result = subprocess.run(cmd, check=True)
        print_step_footer("Unified SAM2 Training")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Unified training failed with return code "
              f"{e.returncode}")
        sys.exit(1)


def main():
    """Main pipeline execution function."""
    pipeline_start_time = datetime.now()
    
    print("="*80)
    print("SAM2 FISH SEGMENTATION COMPLETE PIPELINE")
    print("="*80)
    print(f"Started at: {pipeline_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Execution mode: {args.mode}")
    
    # Check if hybrid mode is enabled
    if USE_HYBRID_MODE:
        print("Pipeline mode: HYBRID (Mask R-CNN + SAM2)")
    else:
        print("Pipeline mode: STANDARD (SAM2 only)")
    
    check_prerequisites()
    
    try:
        # Hybrid mode pipeline
        if USE_HYBRID_MODE and args.mode == 'full':
            print("\nRunning HYBRID pipeline...")
            
            # Step 1: Train unified SAM2 model
            # train_unified_sam2()
            
            # Step 2: Run hybrid inference
            run_hybrid_inference()
            
            print("\n" + "="*80)
            print("!!! HYBRID PIPELINE FINISHED SUCCESSFULLY !!!")
            print("="*80)
            
        # Standard per-class pipeline
        elif args.mode == 'singleclass':
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
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    # --mode single_class --target_class 0