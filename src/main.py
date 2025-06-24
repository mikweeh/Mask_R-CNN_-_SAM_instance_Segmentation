#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main script that runs the complete fish segmentation pipeline:
1. Filter COCO dataset (filter_coco.py)
2. Train Mask R-CNN model (train.py)
3. Convert predictions to YOLOv11 format (coco2yolo11.py)
4. Combine inference labels with original YOLO labels (adapt2rbf.py)
"""

import subprocess
import sys
import os
import json
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

# Mapping from COCO class IDs to YOLO class IDs (0-indexed)
COCO_TO_YOLO_CLASS_MAPPING = {1: 0, 2: 1}

# YOLO class IDs that will be replaced by inference results (for adapt2rbf.py)
TARGET_CLASSES_FOR_REPLACEMENT = [0, 1]

# =============================================================================
# REST OF CONFIGURATION
# =============================================================================

# Model configuration
MODEL_NAME = "m01"
NUM_CLASSES = len(CLASSES_TO_KEEP) + 1  # +1 for background class

# Training parameters
BATCH_SIZE = 1
NUM_EPOCHS = 2
LEARNING_RATE = 0.0001
IMG_SIZE = 2048
CONFIDENCE_THRESHOLD = 0.3
DICE_WEIGHT = 0.0
MASK_RESOLUTION = 56
BASE_MIN_ANCHOR = 16

# Training options
USE_FOCAL_DICE = False
OVERSAMPLE_SMALL_OBJECTS = True
USE_COPY_PASTE = True

# RPN parameters
RPN_PRE_NMS_TOP_N_TRAIN = 1500
RPN_POST_NMS_TOP_N_TRAIN = 600
RPN_NMS_THRESH = 0.6

# Output configuration
REPORT_OUTPUT_PATH = "results"

# Inference configuration (for coco2yolo11.py)
MODEL_WEIGHTS_PATH = f"weights/{MODEL_NAME}.pth"
INPUT_IMAGES_FOLDER = "dataset/test"
OUTPUT_LABELS_FOLDER = "dataset/inference/labels"
OUTPUT_IMAGES_FOLDER = "dataset/inference/images"

# Label combination configuration (for adapt2rbf.py)
INFERENCE_LABELS_FOLDER = "dataset/inference/labels"
ORIGINAL_YOLO_LABELS_FOLDER = "dataset/original_yolo/train/labels"
COMBINED_LABELS_OUTPUT_FOLDER = "dataset/inference/labels_full"

# Generate class names mapping for YOLO (derived from above)
CLASS_NAMES_MAPPING = {}
for coco_id, yolo_id in COCO_TO_YOLO_CLASS_MAPPING.items():
    if coco_id <= len(CLASSES_TO_KEEP):
        CLASS_NAMES_MAPPING[yolo_id] = CLASSES_TO_KEEP[coco_id - 1]

# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================

def print_step_header(step_num, step_name):
    """Print a formatted header for each pipeline step"""
    print("\n" + "="*80)
    print(f"STEP {step_num}: {step_name}")
    print("="*80)

def print_step_footer(step_name):
    """Print a formatted footer for each pipeline step"""
    print("-"*80)
    print(f"{step_name} completed successfully!")
    print("-"*80)

def run_filter_coco():
    """
    Run the COCO dataset filtering script.
    """
    print_step_header(1, "FILTERING COCO DATASET")
    
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Classes to keep: {CLASSES_TO_KEEP}")
    
    try:
        # Run filter_coco.py with dataset path argument
        cmd = [sys.executable, "src/utils/filter_coco.py", DATASET_PATH]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        
        print_step_footer("COCO Dataset Filtering")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: COCO filtering failed with return code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: src/utils/filter_coco.py not found in current directory")
        sys.exit(1)

def run_training():
    """
    Run the training script with all parameters including class configuration.
    """
    print_step_header(2, "TRAINING MASK R-CNN MODEL")
    
    print(f"Model name: {MODEL_NAME}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"Classes mapping: {COCO_TO_YOLO_CLASS_MAPPING}")
    print(f"Class names: {CLASS_NAMES_MAPPING}")
    
    try:
        # Build command with all training parameters including class configuration
        cmd = [
            sys.executable, "src/utils/train.py",
            "--dataset_path", DATASET_PATH,
            "--model_name", MODEL_NAME,
            "--num_classes", str(NUM_CLASSES),
            "--batch_size", str(BATCH_SIZE),
            "--num_epochs", str(NUM_EPOCHS),
            "--learning_rate", str(LEARNING_RATE),
            "--img_size", str(IMG_SIZE),
            "--confidence_threshold", str(CONFIDENCE_THRESHOLD),
            "--dice_weight", str(DICE_WEIGHT),
            "--mask_resolution", str(MASK_RESOLUTION),
            "--base_min_anchor", str(BASE_MIN_ANCHOR),
            "--rpn_pre_nms_top_n_train", str(RPN_PRE_NMS_TOP_N_TRAIN),
            "--rpn_post_nms_top_n_train", str(RPN_POST_NMS_TOP_N_TRAIN),
            "--rpn_nms_thresh", str(RPN_NMS_THRESH),
            "--report_output_path", REPORT_OUTPUT_PATH,
            # Pass class configuration from main.py
            "--class_mapping", json.dumps(COCO_TO_YOLO_CLASS_MAPPING),
            "--class_names", json.dumps(CLASS_NAMES_MAPPING)
        ]
        
        # Add boolean flags
        if USE_FOCAL_DICE:
            cmd.append("--use_focal_dice")
        if OVERSAMPLE_SMALL_OBJECTS:
            cmd.append("--oversample_small_objects")
        if USE_COPY_PASTE:
            cmd.append("--use_copy_paste")
        
        print(f"Running training with {len(cmd)} parameters...")
        result = subprocess.run(cmd, check=True)
        
        print_step_footer("Model Training")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed with return code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: src/utils/train.py not found in current directory")
        sys.exit(1)

def run_coco2yolo():
    """
    Run the COCO to YOLO conversion script with class configuration.
    """
    print_step_header(3, "CONVERTING TO YOLOV11 FORMAT")
    
    print(f"Model weights path: {MODEL_WEIGHTS_PATH}")
    print(f"Class mapping: {COCO_TO_YOLO_CLASS_MAPPING}")
    
    try:
        # Build command with all inference parameters including class configuration
        cmd = [
            sys.executable, "src/utils/coco2yolo11.py",
            "--model_path", MODEL_WEIGHTS_PATH,
            "--input_images_folder", INPUT_IMAGES_FOLDER,
            "--output_labels_folder", OUTPUT_LABELS_FOLDER,
            "--output_images_folder", OUTPUT_IMAGES_FOLDER,
            "--num_classes", str(NUM_CLASSES),
            "--img_size", str(IMG_SIZE),
            "--confidence_threshold", str(CONFIDENCE_THRESHOLD),
            "--mask_resolution", str(MASK_RESOLUTION),
            "--base_min_anchor", str(BASE_MIN_ANCHOR),
            # Pass class configuration from main.py
            "--class_mapping", json.dumps(COCO_TO_YOLO_CLASS_MAPPING)
        ]
        
        print(f"Running COCO to YOLO conversion with {len(cmd)} parameters...")
        result = subprocess.run(cmd, check=True)
        
        print_step_footer("COCO to YOLOv11 Conversion")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: COCO to YOLO conversion failed with return code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: src/utils/coco2yolo11.py not found in current directory")
        sys.exit(1)

def run_adapt2rbf():
    """
    Run the label combination script with class configuration.
    """
    print_step_header(4, "COMBINING INFERENCE WITH ORIGINAL LABELS")
    
    print(f"Target classes for replacement: {TARGET_CLASSES_FOR_REPLACEMENT}")
    
    try:
        # Build command with label combination parameters including class configuration
        cmd = [
            sys.executable, "src/utils/adapt2rbf.py",
            "--inference_folder", INFERENCE_LABELS_FOLDER,
            "--original_folder", ORIGINAL_YOLO_LABELS_FOLDER,
            "--output_folder", COMBINED_LABELS_OUTPUT_FOLDER,
            # Pass class configuration from main.py
            "--target_classes", json.dumps(TARGET_CLASSES_FOR_REPLACEMENT)
        ]
        
        print(f"Running label combination with {len(cmd)} parameters...")
        result = subprocess.run(cmd, check=True)
        
        print_step_footer("Label Combination")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Label combination failed with return code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: src/utils/adapt2rbf.py not found in current directory")
        sys.exit(1)

def check_prerequisites():
    """Check if all required scripts and directories exist."""
    print("Checking prerequisites...")
    
    required_scripts = [
        "src/utils/filter_coco.py", 
        "src/utils/train.py", 
        "src/utils/coco2yolo11.py",
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

def print_pipeline_summary():
    """Print a summary of the pipeline configuration."""
    print("\n" + "="*80)
    print("FISH SEGMENTATION PIPELINE CONFIGURATION")
    print("="*80)
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Classes to Keep: {CLASSES_TO_KEEP}")
    print(f"COCO to YOLO Class Mapping: {COCO_TO_YOLO_CLASS_MAPPING}")
    print(f"Class Names Mapping: {CLASS_NAMES_MAPPING}")
    print(f"Target Classes for Replacement: {TARGET_CLASSES_FOR_REPLACEMENT}")
    print(f"Model Name: {MODEL_NAME}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print("="*80)

def main():
    """Main pipeline execution function."""
    pipeline_start_time = datetime.now()
    
    print("FISH SEGMENTATION COMPLETE PIPELINE")
    print(f"Started at: {pipeline_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print_pipeline_summary()
    check_prerequisites()
    
    try:
        run_filter_coco()
        run_training()
        run_coco2yolo()
        run_adapt2rbf()
        
        pipeline_end_time = datetime.now()
        duration = pipeline_end_time - pipeline_start_time
        
        print("\n" + "="*80)
        print("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY! 🎉")
        print("="*80)
        print(f"Total duration: {str(duration).split('.')[0]}")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error in pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
