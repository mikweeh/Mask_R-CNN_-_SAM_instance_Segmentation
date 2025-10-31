#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fish Segmentation Pipeline - v3.0.0 (COMPLETE)

Complete workflow:
0. Setup dataset structure (filter_coco.py)
1. Train Mask R-CNN (detection + classification)
2. Train unified SAM2 (segmentation refinement)
3. Run hybrid inference (Mask R-CNN + SAM2)
4. Create upload folder with data.yaml (adapt2rbf.py)

Author: Fish Segmentation Project
Version: 3.0.0
"""

"""
USAGE GUIDE:
============

# Full pipeline (setup + train all models + inference + upload folder)
python src/main.py --mode full

# Individual steps:
# -----------------

# Step 0: Setup dataset structure
python src/main.py --mode setup

# Step 1: Train Mask R-CNN (detection + classification)
python src/main.py --mode train-maskrcnn

# Step 2: Train all SAM2 models (one per class)
python src/main.py --mode train-sam2

# Step 2 (alternative): Train SAM2 for specific class only
python src/main.py --mode train-sam2-class --class_index 0  # Chromis chromis
python src/main.py --mode train-sam2-class --class_index 1  # Coris julis

# Step 3+4: Inference + create upload folder (requires all models trained)
python src/main.py --mode inference

NOTES:
------
- Each SAM2 training generates a PDF report in results/
- Inference processes all classes sequentially (saves GPU memory)
- Upload folder (dataset/upload/) contains: images + labels + data.yaml
- All models use IMG_SIZE=2048 (configured in CONFIGURATION section)

REQUIREMENTS:
-------------
- Mask R-CNN model required before training SAM2 models
- All SAM2 models required before inference
- Dataset structure: dataset/original_coco/ and dataset/original_yolo/
"""


import argparse
import subprocess
import sys
import os
import json
import shutil
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset
DATASET_PATH = 'dataset'
CLASSES_TO_KEEP = ["Chromis chromis", "Coris julis"]
TARGET_CLASSES_FOR_REPLACEMENT = [0, 1]

# Models
SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"
MASKRCNN_MODEL_PATH = "weights/mask_rcnn_fish_model_2048.pth"

# SAM2 models - one per class (scalable)
SAM2_MODEL_PATHS = {
    0: "weights/sam2_chromis_2048.pth",
    1: "weights/sam2_coris_2048.pth"
}

# Training parameters
NUM_EPOCHS = 100
BATCH_SIZE = 1
LEARNING_RATE = 5e-6

# Input image size
MASKRCNN_IMG_SIZE = 2048
SAM2_IMG_SIZE = 2048

# Mask R-CNN mask resolution
MASK_RESOLUTION = 112  # Can be 28, 56, or 112

GRADIENT_ACCUMULATION_STEPS = 2
DICE_WEIGHT = 2.0
MAX_INSTANCES_PER_IMAGE = 100

# Output
REPORT_OUTPUT_PATH = 'results'
INFERENCE_FOLDER_PATH = os.path.join(DATASET_PATH, 'inference')
OUTPUT_LABELS_FOLDER = os.path.join(INFERENCE_FOLDER_PATH, 'labels')
OUTPUT_IMAGES_FOLDER = os.path.join(INFERENCE_FOLDER_PATH, 'images')
UPLOAD_FOLDER = os.path.join(DATASET_PATH, 'upload')
ORIGINAL_YOLO_DATA_YAML = os.path.join(
    DATASET_PATH, 'original_yolo/data.yaml'
)

# Mask R-CNN Inference Output (STANDALONE)
INFERENCE_MASKRCNN_FOLDER = os.path.join(DATASET_PATH, 'inference_maskrcnn')
OUTPUT_MASKRCNN_LABELS = os.path.join(INFERENCE_MASKRCNN_FOLDER, 'labels')
OUTPUT_MASKRCNN_IMAGES = os.path.join(INFERENCE_MASKRCNN_FOLDER, 'images')

# Class mapping
CLASS_NAMES_MAPPING = {i: name for i, name in enumerate(CLASSES_TO_KEEP)}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(text)
    print("="*80)

def check_prerequisites():
    """Check if required directories exist."""
    print("Checking prerequisites...")
    
    # Need BOTH COCO and YOLO formats
    required_dirs = [
        # COCO format (for training)
        os.path.join(DATASET_PATH, 'original_coco/train'),
        os.path.join(DATASET_PATH, 'original_coco/valid'),
        os.path.join(DATASET_PATH, 'original_coco/test'),
        # YOLO format (for combining labels)
        os.path.join(DATASET_PATH, 'original_yolo/test/labels'),
    ]
    
    # Check data.yaml exists
    if not os.path.exists(ORIGINAL_YOLO_DATA_YAML):
        print(f"WARNING: data.yaml not found at {ORIGINAL_YOLO_DATA_YAML}")
        print("You'll need this for uploading to Roboflow")
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"ERROR: Required directory not found: {dir_path}")
            print("\nYou need BOTH:")
            print("  - dataset/original_coco/ (for training)")
            print("  - dataset/original_yolo/ (for label combining)")
            sys.exit(1)
    
    print("✓ Prerequisites check passed")
    print("  ✓ COCO format dataset (for training)")
    print("  ✓ YOLO format dataset (for label combining)")


# =============================================================================
# STEP 0: DATASET SETUP (filter_coco.py)
# =============================================================================

def setup_dataset():
    """
    Setup dataset structure using filter_coco.py.
    This prepares train/valid/test folders with proper structure.
    """
    print_header("STEP 0: DATASET SETUP")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Classes: {CLASSES_TO_KEEP}")
    print("Setting up train/valid/test folders...")
    
    try:
        cmd = [
            sys.executable, "src/utils/filter_coco.py",
            DATASET_PATH,
            "--classes_to_keep", json.dumps(CLASSES_TO_KEEP),
            "--target_class_index", "0"  # Just to setup, not filtering
        ]
        
        print(f"\nRunning dataset setup...")
        result = subprocess.run(cmd, check=True)
        print("✓ Dataset setup completed")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Dataset setup failed (code {e.returncode})")
        sys.exit(1)
    except FileNotFoundError:
        print("✗ ERROR: src/utils/filter_coco.py not found")
        sys.exit(1)


# =============================================================================
# STEP 1: TRAIN MASK R-CNN
# =============================================================================

def train_maskrcnn():
    """Train Mask R-CNN for detection and classification."""
    print_header("STEP 1: TRAINING MASK R-CNN")
    print(f"Model: {MASKRCNN_MODEL_PATH}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"IMG_SIZE: {MASKRCNN_IMG_SIZE}")
    print(f"MASK_RESOLUTION: {MASK_RESOLUTION}")
    print(f"Classes: {CLASSES_TO_KEEP}")
    
    try:
        cmd = [
            sys.executable, "src/utils/train.py",
            "--dataset_path", DATASET_PATH,
            "--model_path", MASKRCNN_MODEL_PATH,
            "--num_epochs", str(NUM_EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--learning_rate", str(LEARNING_RATE),
            "--img_size", str(MASKRCNN_IMG_SIZE),
            "--mask_resolution", str(MASK_RESOLUTION)
        ]
        
        print(f"\nRunning Mask R-CNN training...")
        result = subprocess.run(cmd, check=True)
        print("✓ Mask R-CNN training completed")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Training failed (code {e.returncode})")
        sys.exit(1)
    except FileNotFoundError:
        print("✗ ERROR: src/utils/train.py not found")
        sys.exit(1)


# =============================================================================
# STEP 1.5: MASK R-CNN STANDALONE INFERENCE
# =============================================================================

def run_maskrcnn_inference():
    """Run standalone Mask R-CNN inference (before SAM2 training)."""
    print_header("STEP 1.5: MASK R-CNN STANDALONE INFERENCE")
    print(f"Model: {MASKRCNN_MODEL_PATH}")
    print(f"IMG_SIZE: {MASKRCNN_IMG_SIZE}")
    print(f"MASK_RESOLUTION: {MASK_RESOLUTION}")
    print(f"Output: {OUTPUT_MASKRCNN_LABELS}")
    
    # Clean inference folders
    if os.path.exists(INFERENCE_MASKRCNN_FOLDER):
        print(f"\nCleaning existing Mask R-CNN inference folder...")
        shutil.rmtree(INFERENCE_MASKRCNN_FOLDER)
    
    try:
        cmd = [
            sys.executable, "src/utils/infer_maskrcnn.py",
            "--maskrcnn_model", MASKRCNN_MODEL_PATH,
            "--input_folder", os.path.join(DATASET_PATH, "test"),
            "--output_labels", OUTPUT_MASKRCNN_LABELS,
            "--output_images", OUTPUT_MASKRCNN_IMAGES,
            "--detection_threshold", "0.5",
            "--maskrcnn_img_size", str(MASKRCNN_IMG_SIZE),
            "--mask_resolution", str(MASK_RESOLUTION),
            "--class_names", json.dumps(CLASS_NAMES_MAPPING)
        ]
        
        print(f"\nRunning Mask R-CNN inference...")
        result = subprocess.run(cmd, check=True)
        print("✓ Mask R-CNN inference completed")
        print(f"  Labels saved to: {OUTPUT_MASKRCNN_LABELS}")
        print(f"  Images saved to: {OUTPUT_MASKRCNN_IMAGES}")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Mask R-CNN inference failed (code {e.returncode})")
        sys.exit(1)
    except FileNotFoundError:
        print("✗ ERROR: src/utils/infer_maskrcnn.py not found")
        sys.exit(1)


# =============================================================================
# STEP 2: TRAIN SAM2 MODELS (ONE PER CLASS)
# =============================================================================

def train_sam2_per_class(class_index):
    """
    Train a SAM2 model for a specific class.
    
    Args:
        class_index: Index of the class to train (0, 1, 2, ...)
    
    Returns:
        None
    
    Raises:
        SystemExit: If training fails
    """
    if class_index not in CLASS_NAMES_MAPPING:
        print(f"✗ ERROR: Invalid class_index {class_index}")
        sys.exit(1)
    
    class_name = CLASS_NAMES_MAPPING[class_index]
    model_path = SAM2_MODEL_PATHS[class_index]
    
    print_header(
        f"STEP 2.{class_index + 1}: TRAINING SAM2 FOR {class_name.upper()}"
    )
    print(f"Model: {model_path}")
    print(f"Training on: {class_name} (class {class_index})")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"IMG_SIZE: {SAM2_IMG_SIZE}")
    
    try:
        cmd = [
            sys.executable, "src/utils/train_sam2.py",
            "--dataset_path", DATASET_PATH,
            "--model_path", model_path,
            "--sam2_model_id", SAM2_MODEL_ID,
            "--num_epochs", str(NUM_EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--learning_rate", str(LEARNING_RATE),
            "--img_size", str(SAM2_IMG_SIZE),
            "--report_output_path", REPORT_OUTPUT_PATH,
            "--class_names", json.dumps(CLASS_NAMES_MAPPING),
            "--target_class_index", str(class_index),
            "--gradient_accumulation_steps",
            str(GRADIENT_ACCUMULATION_STEPS),
            "--dice_weight", str(DICE_WEIGHT),
            "--max_instances_per_image", str(MAX_INSTANCES_PER_IMAGE)
        ]
        
        print(f"\nRunning SAM2 training for {class_name}...")
        result = subprocess.run(cmd, check=True)
        print(f"✓ SAM2 training completed for {class_name}")
        print(f"  Model saved to: {model_path}")
        print(f"  Report saved to: {REPORT_OUTPUT_PATH}/")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Training failed (code {e.returncode})")
        sys.exit(1)


def train_all_sam2_models():
    """Train SAM2 models for all classes sequentially."""
    print_header("STEP 2: TRAINING SAM2 MODELS (ONE PER CLASS)")
    print(f"Training {len(CLASS_NAMES_MAPPING)} SAM2 models:")
    for class_idx, class_name in CLASS_NAMES_MAPPING.items():
        print(f"  - Class {class_idx}: {class_name}")
    print("="*80)
    
    for class_idx in sorted(CLASS_NAMES_MAPPING.keys()):
        train_sam2_per_class(class_idx)
    
    print_header("ALL SAM2 MODELS TRAINED SUCCESSFULLY ✓")



# =============================================================================
# STEP 3: HYBRID INFERENCE (SEQUENTIAL BY CLASS)
# =============================================================================

def run_inference():
    """
    Run hybrid Mask R-CNN + SAM2 inference sequentially by class.
    """
    print_header("STEP 3: HYBRID INFERENCE (Mask R-CNN + Multi-SAM2)")
    print(f"Mask R-CNN: {MASKRCNN_MODEL_PATH} "
          f"(IMG_SIZE={MASKRCNN_IMG_SIZE})")
    
    for class_idx, model_path in SAM2_MODEL_PATHS.items():
        class_name = CLASS_NAMES_MAPPING[class_idx]
        print(f"SAM2 [{class_name}]: {model_path}")
    
    print(f"Output: {OUTPUT_LABELS_FOLDER}")
    print("\nStrategy: Sequential processing by class")
    print("  1. Mask R-CNN detects all bboxes")
    for idx, class_name in CLASS_NAMES_MAPPING.items():
        print(f"  {idx + 2}. SAM2_{class_name} processes all {class_name}")
    
    # Clean inference folders
    if os.path.exists(INFERENCE_FOLDER_PATH):
        print(f"\nCleaning existing inference folder...")
        shutil.rmtree(INFERENCE_FOLDER_PATH)
    
    try:
        # Build command with all SAM2 model paths
        cmd = [
            sys.executable, "src/utils/infer_hybrid.py",
            "--maskrcnn_model", MASKRCNN_MODEL_PATH,
            "--sam2_model_paths", json.dumps(SAM2_MODEL_PATHS),
            "--sam2_model_id", SAM2_MODEL_ID,
            "--input_folder", os.path.join(DATASET_PATH, "test"),
            "--output_labels", OUTPUT_LABELS_FOLDER,
            "--output_images", OUTPUT_IMAGES_FOLDER,
            "--detection_threshold", "0.5",
            "--maskrcnn_img_size", str(MASKRCNN_IMG_SIZE),
            "--class_names", json.dumps(CLASS_NAMES_MAPPING)
        ]
        
        print(f"\nRunning hybrid inference...")
        result = subprocess.run(cmd, check=True)
        print("✓ Inference completed")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Inference failed (code {e.returncode})")
        sys.exit(1)


# =============================================================================
# STEP 4: CREATE UPLOAD FOLDER (adapt2rbf.py)
# =============================================================================

def create_upload_folder():
    """
    Create upload folder using adapt2rbf.py.
    This combines inference labels with original labels and creates data.yaml.
    """
    print_header("STEP 4: CREATING UPLOAD FOLDER")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print("This will create images + labels + data.yaml ready for Roboflow")
    
    # Clean upload folder
    if os.path.exists(UPLOAD_FOLDER):
        print(f"Cleaning existing upload folder...")
        shutil.rmtree(UPLOAD_FOLDER)
    
    try:
        cmd = [
            sys.executable, "src/utils/adapt2rbf.py",
            "--inference_folder", OUTPUT_LABELS_FOLDER,
            "--original_folder", os.path.join(DATASET_PATH,
                                             "original_yolo/test/labels"),
            "--output_folder", os.path.join(INFERENCE_FOLDER_PATH,
                                            "labels_full"),
            "--target_classes", json.dumps(TARGET_CLASSES_FOR_REPLACEMENT),
            "--upload_folder", UPLOAD_FOLDER,
            "--original_img_folder", os.path.join(DATASET_PATH, "test")
        ]
        
        print(f"\nRunning adapt2rbf...")
        result = subprocess.run(cmd, check=True)
        print("✓ Upload folder created")
        
        # Verify contents
        if os.path.exists(UPLOAD_FOLDER):
            img_count = len([f for f in os.listdir(UPLOAD_FOLDER)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            label_count = len([f for f in os.listdir(UPLOAD_FOLDER)
                              if f.endswith('.txt')])
            yaml_exists = os.path.exists(os.path.join(UPLOAD_FOLDER,
                                                       'data.yaml'))
            
            print(f"\nUpload folder contents:")
            print(f"   - {img_count} images")
            print(f"   - {label_count} label files")
            print(f"   - data.yaml: {'✓' if yaml_exists else '✗'}")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Upload folder creation failed (code {e.returncode})")
        sys.exit(1)
    except FileNotFoundError:
        print("✗ ERROR: src/utils/adapt2rbf.py not found")
        sys.exit(1)


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """Execute the complete pipeline."""
    parser = argparse.ArgumentParser(
        description='Fish Segmentation Pipeline v3.0.0 - Multi-SAM2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (everything from scratch)
  python src/main.py --mode full
  
  # Only setup dataset structure
  python src/main.py --mode setup
  
  # Only train Mask R-CNN (after setup)
  python src/main.py --mode train-maskrcnn
  
  # Train all SAM2 models (requires Mask R-CNN)
  python src/main.py --mode train-sam2
  
  # Train SAM2 for specific class only
  python src/main.py --mode train-sam2-class --class_index 0
  
  # Only inference + upload (requires all models)
  python src/main.py --mode inference
"""
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'setup', 'train-maskrcnn', 'train-sam2',
                 'train-sam2-class', 'inference', 'inference-maskrcnn'],
        default='full',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--class_index',
        type=int,
        default=None,
        help='Class index for train-sam2-class mode'
    )
    
    args = parser.parse_args()
    start_time = datetime.now()
    
    print("="*80)
    print("FISH SEGMENTATION PIPELINE v3.0.0 - MULTI-SAM2")
    print("="*80)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode}")
    print("="*80)
    
    check_prerequisites()
    
    try:
        # Execute based on mode
        if args.mode == 'full':
            # Complete workflow
            setup_dataset()
            train_maskrcnn()
            train_all_sam2_models()
            run_inference()
            create_upload_folder()
            
        elif args.mode == 'setup':
            # Only setup dataset
            setup_dataset()
            
        elif args.mode == 'train-maskrcnn':
            # Only train Mask R-CNN
            train_maskrcnn()
            run_maskrcnn_inference()
            
        elif args.mode == 'train-sam2':
            # Train all SAM2 models (requires Mask R-CNN)
            if not os.path.exists(MASKRCNN_MODEL_PATH):
                print(f"✗ ERROR: Mask R-CNN model not found: "
                      f"{MASKRCNN_MODEL_PATH}")
                print("  Run with --mode train-maskrcnn first")
                sys.exit(1)
            train_all_sam2_models()
            
        elif args.mode == 'train-sam2-class':
            # Train SAM2 for specific class
            if args.class_index is None:
                print("✗ ERROR: --class_index required for "
                      "train-sam2-class mode")
                sys.exit(1)
            
            if not os.path.exists(MASKRCNN_MODEL_PATH):
                print(f"✗ ERROR: Mask R-CNN model not found: "
                      f"{MASKRCNN_MODEL_PATH}")
                sys.exit(1)
            
            train_sam2_per_class(args.class_index)

        elif args.mode == 'inference-maskrcnn':
            # Only Mask R-CNN inference
            if not os.path.exists(MASKRCNN_MODEL_PATH):
                print(f"✗ ERROR: Mask R-CNN model not found: {MASKRCNN_MODEL_PATH}")
                sys.exit(1)
            run_maskrcnn_inference()

        elif args.mode == 'inference':
            # Only inference + upload (requires all models)
            if not os.path.exists(MASKRCNN_MODEL_PATH):
                print(f"✗ ERROR: Mask R-CNN model not found: "
                      f"{MASKRCNN_MODEL_PATH}")
                sys.exit(1)
            
            # Check all SAM2 models exist
            missing_models = []
            for class_idx, model_path in SAM2_MODEL_PATHS.items():
                if not os.path.exists(model_path):
                    missing_models.append(
                        f"Class {class_idx} "
                        f"({CLASS_NAMES_MAPPING[class_idx]}): {model_path}"
                    )
            
            if missing_models:
                print("✗ ERROR: Missing SAM2 models:")
                for msg in missing_models:
                    print(f"  - {msg}")
                print("\n  Run with --mode train-sam2 first")
                sys.exit(1)
            
            run_inference()
            create_upload_folder()
        
        # Success summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print_header("PIPELINE COMPLETED SUCCESSFULLY ✓")
        print(f"Duration: {duration}")
        
        if args.mode in ['full', 'inference']:
            print(f"\nUpload folder ready: {UPLOAD_FOLDER}")
            print("  Contains: images + labels + data.yaml")
            print("  Ready to upload to Roboflow!")
        
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
    # --mode full
    # --mode inference