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
MASKRCNN_MODEL_PATH = "weights/maskrcnn_model.pth"
UNIFIED_SAM2_MODEL = "weights/sam2_fish_unified.pth"

# Training parameters
NUM_EPOCHS = 300
BATCH_SIZE = 1
LEARNING_RATE = 5e-6
MASKRCNN_IMG_SIZE = 2048  # IMG_SIZE used for Mask R-CNN training/inference
SAM2_IMG_SIZE = 1024      # IMG_SIZE used for SAM2 training
GRADIENT_ACCUMULATION_STEPS = 2
DICE_WEIGHT = 2.0
MAX_INSTANCES_PER_IMAGE = 15

# Output
REPORT_OUTPUT_PATH = 'results'
INFERENCE_FOLDER_PATH = os.path.join(DATASET_PATH, 'inference')
OUTPUT_LABELS_FOLDER = os.path.join(INFERENCE_FOLDER_PATH, 'labels')
OUTPUT_IMAGES_FOLDER = os.path.join(INFERENCE_FOLDER_PATH, 'images')
UPLOAD_FOLDER = os.path.join(DATASET_PATH, 'upload')
ORIGINAL_YOLO_DATA_YAML = os.path.join(DATASET_PATH, 'original_yolo/data.yaml')

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
    print(f"IMG_SIZE: {MASKRCNN_IMG_SIZE}")  # ADD THIS LINE
    print(f"Classes: {CLASSES_TO_KEEP}")
    
    try:
        cmd = [
            sys.executable, "src/utils/train.py",
            "--dataset_path", DATASET_PATH,
            "--model_path", MASKRCNN_MODEL_PATH,
            "--num_epochs", str(NUM_EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--learning_rate", str(LEARNING_RATE),
            "--img_size", str(MASKRCNN_IMG_SIZE)  # ADD THIS LINE
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
# STEP 2: TRAIN UNIFIED SAM2
# =============================================================================

def train_sam2():
    """Train unified SAM2 model on all fish classes."""
    print_header("STEP 2: TRAINING UNIFIED SAM2")
    print(f"Model: {UNIFIED_SAM2_MODEL}")
    print(f"Training on ALL classes: {CLASSES_TO_KEEP}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"IMG_SIZE: {SAM2_IMG_SIZE}")  # ADD THIS LINE
    
    try:
        cmd = [
            sys.executable, "src/utils/train_sam2.py",
            "--dataset_path", DATASET_PATH,
            "--model_path", UNIFIED_SAM2_MODEL,
            "--sam2_model_id", SAM2_MODEL_ID,
            "--num_epochs", str(NUM_EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--learning_rate", str(LEARNING_RATE),
            "--img_size", str(SAM2_IMG_SIZE),  # MODIFY THIS LINE
            "--report_output_path", REPORT_OUTPUT_PATH,
            "--class_names", json.dumps(CLASS_NAMES_MAPPING),
            "--target_class_index", "0",
            "--gradient_accumulation_steps",
            str(GRADIENT_ACCUMULATION_STEPS),
            "--dice_weight", str(DICE_WEIGHT),
            "--max_instances_per_image", str(MAX_INSTANCES_PER_IMAGE),
            "--train_unified"
        ]
        
        print(f"\nRunning unified SAM2 training...")
        result = subprocess.run(cmd, check=True)
        print("✓ SAM2 training completed")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Training failed (code {e.returncode})")
        sys.exit(1)


# =============================================================================
# STEP 3: HYBRID INFERENCE
# =============================================================================

def run_inference():
    """Run hybrid Mask R-CNN + SAM2 inference."""
    print_header("STEP 3: HYBRID INFERENCE (Mask R-CNN + SAM2)")
    print(f"Mask R-CNN: {MASKRCNN_MODEL_PATH} (IMG_SIZE={MASKRCNN_IMG_SIZE})")
    print(f"SAM2: {UNIFIED_SAM2_MODEL}")
    print(f"Output: {OUTPUT_LABELS_FOLDER}")
    
    # Clean inference folders
    if os.path.exists(INFERENCE_FOLDER_PATH):
        print(f"Cleaning existing inference folder...")
        shutil.rmtree(INFERENCE_FOLDER_PATH)
    
    try:
        cmd = [
            sys.executable, "src/utils/infer_hybrid.py",
            "--maskrcnn_model", MASKRCNN_MODEL_PATH,
            "--sam2_model", UNIFIED_SAM2_MODEL,
            "--sam2_model_id", SAM2_MODEL_ID,
            "--input_folder", os.path.join(DATASET_PATH, "test"),
            "--output_labels", OUTPUT_LABELS_FOLDER,
            "--output_images", OUTPUT_IMAGES_FOLDER,
            "--detection_threshold", "0.5",
            "--maskrcnn_img_size", str(MASKRCNN_IMG_SIZE)  # ADD THIS LINE
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
            "--inferencefolder", OUTPUT_LABELS_FOLDER,
            "--originalfolder", os.path.join(DATASET_PATH,
                                             "original_yolo/test/labels"),
            "--outputfolder", os.path.join(INFERENCE_FOLDER_PATH,
                                            "labels_full"),
            "--targetclasses", json.dumps(TARGET_CLASSES_FOR_REPLACEMENT),
            "--uploadfolder", UPLOAD_FOLDER,
            "--originalimgfolder", os.path.join(DATASET_PATH, "test")
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
        description='Fish Segmentation Pipeline v3.0.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (everything from scratch)
  python src/main.py --mode full
  
  # Only setup dataset structure
  python src/main.py --mode setup
  
  # Only train Mask R-CNN (after setup)
  python src/main.py --mode train-maskrcnn
  
  # Only train SAM2 (requires Mask R-CNN)
  python src/main.py --mode train-sam2
  
  # Only inference + upload (requires both models)
  python src/main.py --mode inference
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'setup', 'train-maskrcnn', 'train-sam2',
                 'inference'],
        default='full',
        help='Execution mode'
    )
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print("="*80)
    print("FISH SEGMENTATION PIPELINE v3.0.0")
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
            train_sam2()
            run_inference()
            create_upload_folder()
            
        elif args.mode == 'setup':
            # Only setup dataset
            setup_dataset()
            
        elif args.mode == 'train-maskrcnn':
            # Only train Mask R-CNN
            train_maskrcnn()
            
        elif args.mode == 'train-sam2':
            # Only train SAM2 (requires Mask R-CNN)
            if not os.path.exists(MASKRCNN_MODEL_PATH):
                print(f"✗ ERROR: Mask R-CNN model not found: "
                      f"{MASKRCNN_MODEL_PATH}")
                print("  Run with --mode train-maskrcnn first")
                sys.exit(1)
            train_sam2()
            
        elif args.mode == 'inference':
            # Only inference + upload (requires both models)
            if not os.path.exists(MASKRCNN_MODEL_PATH):
                print(f"✗ ERROR: Mask R-CNN model not found: "
                      f"{MASKRCNN_MODEL_PATH}")
                sys.exit(1)
            if not os.path.exists(UNIFIED_SAM2_MODEL):
                print(f"✗ ERROR: SAM2 model not found: "
                      f"{UNIFIED_SAM2_MODEL}")
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
            print("   Contains: images + labels + data.yaml")
            print("   Ready to upload to Roboflow!")

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