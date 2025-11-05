#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fish Segmentation Pipeline - v3.1.0 (WITH CONFIG.YAML)

Complete workflow:
0. Setup dataset structure (filter_coco.py)
1. Train Mask R-CNN (detection + classification)
2. Train unified SAM2 (segmentation refinement)
3. Run hybrid inference (Mask R-CNN + SAM2)
4. Create upload folder with data.yaml (adapt2rbf.py)

Author: Fish Segmentation Project
Version: 3.1.0 - Now with centralized YAML configuration
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

# Use custom config file
python src/main.py --mode full --config path/to/custom_config.yaml

NOTES:
------
- Configuration is now centralized in config.yaml
- Each SAM2 training generates a PDF report in results/
- Inference processes all classes sequentially (saves GPU memory)
- Upload folder (dataset/upload/) contains: images + labels + data.yaml
- All models use IMG_SIZE from config.yaml

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
import yaml
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path='src/config.yaml'):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        sys.exit(1)

# Global configuration - will be set in main()
CONFIG = None

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
    
    dataset_path = CONFIG['dataset']['dataset_path']
    
    # Need BOTH COCO and YOLO formats
    required_dirs = [
        # COCO format (for training)
        os.path.join(dataset_path, 'original_coco/train'),
        os.path.join(dataset_path, 'original_coco/valid'),
        os.path.join(dataset_path, 'original_coco/test'),
        # YOLO format (for combining labels)
        os.path.join(dataset_path, 'original_yolo/test/labels'),
    ]
    
    # Check data.yaml exists
    original_yolo_data_yaml = CONFIG['output']['original_yolo_data_yaml']
    if not os.path.exists(original_yolo_data_yaml):
        print(f"WARNING: data.yaml not found at {original_yolo_data_yaml}")
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
    
    dataset_path = CONFIG['dataset']['dataset_path']
    classes_to_keep = CONFIG['dataset']['classes_to_keep']
    
    print(f"Dataset: {dataset_path}")
    print(f"Classes: {classes_to_keep}")
    print("Setting up train/valid/test folders...")
    
    try:
        cmd = [
            sys.executable, "src/utils/filter_coco.py",
            dataset_path,
            "--classes_to_keep", json.dumps(classes_to_keep),
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
    
    dataset_path = CONFIG['dataset']['dataset_path']
    model_path = CONFIG['models']['maskrcnn_model_path']
    
    train_config = CONFIG['training_maskrcnn']
    output_config = CONFIG['output']
    class_mapping = CONFIG['dataset']['class_mapping']
    class_names = CONFIG['dataset']['class_names_mapping']
    
    print(f"Model: {model_path}")
    print(f"Epochs: {train_config['num_epochs']}")
    print(f"IMG_SIZE: {train_config['img_size']}")
    print(f"MASK_RESOLUTION: {train_config['mask_resolution']}")
    print(f"Classes: {CONFIG['dataset']['classes_to_keep']}")
    
    try:
        cmd = [
            sys.executable, "src/utils/train.py",
            "--dataset_path", dataset_path,
            "--model_path", model_path,
            "--num_epochs", str(train_config['num_epochs']),
            "--batch_size", str(train_config['batch_size']),
            "--learning_rate", str(train_config['learning_rate']),
            "--img_size", str(train_config['img_size']),
            "--mask_resolution", str(train_config['mask_resolution']),
            "--num_classes", str(train_config['num_classes']),
            "--confidence_threshold", str(train_config['confidence_threshold']),
            "--dice_weight", str(train_config['dice_weight']),
            "--base_min_anchor", str(train_config['base_min_anchor']),
            "--rpn_pre_nms_top_n_train", str(train_config['rpn_pre_nms_top_n_train']),
            "--rpn_post_nms_top_n_train", str(train_config['rpn_post_nms_top_n_train']),
            "--rpn_nms_thresh", str(train_config['rpn_nms_thresh']),
            "--report_output_path", output_config['report_output_path'],
            "--class_mapping", json.dumps(class_mapping),
            "--class_names", json.dumps(class_names)
        ]
        
        # Add boolean flags
        if train_config['use_focal_dice']:
            cmd.append("--use_focal_dice")
        else:
            cmd.append("--no_use_focal_dice")
        
        if train_config['oversample_small_objects']:
            cmd.append("--oversample_small_objects")
        else:
            cmd.append("--no_oversample_small_objects")
        
        if train_config['use_copy_paste']:
            cmd.append("--use_copy_paste")
        else:
            cmd.append("--no_use_copy_paste")
        
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
    
    dataset_path = CONFIG['dataset']['dataset_path']
    model_path = CONFIG['models']['maskrcnn_model_path']
    maskrcnn_img_size = CONFIG['training_maskrcnn']['img_size']
    mask_resolution = CONFIG['training_maskrcnn']['mask_resolution']
    detection_threshold = CONFIG['inference']['detection_threshold']
    
    # Get additional inference parameters
    min_mask_area = CONFIG['inference']['min_mask_area']
    polygon_tolerance = CONFIG['inference']['polygon_tolerance']
    maskrcnn_mask_threshold = CONFIG['inference']['maskrcnn_mask_threshold']
    
    output_labels = CONFIG['output']['output_maskrcnn_labels']
    output_images = CONFIG['output']['output_maskrcnn_images']
    inference_folder = CONFIG['output']['inference_maskrcnn_folder']
    
    class_names = CONFIG['dataset']['class_names_mapping']
    
    print(f"Model: {model_path}")
    print(f"IMG_SIZE: {maskrcnn_img_size}")
    print(f"MASK_RESOLUTION: {mask_resolution}")
    print(f"Output: {output_labels}")
    
    # Clean inference folders
    if os.path.exists(inference_folder):
        print(f"\nCleaning existing Mask R-CNN inference folder...")
        shutil.rmtree(inference_folder)
    
    try:
        cmd = [
            sys.executable, "src/utils/infer_maskrcnn.py",
            "--maskrcnn_model", model_path,
            "--input_folder", os.path.join(dataset_path, "test"),
            "--output_labels", output_labels,
            "--output_images", output_images,
            "--detection_threshold", str(detection_threshold),
            "--maskrcnn_img_size", str(maskrcnn_img_size),
            "--mask_resolution", str(mask_resolution),
            "--class_names", json.dumps(class_names),
            "--min_mask_area", str(min_mask_area),
            "--polygon_tolerance", str(polygon_tolerance),
            "--maskrcnn_mask_threshold", str(maskrcnn_mask_threshold)
        ]
        
        print(f"\nRunning Mask R-CNN inference...")
        result = subprocess.run(cmd, check=True)
        print("✓ Mask R-CNN inference completed")
        print(f"  Labels saved to: {output_labels}")
        print(f"  Images saved to: {output_images}")
        
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
    class_names_mapping = CONFIG['dataset']['class_names_mapping']
    
    if class_index not in class_names_mapping:
        print(f"✗ ERROR: Invalid class_index {class_index}")
        sys.exit(1)
    
    class_name = class_names_mapping[class_index]
    model_path = CONFIG['models']['sam2_model_paths'][class_index]
    
    dataset_path = CONFIG['dataset']['dataset_path']
    sam2_model_id = CONFIG['models']['sam2_model_id']
    
    train_config = CONFIG['training_sam2']
    output_config = CONFIG['output']
    
    print_header(
        f"STEP 2.{class_index + 1}: TRAINING SAM2 FOR {class_name.upper()}"
    )
    print(f"Model: {model_path}")
    print(f"Training on: {class_name} (class {class_index})")
    print(f"Epochs: {train_config['num_epochs']}")
    print(f"IMG_SIZE: {train_config['img_size']}")
    
    try:
        cmd = [
            sys.executable, "src/utils/train_sam2.py",
            "--dataset_path", dataset_path,
            "--model_path", model_path,
            "--sam2_model_id", sam2_model_id,
            "--num_epochs", str(train_config['num_epochs']),
            "--batch_size", str(train_config['batch_size']),
            "--learning_rate", str(train_config['learning_rate']),
            "--img_size", str(train_config['img_size']),
            "--report_output_path", output_config['report_output_path'],
            "--class_names", json.dumps(class_names_mapping),
            "--target_class_index", str(class_index),
            "--gradient_accumulation_steps",
            str(train_config['gradient_accumulation_steps']),
            "--dice_weight", str(train_config['dice_weight']),
            "--max_instances_per_image", str(train_config['max_instances_per_image'])
        ]
        
        # Add train_unified flag if needed
        if train_config.get('train_unified', False):
            cmd.append("--train_unified")
        
        print(f"\nRunning SAM2 training for {class_name}...")
        result = subprocess.run(cmd, check=True)
        print(f"✓ SAM2 training completed for {class_name}")
        print(f"  Model saved to: {model_path}")
        print(f"  Report saved to: {output_config['report_output_path']}/")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: Training failed (code {e.returncode})")
        sys.exit(1)


def train_all_sam2_models():
    """Train SAM2 models for all classes sequentially."""
    print_header("STEP 2: TRAINING SAM2 MODELS (ONE PER CLASS)")
    
    class_names_mapping = CONFIG['dataset']['class_names_mapping']
    
    print(f"Training {len(class_names_mapping)} SAM2 models:")
    for class_idx, class_name in class_names_mapping.items():
        print(f"  - Class {class_idx}: {class_name}")
    print("="*80)
    
    for class_idx in sorted(class_names_mapping.keys()):
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
    
    dataset_path = CONFIG['dataset']['dataset_path']
    maskrcnn_model_path = CONFIG['models']['maskrcnn_model_path']
    sam2_model_paths = CONFIG['models']['sam2_model_paths']
    sam2_model_id = CONFIG['models']['sam2_model_id']
    
    maskrcnn_img_size = CONFIG['training_maskrcnn']['img_size']
    detection_threshold = CONFIG['inference']['detection_threshold']
    
    output_labels = CONFIG['output']['output_labels_folder']
    output_images = CONFIG['output']['output_images_folder']
    inference_folder = CONFIG['output']['inference_folder_path']
    
    class_names_mapping = CONFIG['dataset']['class_names_mapping']
    
    print(f"Mask R-CNN: {maskrcnn_model_path} "
          f"(IMG_SIZE={maskrcnn_img_size})")
    
    for class_idx, model_path in sam2_model_paths.items():
        class_name = class_names_mapping[class_idx]
        print(f"SAM2 [{class_name}]: {model_path}")
    
    print(f"Output: {output_labels}")
    print("\nStrategy: Sequential processing by class")
    print("  1. Mask R-CNN detects all bboxes")
    for idx, class_name in class_names_mapping.items():
        print(f"  {idx + 2}. SAM2_{class_name} processes all {class_name}")
    
    # Clean inference folders
    if os.path.exists(inference_folder):
        print(f"\nCleaning existing inference folder...")
        shutil.rmtree(inference_folder)
    
    # Get additional inference parameters
    min_mask_area = CONFIG['inference']['min_mask_area']
    polygon_tolerance = CONFIG['inference']['polygon_tolerance']
    mask_threshold = CONFIG['inference']['mask_threshold']
    erode_iterations = CONFIG['inference']['erode_iterations']
    mask_resolution = CONFIG['training_maskrcnn']['mask_resolution']
    
    try:
        # Build command with all SAM2 model paths and ALL inference parameters
        cmd = [
            sys.executable, "src/utils/infer_hybrid.py",
            "--maskrcnn_model", maskrcnn_model_path,
            "--sam2_model_paths", json.dumps(sam2_model_paths),
            "--sam2_model_id", sam2_model_id,
            "--input_folder", os.path.join(dataset_path, "test"),
            "--output_labels", output_labels,
            "--output_images", output_images,
            "--detection_threshold", str(detection_threshold),
            "--maskrcnn_img_size", str(maskrcnn_img_size),
            "--class_names", json.dumps(class_names_mapping),
            "--min_mask_area", str(min_mask_area),
            "--polygon_tolerance", str(polygon_tolerance),
            "--mask_threshold", str(mask_threshold),
            "--erode_iterations", str(erode_iterations),
            "--mask_resolution", str(mask_resolution)
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
    
    dataset_path = CONFIG['dataset']['dataset_path']
    output_labels = CONFIG['output']['output_labels_folder']
    output_labels_full = CONFIG['output']['output_labels_full_folder']
    upload_folder = CONFIG['output']['upload_folder']
    original_yolo_labels = CONFIG['output']['original_yolo_labels_folder']
    original_img_folder = CONFIG['output']['original_img_folder']
    target_classes = CONFIG['dataset']['target_classes_for_replacement']
    
    print(f"Upload folder: {upload_folder}")
    print("This will create images + labels + data.yaml ready for Roboflow")
    
    # Clean upload folder
    if os.path.exists(upload_folder):
        print(f"Cleaning existing upload folder...")
        shutil.rmtree(upload_folder)
    
    try:
        cmd = [
            sys.executable, "src/utils/adapt2rbf.py",
            "--inference_folder", output_labels,
            "--original_folder", original_yolo_labels,
            "--output_folder", output_labels_full,
            "--target_classes", json.dumps(target_classes),
            "--upload_folder", upload_folder,
            "--original_img_folder", original_img_folder
        ]
        
        print(f"\nRunning adapt2rbf...")
        result = subprocess.run(cmd, check=True)
        print("✓ Upload folder created")
        
        # Verify contents
        if os.path.exists(upload_folder):
            img_count = len([f for f in os.listdir(upload_folder)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            label_count = len([f for f in os.listdir(upload_folder)
                              if f.endswith('.txt')])
            yaml_exists = os.path.exists(os.path.join(upload_folder,
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
        description='Fish Segmentation Pipeline v3.1.0 - Multi-SAM2 with YAML Config',
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
  
  # Use custom config file
  python src/main.py --mode full --config path/to/custom_config.yaml
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
    
    parser.add_argument(
        '--config',
        type=str,
        default='src/config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    start_time = datetime.now()
    
    # Load configuration
    global CONFIG
    CONFIG = load_config(args.config)
    
    print("="*80)
    print("FISH SEGMENTATION PIPELINE v3.1.0 - MULTI-SAM2 WITH YAML CONFIG")
    print("="*80)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode}")
    print(f"Config file: {args.config}")
    print("="*80)
    
    check_prerequisites()
    
    maskrcnn_model_path = CONFIG['models']['maskrcnn_model_path']
    sam2_model_paths = CONFIG['models']['sam2_model_paths']
    class_names_mapping = CONFIG['dataset']['class_names_mapping']
    
    try:
        # Execute based on mode
        if args.mode == 'full':
            # Complete workflow
            setup_dataset()
            train_maskrcnn()
            run_maskrcnn_inference()
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
            if not os.path.exists(maskrcnn_model_path):
                print(f"✗ ERROR: Mask R-CNN model not found: "
                      f"{maskrcnn_model_path}")
                print("  Run with --mode train-maskrcnn first")
                sys.exit(1)
            train_all_sam2_models()
            
        elif args.mode == 'train-sam2-class':
            # Train SAM2 for specific class
            if args.class_index is None:
                print("✗ ERROR: --class_index required for "
                      "train-sam2-class mode")
                sys.exit(1)
            
            if not os.path.exists(maskrcnn_model_path):
                print(f"✗ ERROR: Mask R-CNN model not found: "
                      f"{maskrcnn_model_path}")
                sys.exit(1)
            
            train_sam2_per_class(args.class_index)

        elif args.mode == 'inference-maskrcnn':
            # Only Mask R-CNN inference
            if not os.path.exists(maskrcnn_model_path):
                print(f"✗ ERROR: Mask R-CNN model not found: {maskrcnn_model_path}")
                sys.exit(1)
            run_maskrcnn_inference()

        elif args.mode == 'inference':
            # Only inference + upload (requires all models)
            if not os.path.exists(maskrcnn_model_path):
                print(f"✗ ERROR: Mask R-CNN model not found: "
                      f"{maskrcnn_model_path}")
                sys.exit(1)
            
            # Check all SAM2 models exist
            missing_models = []
            for class_idx, model_path in sam2_model_paths.items():
                if not os.path.exists(model_path):
                    missing_models.append(
                        f"Class {class_idx} "
                        f"({class_names_mapping[class_idx]}): {model_path}"
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
            upload_folder = CONFIG['output']['upload_folder']
            print(f"\nUpload folder ready: {upload_folder}")
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
    # --mode inference-maskrcnn