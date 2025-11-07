#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fish Segmentation Pipeline - v4.0.0 (MODULAR & CONFIG-DRIVEN)

A highly modular, configuration-driven pipeline for fish segmentation using:
- Mask R-CNN for detection and classification
- SAM2 for segmentation refinement
- Flexible inference modes (Mask R-CNN only, SAM2 only, or hybrid)
- Upload folder creation for Roboflow

Author: Fish Segmentation Project
Version: 4.0.0 - Fully modular and configuration-driven architecture

# 1. Setup dataset (creates annotation files)
python src/main.py --mode setup

# 2. Train Mask R-CNN (+ runs inference for SAM2 training)
python src/main.py --mode train-maskrcnn

# 3. Train all SAM2 models (one per class)
python src/main.py --mode train-sam2

# 4. Train SAM2 for specific class
python src/main.py --mode train-sam2-class --class_index 0
python src/main.py --mode train-sam2-class --class_index 1

# 5. Run Mask R-CNN inference only
python src/main.py --mode inference-maskrcnn

# 6. Create upload folder (uses inference_mode from config.yaml)
python src/main.py --mode upload-folder

# 7. Full pipeline (controlled by config.yaml)
python src/main.py --mode full

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
from typing import Dict, List, Optional, Any
from enum import Enum


# =============================================================================
# CONSTANTS & ENUMS
# =============================================================================

class InferenceMode(Enum):
    """Available inference modes."""
    MASKRCNN = "maskrcnn"
    HYBRID = "hybrid"


class PipelineMode(Enum):
    """Available pipeline execution modes."""
    FULL = "full"
    SETUP = "setup"
    TRAIN_MASKRCNN = "train-maskrcnn"
    TRAIN_SAM2 = "train-sam2"
    TRAIN_SAM2_CLASS = "train-sam2-class"
    INFERENCE_MASKRCNN = "inference-maskrcnn"
    UPLOAD_FOLDER = "upload-folder"


# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

class ConfigManager:
    """Manages pipeline configuration from YAML file."""
    
    def __init__(self, config_path: str = 'src/config.yaml'):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            SystemExit: If configuration file not found or invalid
        """
        if not os.path.exists(self.config_path):
            print(f"✗ ERROR: Configuration file not found: {self.config_path}")
            sys.exit(1)
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✓ Configuration loaded from: {self.config_path}")
            return config
        except Exception as e:
            print(f"✗ ERROR: Failed to load configuration: {e}")
            sys.exit(1)
    
    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation.
        
        Args:
            *keys: Nested keys to access
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            config.get('training_maskrcnn', 'num_epochs')
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_inference_mode(self) -> InferenceMode:
        """
        Get inference mode from configuration.
        
        Returns:
            InferenceMode enum value
        """
        mode_str = self.get('pipeline', 'inference_mode', default='hybrid')
        try:
            return InferenceMode(mode_str.lower())
        except ValueError:
            print(f"✗ WARNING: Invalid inference_mode '{mode_str}', "
                  f"defaulting to 'hybrid'")
            return InferenceMode.HYBRID
    
    def should_train_maskrcnn(self) -> bool:
        """Check if Mask R-CNN training is enabled in config."""
        return self.get('pipeline', 'train_maskrcnn', default=True)
    
    def should_train_sam2(self) -> bool:
        """Check if SAM2 training is enabled in config."""
        return self.get('pipeline', 'train_sam2', default=True)
    
    def should_create_upload_folder(self) -> bool:
        """Check if upload folder creation is enabled in config."""
        return self.get('pipeline', 'create_upload_folder', default=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(text: str) -> None:
    """Print formatted header."""
    print("\n" + "="*80)
    print(text)
    print("="*80)


def run_command(cmd: List[str], error_message: str) -> None:
    """
    Run external command and handle errors.
    
    Args:
        cmd: Command to execute
        error_message: Error message to display if command fails
        
    Raises:
        SystemExit: If command fails
    """
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR: {error_message} (exit code {e.returncode})")
        sys.exit(1)
    except FileNotFoundError:
        script_name = cmd[1] if len(cmd) > 1 else "script"
        print(f"\n✗ ERROR: {script_name} not found")
        sys.exit(1)


def clean_directory(directory: str, description: str = "directory") -> None:
    """
    Clean (remove and recreate) a directory.
    
    Args:
        directory: Directory path to clean
        description: Description for logging
    """
    if os.path.exists(directory):
        print(f"Cleaning existing {description}...")
        shutil.rmtree(directory)


def verify_model_exists(model_path: str, model_name: str) -> None:
    """
    Verify that a model file exists.
    
    Args:
        model_path: Path to model file
        model_name: Name of model for error message
        
    Raises:
        SystemExit: If model not found
    """
    if not os.path.exists(model_path):
        print(f"✗ ERROR: {model_name} model not found: {model_path}")
        print(f"  Train {model_name} first")
        sys.exit(1)


# =============================================================================
# PREREQUISITE CHECKING
# =============================================================================

class PrerequisiteChecker:
    """Checks pipeline prerequisites."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize prerequisite checker.
        
        Args:
            config: Configuration manager
        """
        self.config = config
    
    def check_all(self) -> None:
        """Check all prerequisites."""
        print("Checking prerequisites...")
        self._check_directories()
        self._check_data_yaml()
        print("✓ Prerequisites check passed")
    
    def _check_directories(self) -> None:
        """Check required directories exist."""
        dataset_path = self.config.get('dataset', 'dataset_path')
        
        required_dirs = [
            # COCO format (for training)
            os.path.join(dataset_path, 'original_coco/train'),
            os.path.join(dataset_path, 'original_coco/valid'),
            os.path.join(dataset_path, 'original_coco/test'),
            # YOLO format (for combining labels)
            os.path.join(dataset_path, 'original_yolo/test/labels'),
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                print(f"✗ ERROR: Required directory not found: {dir_path}")
                print("\nYou need BOTH:")
                print("  - dataset/original_coco/ (for training)")
                print("  - dataset/original_yolo/ (for label combining)")
                sys.exit(1)
        
        print("  ✓ COCO format dataset (for training)")
        print("  ✓ YOLO format dataset (for label combining)")
    
    def _check_data_yaml(self) -> None:
        """Check data.yaml exists."""
        original_yolo_data_yaml = self.config.get('output', 
                                                    'original_yolo_data_yaml')
        if not os.path.exists(original_yolo_data_yaml):
            print(f"⚠ WARNING: data.yaml not found at {original_yolo_data_yaml}")
            print("  You'll need this for uploading to Roboflow")


# =============================================================================
# PIPELINE STEPS
# =============================================================================

class DatasetSetup:
    """Handles dataset setup."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize dataset setup.
        
        Args:
            config: Configuration manager
        """
        self.config = config
    
    def run(self) -> None:
        """Setup dataset structure using filter_coco.py."""
        print_header("STEP 0: DATASET SETUP")
        
        dataset_path = self.config.get('dataset', 'dataset_path')
        classes_to_keep = self.config.get('dataset', 'classes_to_keep')
        
        print(f"Dataset: {dataset_path}")
        print(f"Classes: {classes_to_keep}")
        print("Setting up train/valid/test folders...")
        
        cmd = [
            sys.executable, "src/utils/filter_coco.py",
            dataset_path,
            "--classes_to_keep", json.dumps(classes_to_keep),
        ]
        
        print("\nRunning dataset setup...")
        run_command(cmd, "Dataset setup failed")
        print("✓ Dataset setup completed")


class MaskRCNNTrainer:
    """Handles Mask R-CNN training."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize Mask R-CNN trainer.
        
        Args:
            config: Configuration manager
        """
        self.config = config
    
    def run(self) -> None:
        """Train Mask R-CNN model."""
        print_header("STEP 1: TRAINING MASK R-CNN")
        
        dataset_path = self.config.get('dataset', 'dataset_path')
        model_path = self.config.get('models', 'maskrcnn_model_path')
        train_config = self.config.get('training_maskrcnn')
        output_config = self.config.get('output')
        class_mapping = self.config.get('dataset', 'class_mapping')
        class_names = self.config.get('dataset', 'class_names_mapping')
        
        print(f"Model: {model_path}")
        print(f"Epochs: {train_config['num_epochs']}")
        print(f"IMG_SIZE: {train_config['img_size']}")
        print(f"MASK_RESOLUTION: {train_config['mask_resolution']}")
        print(f"Classes: {self.config.get('dataset', 'classes_to_keep')}")
        
        cmd = self._build_command(dataset_path, model_path, train_config, 
                                  output_config, class_mapping, class_names)
        
        print("\nRunning Mask R-CNN training...")
        run_command(cmd, "Mask R-CNN training failed")
        print("✓ Mask R-CNN training completed")
    
    def _build_command(self, dataset_path: str, model_path: str,
                       train_config: Dict, output_config: Dict,
                       class_mapping: Dict, class_names: Dict) -> List[str]:
        """Build training command."""
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
        if train_config.get('use_focal_dice', False):
            cmd.append("--use_focal_dice")
        else:
            cmd.append("--no_use_focal_dice")
        
        if train_config.get('oversample_small_objects', False):
            cmd.append("--oversample_small_objects")
        else:
            cmd.append("--no_oversample_small_objects")
        
        if train_config.get('use_copy_paste', False):
            cmd.append("--use_copy_paste")
        else:
            cmd.append("--no_use_copy_paste")
        
        return cmd


class SAM2Trainer:
    """Handles SAM2 training."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize SAM2 trainer.
        
        Args:
            config: Configuration manager
        """
        self.config = config
    
    def train_single_class(self, class_index: int) -> None:
        """
        Train SAM2 model for a specific class.
        
        Args:
            class_index: Index of the class to train
        """
        class_names_mapping = self.config.get('dataset', 'class_names_mapping')
        
        if class_index not in class_names_mapping:
            print(f"✗ ERROR: Invalid class_index {class_index}")
            sys.exit(1)
        
        class_name = class_names_mapping[class_index]
        model_path = self.config.get('models', 'sam2_model_paths')[class_index]
        
        print_header(f"STEP 2.{class_index + 1}: TRAINING SAM2 FOR "
                     f"{class_name.upper()}")
        
        dataset_path = self.config.get('dataset', 'dataset_path')
        sam2_model_id = self.config.get('models', 'sam2_model_id')
        train_config = self.config.get('training_sam2')
        output_config = self.config.get('output')
        
        print(f"Model: {model_path}")
        print(f"Training on: {class_name} (class {class_index})")
        print(f"Epochs: {train_config['num_epochs']}")
        print(f"IMG_SIZE: {train_config['img_size']}")
        
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
        
        if train_config.get('train_unified', False):
            cmd.append("--train_unified")
        
        print(f"\nRunning SAM2 training for {class_name}...")
        run_command(cmd, f"SAM2 training failed for {class_name}")
        print(f"✓ SAM2 training completed for {class_name}")
        print(f"  Model saved to: {model_path}")
        print(f"  Report saved to: {output_config['report_output_path']}/")
    
    def train_all_classes(self) -> None:
        """Train SAM2 models for all classes sequentially."""
        print_header("STEP 2: TRAINING SAM2 MODELS (ONE PER CLASS)")
        
        class_names_mapping = self.config.get('dataset', 'class_names_mapping')
        
        print(f"Training {len(class_names_mapping)} SAM2 models:")
        for class_idx, class_name in class_names_mapping.items():
            print(f"  - Class {class_idx}: {class_name}")
        print("="*80)
        
        for class_idx in sorted(class_names_mapping.keys()):
            self.train_single_class(class_idx)
        
        print_header("ALL SAM2 MODELS TRAINED SUCCESSFULLY ✓")


class InferenceEngine:
    """Handles inference operations."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize inference engine.
        
        Args:
            config: Configuration manager
        """
        self.config = config
    
    def run_maskrcnn_inference(self) -> None:
        """Run standalone Mask R-CNN inference."""
        print_header("MASK R-CNN INFERENCE")
        
        dataset_path = self.config.get('dataset', 'dataset_path')
        model_path = self.config.get('models', 'maskrcnn_model_path')
        maskrcnn_img_size = self.config.get('training_maskrcnn', 'img_size')
        mask_resolution = self.config.get('training_maskrcnn', 'mask_resolution')
        detection_threshold = self.config.get('inference', 'detection_threshold')
        
        # Get inference parameters
        min_mask_area = self.config.get('inference', 'min_mask_area')
        polygon_tolerance = self.config.get('inference', 'polygon_tolerance')
        maskrcnn_mask_threshold = self.config.get('inference', 
                                                    'maskrcnn_mask_threshold')
        
        output_labels = self.config.get('output', 'output_maskrcnn_labels')
        output_images = self.config.get('output', 'output_maskrcnn_images')
        inference_folder = self.config.get('output', 'inference_maskrcnn_folder')
        
        class_names = self.config.get('dataset', 'class_names_mapping')
        
        print(f"Model: {model_path}")
        print(f"IMG_SIZE: {maskrcnn_img_size}")
        print(f"MASK_RESOLUTION: {mask_resolution}")
        print(f"Output: {output_labels}")
        
        # Clean inference folder
        clean_directory(inference_folder, "Mask R-CNN inference folder")
        
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
        
        print("\nRunning Mask R-CNN inference...")
        run_command(cmd, "Mask R-CNN inference failed")
        print("✓ Mask R-CNN inference completed")
        print(f"  Labels saved to: {output_labels}")
        print(f"  Images saved to: {output_images}")
    
    def run_sam2_inference(self) -> None:
        """Run standalone SAM2 inference."""
        print_header("SAM2 INFERENCE")
        
        dataset_path = self.config.get('dataset', 'dataset_path')
        sam2_model_paths = self.config.get('models', 'sam2_model_paths')
        sam2_model_id = self.config.get('models', 'sam2_model_id')
        
        output_labels = self.config.get('output', 'output_sam2_labels')
        output_images = self.config.get('output', 'output_sam2_images')
        inference_folder = self.config.get('output', 'inference_sam2_folder')
        
        class_names = self.config.get('dataset', 'class_names_mapping')
        
        print(f"SAM2 Models:")
        for class_idx, model_path in sam2_model_paths.items():
            class_name = class_names[class_idx]
            print(f"  - Class {class_idx} ({class_name}): {model_path}")
        
        print(f"Output: {output_labels}")
        
        # Clean inference folder
        clean_directory(inference_folder, "SAM2 inference folder")
        
        # Get inference parameters
        min_mask_area = self.config.get('inference', 'min_mask_area')
        polygon_tolerance = self.config.get('inference', 'polygon_tolerance')
        mask_threshold = self.config.get('inference', 'mask_threshold')
        
        cmd = [
            sys.executable, "src/utils/infer_sam2.py",
            "--sam2_model_paths", json.dumps(sam2_model_paths),
            "--sam2_model_id", sam2_model_id,
            "--input_folder", os.path.join(dataset_path, "test"),
            "--output_labels", output_labels,
            "--output_images", output_images,
            "--class_names", json.dumps(class_names),
            "--min_mask_area", str(min_mask_area),
            "--polygon_tolerance", str(polygon_tolerance),
            "--mask_threshold", str(mask_threshold)
        ]
        
        print("\nRunning SAM2 inference...")
        run_command(cmd, "SAM2 inference failed")
        print("✓ SAM2 inference completed")
        print(f"  Labels saved to: {output_labels}")
        print(f"  Images saved to: {output_images}")
    
    def run_hybrid_inference(self) -> None:
        """Run hybrid Mask R-CNN + SAM2 inference."""
        print_header("HYBRID INFERENCE (Mask R-CNN + SAM2)")
        
        dataset_path = self.config.get('dataset', 'dataset_path')
        maskrcnn_model_path = self.config.get('models', 'maskrcnn_model_path')
        sam2_model_paths = self.config.get('models', 'sam2_model_paths')
        sam2_model_id = self.config.get('models', 'sam2_model_id')
        
        maskrcnn_img_size = self.config.get('training_maskrcnn', 'img_size')
        detection_threshold = self.config.get('inference', 'detection_threshold')
        
        output_labels = self.config.get('output', 'output_labels_folder')
        output_images = self.config.get('output', 'output_images_folder')
        inference_folder = self.config.get('output', 'inference_folder_path')
        
        class_names_mapping = self.config.get('dataset', 'class_names_mapping')
        
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
        
        # Clean inference folder
        clean_directory(inference_folder, "inference folder")
        
        # Get inference parameters
        min_mask_area = self.config.get('inference', 'min_mask_area')
        polygon_tolerance = self.config.get('inference', 'polygon_tolerance')
        mask_threshold = self.config.get('inference', 'mask_threshold')
        erode_iterations = self.config.get('inference', 'erode_iterations')
        mask_resolution = self.config.get('training_maskrcnn', 'mask_resolution')
        
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
        
        print("\nRunning hybrid inference...")
        run_command(cmd, "Hybrid inference failed")
        print("✓ Hybrid inference completed")


class UploadFolderCreator:
    """Handles upload folder creation."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize upload folder creator.
        
        Args:
            config: Configuration manager
        """
        self.config = config
    
    def create(self) -> None:
        """Create upload folder based on configuration."""
        print_header("CREATING UPLOAD FOLDER")
        
        inference_mode = self.config.get_inference_mode()
        
        print(f"Inference mode: {inference_mode.value}")
        
        if inference_mode == InferenceMode.MASKRCNN:
            self._create_from_maskrcnn()
        else:  # HYBRID
            self._create_from_hybrid()
    
    def _create_from_maskrcnn(self) -> None:
        """Create upload folder from Mask R-CNN inference."""
        print("Creating upload folder from Mask R-CNN inference...")
        
        output_labels = self.config.get('output', 'output_maskrcnn_labels')
        output_labels_full = self.config.get('output', 
                                              'output_maskrcnn_labels_full')
        upload_folder = self.config.get('output', 'upload_maskrcnn_folder')
        
        self._run_adapt2rbf(output_labels, output_labels_full, upload_folder)
    
    def _create_from_hybrid(self) -> None:
        """Create upload folder from hybrid inference."""
        print("Creating upload folder from hybrid inference...")
        
        output_labels = self.config.get('output', 'output_labels_folder')
        output_labels_full = self.config.get('output', 
                                              'output_labels_full_folder')
        upload_folder = self.config.get('output', 'upload_folder')
        
        self._run_adapt2rbf(output_labels, output_labels_full, upload_folder)
    
    def _run_adapt2rbf(self, inference_labels: str, output_labels_full: str,
                       upload_folder: str) -> None:
        """
        Run adapt2rbf.py to create upload folder.
        
        Args:
            inference_labels: Path to inference labels
            output_labels_full: Path to output combined labels
            upload_folder: Path to upload folder
        """
        original_yolo_labels = self.config.get('output', 
                                                'original_yolo_labels_folder')
        original_img_folder = self.config.get('output', 'original_img_folder')
        target_classes = self.config.get('dataset', 
                                          'target_classes_for_replacement')
        
        print(f"Upload folder: {upload_folder}")
        print("This will create images + labels + data.yaml ready for Roboflow")
        
        # Clean upload folder
        clean_directory(upload_folder, "upload folder")
        
        cmd = [
            sys.executable, "src/utils/adapt2rbf.py",
            "--inference_folder", inference_labels,
            "--original_folder", original_yolo_labels,
            "--output_folder", output_labels_full,
            "--target_classes", json.dumps(target_classes),
            "--upload_folder", upload_folder,
            "--original_img_folder", original_img_folder
        ]
        
        print("\nRunning adapt2rbf...")
        run_command(cmd, "Upload folder creation failed")
        print("✓ Upload folder created")
        
        # Verify contents
        self._verify_upload_folder(upload_folder)
    
    def _verify_upload_folder(self, upload_folder: str) -> None:
        """
        Verify upload folder contents.
        
        Args:
            upload_folder: Path to upload folder
        """
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


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class PipelineOrchestrator:
    """Orchestrates the complete pipeline based on configuration."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize pipeline orchestrator.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.prerequisite_checker = PrerequisiteChecker(config)
        self.dataset_setup = DatasetSetup(config)
        self.maskrcnn_trainer = MaskRCNNTrainer(config)
        self.sam2_trainer = SAM2Trainer(config)
        self.inference_engine = InferenceEngine(config)
        self.upload_creator = UploadFolderCreator(config)
    
    def run_full_pipeline(self) -> None:
        """Run complete pipeline based on configuration."""
        print_header("RUNNING FULL PIPELINE (CONFIG-DRIVEN)")
        
        # Setup
        self.dataset_setup.run()
        
        # Training
        if self.config.should_train_maskrcnn():
            self.maskrcnn_trainer.run()
            self.inference_engine.run_maskrcnn_inference()
        
        if self.config.should_train_sam2():
            # Verify Mask R-CNN model exists
            maskrcnn_model = self.config.get('models', 'maskrcnn_model_path')
            verify_model_exists(maskrcnn_model, "Mask R-CNN")
            self.sam2_trainer.train_all_classes()
        
        # Inference
        self._run_inference_based_on_config()
        
        # Upload folder
        if self.config.should_create_upload_folder():
            self.upload_creator.create()
    
    def _run_inference_based_on_config(self) -> None:
        """Run inference based on configuration."""
        inference_mode = self.config.get_inference_mode()
        
        if inference_mode == InferenceMode.MASKRCNN:
            maskrcnn_model = self.config.get('models', 'maskrcnn_model_path')
            verify_model_exists(maskrcnn_model, "Mask R-CNN")
            self.inference_engine.run_maskrcnn_inference()
        
        elif inference_mode == InferenceMode.SAM2:
            self._verify_sam2_models()
            self.inference_engine.run_sam2_inference()
        
        else:  # HYBRID
            maskrcnn_model = self.config.get('models', 'maskrcnn_model_path')
            verify_model_exists(maskrcnn_model, "Mask R-CNN")
            self._verify_sam2_models()
            self.inference_engine.run_hybrid_inference()
    
    def _verify_sam2_models(self) -> None:
        """Verify all SAM2 models exist."""
        sam2_model_paths = self.config.get('models', 'sam2_model_paths')
        class_names = self.config.get('dataset', 'class_names_mapping')
        
        missing_models = []
        for class_idx, model_path in sam2_model_paths.items():
            if not os.path.exists(model_path):
                missing_models.append(
                    f"Class {class_idx} ({class_names[class_idx]}): {model_path}"
                )
        
        if missing_models:
            print("✗ ERROR: Missing SAM2 models:")
            for msg in missing_models:
                print(f"  - {msg}")
            print("\n  Run with --mode train-sam2 first")
            sys.exit(1)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='Fish Segmentation Pipeline v4.0.0 - Modular & Config-Driven',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (config-driven: trains/infers based on config.yaml)
  python src/main.py --mode full
  
  # Individual steps
  python src/main.py --mode setup
  python src/main.py --mode train-maskrcnn
  python src/main.py --mode train-sam2
  python src/main.py --mode train-sam2-class --class_index 0
  
  # Inference modes (uses inference_mode from config.yaml)
  python src/main.py --mode inference-maskrcnn  # Mask R-CNN only
  
  # Upload folder (uses inference_mode from config.yaml)
  python src/main.py --mode upload-folder
  
  # Custom config file
  python src/main.py --mode full --config path/to/custom_config.yaml

Configuration:
  Edit src/config.yaml to control pipeline behavior:
  - pipeline.train_maskrcnn: Enable/disable Mask R-CNN training
  - pipeline.train_sam2: Enable/disable SAM2 training
  - pipeline.inference_mode: Choose 'maskrcnn', 'sam2', or 'hybrid'
  - pipeline.create_upload_folder: Enable/disable upload folder creation
"""
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=[mode.value for mode in PipelineMode],
        default=PipelineMode.FULL.value,
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
        help='Path to configuration YAML file (default: src/config.yaml)'
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    start_time = datetime.now()
    
    # Load configuration
    config = ConfigManager(args.config)
    
    print("="*80)
    print("FISH SEGMENTATION PIPELINE v4.0.0 - MODULAR & CONFIG-DRIVEN")
    print("="*80)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode}")
    print(f"Config file: {args.config}")
    print("="*80)
    
    # Check prerequisites
    prerequisite_checker = PrerequisiteChecker(config)
    prerequisite_checker.check_all()
    
    # Create orchestrator and components
    orchestrator = PipelineOrchestrator(config)
    
    try:
        # Execute based on mode
        mode = PipelineMode(args.mode)
        
        if mode == PipelineMode.FULL:
            orchestrator.run_full_pipeline()
        
        elif mode == PipelineMode.SETUP:
            orchestrator.dataset_setup.run()
        
        elif mode == PipelineMode.TRAIN_MASKRCNN:
            orchestrator.maskrcnn_trainer.run()
            orchestrator.inference_engine.run_maskrcnn_inference()
        
        elif mode == PipelineMode.TRAIN_SAM2:
            maskrcnn_model = config.get('models', 'maskrcnn_model_path')
            verify_model_exists(maskrcnn_model, "Mask R-CNN")
            orchestrator.sam2_trainer.train_all_classes()
        
        elif mode == PipelineMode.TRAIN_SAM2_CLASS:
            if args.class_index is None:
                print("✗ ERROR: --class_index required for train-sam2-class mode")
                sys.exit(1)
            
            maskrcnn_model = config.get('models', 'maskrcnn_model_path')
            verify_model_exists(maskrcnn_model, "Mask R-CNN")
            orchestrator.sam2_trainer.train_single_class(args.class_index)
        
        elif mode == PipelineMode.INFERENCE_MASKRCNN:
            maskrcnn_model = config.get('models', 'maskrcnn_model_path')
            verify_model_exists(maskrcnn_model, "Mask R-CNN")
            orchestrator.inference_engine.run_maskrcnn_inference()
        
        elif mode == PipelineMode.UPLOAD_FOLDER:
            orchestrator.upload_creator.create()
        
        # Success summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print_header("PIPELINE COMPLETED SUCCESSFULLY ✓")
        print(f"Duration: {duration}")
        
        if mode in [PipelineMode.FULL, PipelineMode.UPLOAD_FOLDER]:
            inference_mode = config.get_inference_mode()
            
            if inference_mode == InferenceMode.MASKRCNN:
                upload_folder = config.get('output', 'upload_maskrcnn_folder')
                print(f"\nMask R-CNN upload folder ready: {upload_folder}")
            else:
                upload_folder = config.get('output', 'upload_folder')
                print(f"\nHybrid upload folder ready: {upload_folder}")
            
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
