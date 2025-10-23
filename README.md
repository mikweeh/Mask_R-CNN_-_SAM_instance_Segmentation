# Fish Instance Segmentation Pipeline v3.0.0

This repository contains the code for high-precision instance segmentation of small fish (Chromis chromis and Coris julis) using a hybrid approach that combines Mask R-CNN for detection/classification with SAM2 for mask refinement.

This code generates highly accurate segmentation masks to improve dataset quality. The hybrid approach (v3.0.0) provides significantly better segmentation boundaries than Mask R-CNN alone (v1.0.0) while maintaining proper class discrimination.


## Overview

The pipeline combines the strengths of two models:
- Mask R-CNN: Excellent at detecting fish and classifying species
- SAM2 (Segment Anything Model 2): Excellent at precise segmentation boundaries

This hybrid approach achieves the best of both worlds: correct species identification with high-quality mask boundaries.


## Installation

The repository runs in a Docker container and follows the same structure as other SRV group repositories.

1. Navigate to the 'repo' folder
2. Run 'docker compose up' to start the container
3. Execute the pipeline with:
   docker exec -it segmentation_cnt python3 ./src/main.py --mode full


## Dataset Preparation

Before running the pipeline, you must:

1. Set up the .env file with the DATASET variable pointing to your dataset's absolute path
2. Download your dataset from Roboflow in BOTH formats:
   - COCO format (for training Mask R-CNN and SAM2)
   - YOLOv11 format (for combining final labels)

3. Organize the dataset folder with this exact structure:
```
 dataset/
     |_ original_coco/       # COCO format dataset (for training)
     |      |_ train/
     |      |      |_ _annotations.coco.json
     |      |      |_ [image files]
     |      |_ valid/
     |      |      |_ _annotations.coco.json
     |      |      |_ [image files]
     |      |_ test/
     |             |_ _annotations.coco.json
     |             |_ [image files]
     |
     |_ original_yolo/       # YOLOv11 format dataset (for label combining)
     |      |_ data.yaml     # Class information
     |      |_ train/
     |      |      |_ labels/
     |      |      |_ [image files]
     |      |_ valid/
     |      |      |_ labels/
     |      |      |_ [image files]
     |      |_ test/
     |             |_ labels/
     |             |_ [image files]
     |
     |_ train.txt            # (Optional) List of training image filenames
     |_ valid.txt            # (Optional) List of validation image filenames
     |_ test.txt             # (Optional) List of test image filenames
```
Note: The `.txt` files are optional. If not provided, they will be generated automatically from the COCO annotations.


## Usage Modes

The pipeline supports multiple execution modes through the `--mode` argument:


### Mode 1: Full Pipeline (RECOMMENDED for first run)
```
docker exec -it segmentation_cnt python3 ./src/main.py --mode full
```
This executes the complete workflow:

Step 0: Dataset Setup (`filter_coco.py`)
  - Creates train/valid/test folder structure
  - Prepares COCO annotations for training
  - Copies images to working directories

Step 1: Train Mask R-CNN (`train.py`)
  - Trains Mask R-CNN for fish detection and classification
  - Uses COCO format annotations
  - Generates training report (PDF) in `./results/`
  - Saves model to `./weights/maskrcnn_model.pth`
  - Training time: ~several hours depending on epochs and dataset size

Step 2: Train Unified SAM2 (`train_sam2.py --train_unified`)
  - Trains a single SAM2 model on ALL fish classes
  - Learns to segment fish with high precision
  - Generates training report (PDF) in `./results/`
  - Saves model to `./weights/sam2_fish_unified.pth`
  - Training time: ~several hours depending on epochs

Step 3: Hybrid Inference (`infer_hybrid.py`)
  - Mask R-CNN detects fish and provides bounding boxes + class labels
  - SAM2 refines the segmentation masks within those boxes
  - Outputs YOLOv11 format polygon masks
  - Results saved to `./dataset/inference/labels/`
  - Visualization images saved to `./dataset/inference/images/`

Step 4: Create Upload Folder (`adapt2rbf.py`)
  - Combines inference results with original YOLO labels
  - Replaces classes 0 and 1 with improved masks
  - Preserves all other classes from original labels
  - Creates `./dataset/upload/` folder with:
    * Images (with 1px modification to avoid Roboflow duplicates)
    * Label files (`.txt` in YOLOv11 polygon format)
    * `data.yaml` (class information)
  - Ready to upload directly to Roboflow


### Mode 2: Setup Only
```
docker exec -it segmentation_cnt python3 ./src/main.py --mode setup
```
Runs only Step 0 (dataset setup). Use this to prepare folder structure before training.


### Mode 3: Train Mask R-CNN Only
```
docker exec -it segmentation_cnt python3 ./src/main.py --mode train-maskrcnn
```
Runs only Step 1. Use this to train or retrain the Mask R-CNN model with different parameters.


### Mode 4: Train SAM2 Only
```
docker exec -it segmentation_cnt python3 ./src/main.py --mode train-sam2
```
Runs only Step 2. Requires existing Mask R-CNN model. Use this to train or retrain SAM2 with different parameters.


### Mode 5: Inference Only
```
docker exec -it segmentation_cnt python3 ./src/main.py --mode inference
```
Runs Steps 3 and 4 only. Requires both trained models (Mask R-CNN and SAM2). Use this when you have trained models and just want to generate new predictions on test images.


## Configuration

All pipeline parameters can be configured in `src/main.py`:

Dataset Configuration:
- DATASET_PATH: Path to dataset folder
- CLASSES_TO_KEEP: List of class names
- TARGET_CLASSES_FOR_REPLACEMENT: YOLO class IDs to replace

Model Configuration:
- MASKRCNN_MODEL_PATH: Path to save/load Mask R-CNN model
- UNIFIED_SAM2_MODEL: Path to save/load unified SAM2 model
- SAM2_MODEL_ID: Hugging Face model ID for SAM2 base model

Training Parameters:
- NUM_EPOCHS: Number of training epochs (default: 40)
- BATCH_SIZE: Batch size for training (default: 1)
- LEARNING_RATE: Learning rate (default: 5e-6)
- IMG_SIZE: Input image size for SAM2 (default: 1024)
- GRADIENT_ACCUMULATION_STEPS: Effective batch size multiplier
- DICE_WEIGHT: Weight for Dice loss in SAM2 training
- MAX_INSTANCES_PER_IMAGE: Maximum objects to process per image

Inference Parameters:
- POINTS_PER_SIDE: Grid density for SAM2 (lower = fewer masks)
- PRED_IOU_THRESH: IoU threshold for mask filtering
- STABILITY_SCORE_THRESH: Confidence threshold
- MIN_MASK_REGION_AREA: Minimum mask size in pixels

Output Configuration:
- REPORT_OUTPUT_PATH: Folder for training reports
- OUTPUT_LABELS_FOLDER: Folder for inference labels
- OUTPUT_IMAGES_FOLDER: Folder for visualization images
- UPLOAD_FOLDER: Folder for Roboflow upload


## Output Structure

After running the full pipeline, your folder structure will be:
```
 dataset/
     |_ train/               # Working training images
     |_ valid/               # Working validation images
     |_ test/                # Working test images
     |_ original_coco/       # Original COCO dataset
     |_ original_yolo/       # Original YOLO dataset
     |_ inference/           # Inference results
     |      |_ images/       # Visualizations with masks overlaid
     |      |_ labels/       # Hybrid inference labels (YOLOv11 format)
     |      |_ labels_full/  # Combined labels (inference + original)
     |_ upload/              # Ready for Roboflow upload
            |_ [images]      # Test images (modified by 1px)
            |_ [labels]      # Final label files
            |_ data.yaml     # Class information

 weights/
     |_ maskrcnn_model.pth       # Trained Mask R-CNN
     |_ sam2_fish_unified.pth    # Trained SAM2

 results/
     |_ maskrcnn_training_report_[timestamp].pdf
     |_ sam2_training_report_[timestamp].pdf
```

## Utility Scripts

Additional utility scripts are available in `./src/utils/`:

Core Pipeline Scripts:
- `main.py`: Main orchestration script
- `filter_coco.py`: Dataset preparation
- `train.py`: Mask R-CNN training
- `train_sam2.py`: SAM2 training
- `infer_hybrid.py`: Hybrid inference
- `adapt2rbf.py`: Upload folder creation

Legacy/Alternative Scripts:
- `coco2yolo11.py`: Mask R-CNN inference (use if you want to see Mask R-CNN-only results)
- `infer_sam2.py`: SAM2-only inference (legacy, not recommended)


## Typical Workflow

1. First Time Setup:
   - Download dataset from Roboflow (both COCO and YOLO formats)
   - Set up folder structure as described above
   - Configure `.env` file with dataset path
   - Run: `docker exec -it segmentation_cnt python3 ./src/main.py --mode full`
   - Wait several hours for training to complete

2. Upload to Roboflow:
   - Navigate to `./dataset/upload/`
   - Upload all contents to Roboflow
   - Review and correct any remaining errors
   - Export improved dataset

3. Iterative Improvement:
   - Download improved dataset from Roboflow (both in COCO and Yolo format)
   - Re-run training with: `--mode full`
   - Generate new predictions
   - Upload back to Roboflow
   - Repeat until satisfied with quality


## Notes and Recommendations

- The hybrid approach (v3.0.0) provides significantly better results than Mask R-CNN alone
- Training is computationally intensive; GPU is strongly recommended
- First run will download SAM2 base model from Hugging Face (~900MB)
- The 1px modification to images prevents Roboflow duplicate detection while maintaining original filenames
- Both COCO and YOLO format datasets are required (not optional)
- Test set images are modified and uploaded, not train/valid (to avoid data leakage)


## Troubleshooting

If inference fails:
- Check that both models exist in `./weights/`
- Verify folder structure matches the specification exactly
- Ensure COCO and YOLO datasets are properly aligned

If training is slow:
- Reduce NUM_EPOCHS for testing
- Reduce IMG_SIZE for SAM2
- Ensure GPU is being used (check CUDA availability)

If masks are too fragmented:
- Increase MIN_MASK_REGION_AREA
- Increase PRED_IOU_THRESH and STABILITY_SCORE_THRESH
- Reduce POINTS_PER_SIDE

If masks are missing small fish:
- Decrease MIN_MASK_REGION_AREA
- Increase POINTS_PER_SIDE
- Lower PRED_IOU_THRESH and STABILITY_SCORE_THRESH


## Version History

v1.0.0: Mask R-CNN only approach
v2.0.0: SAM2 only approach (experimental, not recommended)
v3.0.0: Hybrid Mask R-CNN + SAM2 approach (current, recommended)


## License

Internal SRV group project. Contact project maintainer for usage permissions.
