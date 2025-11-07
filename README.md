
# Fish Segmentation Pipeline (Mask R-CNN + SAM2)

## Overview
This project provides a modular, configuration-driven pipeline for fish instance segmentation using Mask R-CNN for detection/classification and SAM2 for mask refinement, with consistent dataset setup and upload packaging for Roboflow.

## Key Features
- Single dataset structure with three COCO annotation files per split: one multi-class for Mask R-CNN and one binary per SAM2 class.
- Config-driven execution with modes for setup, training (per model), inference, and upload packaging.
- Separate inference outputs: Mask R-CNN writes to `dataset/inference_maskrcnn/`, Hybrid writes to `dataset/inference/`.
- Upload folder creation that merges inferred labels with original labels and prepares `data.yaml` for Roboflow.

## Requirements

The repository is in a container that provides all the requirements. You
should have cuda 4.2 (or higher installed). It is used like all the others from the
SRV group (see other previous repos for more details, for example
'Docker_ROSpy311')

Once connected and located in the 'repo' folder, just run `docker compose up -d`
to start the container. It is a development environment, so the container is deleted
when stopped (`docker compose down`) and all your saved files remain at the repo folder.

## Dataset Structure
Input (expected):
```
dataset/
└── original_coco/
    ├── train/
    │   ├── _annotations.coco.json
    │   └── *.jpg|*.png
    ├── valid/
    │   ├── _annotations.coco.json
    │   └── *.jpg|*.png
    └── test/
        ├── _annotations.coco.json
        └── *.jpg|*.png
```
After setup, images are copied once and three annotation files are created per split:
```
dataset/
├── train/
│   ├── _annotations.coco.json          # Multi-class (for Mask R-CNN)
│   ├── _annotations_class0.coco.json   # Class 0 only (for SAM2)
│   └── _annotations_class1.coco.json   # Class 1 only (for SAM2)
├── valid/
└── test/
```

## Configuration
Edit `src/config.yaml` to control the pipeline.

Important keys:
- `dataset.classes_to_keep`: class names as strings, e.g., `['Chromis chromis', 'Coris julis']`
- `dataset.class_mapping`: COCO category ID → model class index, e.g., `{1: 0, 2: 1}`
- `dataset.class_names_mapping`: model class index → human name, e.g., `{0: 'Chromis chromis', 1: 'Coris julis'}`
- `pipeline.*`: controls which steps run in `--mode full` and which inference path (`maskrcnn` or `hybrid`)
- `output.*`: folders for inference artifacts and upload packaging

Example:
```
pipeline:
  train_maskrcnn: true
  train_sam2: true
  inference_mode: 'hybrid'     # 'maskrcnn' or 'hybrid'
  create_upload_folder: true

dataset:
  dataset_path: "dataset"
  classes_to_keep: ['Chromis chromis', 'Coris julis']

output:
  inference_maskrcnn_folder: "dataset/inference_maskrcnn"
  output_maskrcnn_labels: "dataset/inference_maskrcnn/labels"
  output_maskrcnn_images: "dataset/inference_maskrcnn/images"
  inference_folder_path: "dataset/inference"
  output_labels_folder: "dataset/inference/labels"
  output_images_folder: "dataset/inference/images"
  upload_maskrcnn_folder: "dataset/upload_maskrcnn"
  upload_folder: "dataset/upload"
```

## Setup (Create Annotation Files)
Create `train.txt/valid.txt/test.txt` if missing, copy images once to `dataset/{split}`, and emit three COCO annotation files per split:
```
python src/main.py --mode setup
```
You should see `_annotations.coco.json`, `_annotations_class0.coco.json`, and `_annotations_class1.coco.json` under each of `train/valid/test`.

## Training
### Mask R-CNN (multi-class)
Uses `dataset/{split}/_annotations.coco.json`:
```
python src/main.py --mode train-maskrcnn
```

### SAM2 (binary per class)
Trains one SAM2 model per class using `_annotations_class{index}.coco.json` automatically:
```
# Train all classes
python src/main.py --mode train-sam2

# Train a single class (e.g., class 0)
python src/main.py --mode train-sam2-class --class_index 0
```

## Inference
### Mask R-CNN only
Writes labels and visualizations to `dataset/inference_maskrcnn/`:
```
python src/main.py --mode inference-maskrcnn
```

### Hybrid (Mask R-CNN detection + SAM2 refinement)
Runs when `pipeline.inference_mode: 'hybrid'` and you invoke the full pipeline:
```
python src/main.py --mode full
```
Outputs go to `dataset/inference/` under `labels/` and `images/`.

## Upload Folder Creation
Creates an upload folder (images + merged `labels_full` + `data.yaml`) ready for Roboflow:
```
python src/main.py --mode upload-folder
```
- If `inference_mode: 'maskrcnn'`, upload → `dataset/upload_maskrcnn/`
- If `inference_mode: 'hybrid'`, upload → `dataset/upload/`

## Full Pipeline
Executes setup, training, inference, and upload according to `config.yaml`:
```
python src/main.py --mode full
```

## Outputs
- `dataset/inference_maskrcnn/labels` and `images`: Mask R-CNN results
- `dataset/inference/labels` and `images`: Hybrid results
- `dataset/*/labels_full`: merged labels used for upload packaging
- `dataset/upload_maskrcnn/` or `dataset/upload/`: packaged dataset with images, labels, and `data.yaml`

## Tips and Gotchas
- `dataset.classes_to_keep` must be class names from your COCO categories (not numeric IDs)
- If `train.txt/valid.txt/test.txt` are missing, setup creates them from the `original_coco` split folders
- Filenames in `.txt` can include commas or extensions—the setup script normalizes them
- Ensure exactly one `_annotations.coco.json` per split exists in `original_coco/{train,valid,test}`
- For hybrid, ensure SAM2 model paths exist before running `--mode full` or `--mode upload-folder`

## Commands Quick Reference
- Setup: `python src/main.py --mode setup`
- Train Mask R-CNN: `python src/main.py --mode train-maskrcnn`
- Train all SAM2: `python src/main.py --mode train-sam2`
- Train one SAM2: `python src/main.py --mode train-sam2-class --class_index N`
- Inference (Mask R-CNN): `python src/main.py --mode inference-maskrcnn`
- Upload folder: `python src/main.py --mode upload-folder`
- Full pipeline: `python src/main.py --mode full`

## File Map
- `src/utils/filter_coco.py`: creates per-split multi-class and class-specific COCO annotations without duplicating images
- `src/utils/train.py`: Mask R-CNN trainer consuming multi-class COCO annotations
- `src/utils/train_sam2.py`: SAM2 trainer consuming per-class COCO annotations based on `TARGET_CLASS_INDEX`
- `src/main.py`: orchestrates setup, training, inference, and upload using `config.yaml`
