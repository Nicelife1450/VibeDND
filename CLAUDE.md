# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VibeDND is a COCO-format dataset for **配网无人机巡检缺陷检测** (power distribution network drone inspection defect detection). Current dataset contains 3665 images with 6845 annotations across 72 defect categories.

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Build dataset from raw inspection reports
```bash
python3 src/build_coco_dataset.py
```
The script scans `巡检报告/` directories, detects red circle annotations via HSV color detection, and generates `dataset/annotations.json` and `dataset/defect_severity.json`.

### Convert COCO to YOLO format
```bash
python3 src/coco_to_yolo.py [--annotations dataset/annotations_merged.json] [--output dataset/yolo]
```
Converts COCO annotations to YOLO format with train/val split (8:2). Merges categories with < 10 annotations into "其他缺陷". Accepts any COCO JSON via `--annotations` (defaults to `dataset/annotations.json`); image files are resolved by searching `images/`, `augmented_images/`, `augmented_rare/` in that order.

### Offline augmentation for rare categories
```bash
# Geometric/photometric augmentation (3 variants per rare image, written into dataset/images/)
python3 src/offline_rare_augmentation.py --rare-threshold 50 --variants 3

# Copy-paste rare-category patches onto background images (output: dataset/augmented_images/)
python3 src/copy_paste_augmentation.py --rare-threshold 50 --target-per-category 200
```
Both target categories with < `rare-threshold` annotations and output a new COCO JSON (original + added entries), seeded with 42. Feed the result back through `coco_to_yolo.py --annotations <output.json>` to build an augmented YOLO dataset.

### Pseudo-labeling (semi-supervised)
```bash
# 1. Teacher model labels unannotated images found in 巡检报告/缺陷原图/
python3 src/generate_pseudolabels.py --model runs/detect/train/weights/best.pt --conf 0.6

# 2. Merge pseudo labels with ground truth
python3 src/merge_pseudolabels.py
```
Pseudo labels map YOLO class indices back to COCO category IDs via class-name matching. Pseudo images are copied to `dataset/pseudo_images/` — note this dir is NOT searched by `coco_to_yolo.py` by default.

### Visualize annotations
```bash
python3 src/visualize_dataset.py --num 10
```
Options: `--dataset` (default `dataset/annotations.json`), `--output` (default `dataset/visualizations`), `--num` (sample count).

## Architecture

```
src/
├── build_coco_dataset.py        # Dataset builder - scans 巡检报告, detects red circles, outputs clean COCO format
├── coco_to_yolo.py              # COCO → YOLO converter with train/val split (--annotations/--output args)
├── offline_rare_augmentation.py # Geometric/photometric augmentation of images containing rare categories
├── copy_paste_augmentation.py   # Copy-paste rare-category patches onto backgrounds (IoU-checked placement)
├── generate_pseudolabels.py     # Teacher-model inference over unannotated 巡检报告 images → pseudo COCO
├── merge_pseudolabels.py        # Merge pseudo COCO into ground-truth COCO (re-indexes IDs)
└── visualize_dataset.py   # Annotation visualizer - draws bounding boxes on images

dataset/
├── annotations.json        # COCO format: images, annotations, categories (72 categories)
├── defect_severity.json    # defect_type → severity mapping (一般/严重/危急缺陷)
├── images/                 # Renamed images (DND_xxxxxxxx.jpg) + AUG_RARE_* offline augmentations
├── augmented_images/       # AUG_CP_* copy-paste augmentations (created on demand)
├── pseudo_images/          # PSEUDO_* copies of teacher-labeled images (created on demand)
├── annotations_*.json      # Augmented/merged COCO variants (copypaste, augmented_rare, merged, pseudo)
└── yolo/                   # YOLO format dataset (51 categories after merging)
    ├── dataset.yaml         # YOLO training config
    ├── images/
    │   ├── train/          # Symlinks to ../images/*.jpg
    │   └── val/
    └── labels/
        ├── train/           # YOLO format .txt files
        └── val/

巡检报告/                    # Raw inspection data (not in repo)
└── 2024/
    └── {line_name}/
        ├── 缺陷原图/         # Original images
        └── 缺陷圈图/         # Annotated images with red circles
```

**Dataset builder flow**: Scan inspection folders → match annotated image to original → parse defect metadata from filename → detect red circles via HSV → output COCO format with renamed images.

**Training-data pipeline**: start from `annotations.json` → optionally run rare-category augmentation and/or pseudo-label merging to produce a new COCO JSON → `coco_to_yolo.py --annotations <json>` → train YOLO. Augmentation scripts append entries (IDs continue from `max(id) + 1`, seed 42) rather than modifying the input; new images are named `AUG_RARE_*`, `AUG_CP_*`, `PSEUDO_*`.

**Annotation format**: Bounding boxes detected via HSV red color detection on annotated images, not from original filenames.

## Key Patterns

- Filename format: `{line_name}_{pole_id}_{defect_desc}_{severity}.jpg`
- Image dimensions: 4000x3000 pixels
- HSV red detection: ranges `([0,100,100],[10,255,255])` and `([160,100,100],[180,255,255])`
- Config class in `build_coco_dataset.py` controls detection thresholds
- Chinese fonts required for visualization: tries PingFang.ttc, then STHeiti Light.ttc, falls back to default

## Data Loading

```python
from pycocotools.coco import COCO
coco = COCO('dataset/annotations.json')

# Or directly
import json
with open('dataset/annotations.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
```

## Notes

- Images are renamed to `DND_xxxxxxxx.jpg` format; original filename preserved in `original_name` field
- Some annotated images may lack matching originals or detectable red circles
- Chinese characters must be preserved in all paths and filenames
- `build_coco_dataset.py` includes filename parsing that strips parenthetical details, normalizes defect names, and filters malformed categories
- YOLO dataset uses symlinks - does not duplicate images

## YOLO Training

```bash
# Activate virtual environment
source venv/bin/activate

# Train YOLOv8m (requires GPU)
yolo detect train model=yolov8m.pt data=dataset/yolo/dataset.yaml epochs=100 imgsz=1280 batch=8

# Validate
yolo detect val model=runs/detect/train/weights/best.pt data=dataset/yolo/dataset.yaml

# Predict
yolo detect predict model=runs/detect/train/weights/best.pt source=dataset/images/
```

## Sync to Server

```bash
rsync -avz --exclude='dataset/images' --exclude='venv' --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='runs' -e "ssh -p 1172" ./ mac247:/home/huyue/huyue-project/VibeDND
```
