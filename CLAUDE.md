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

### Clean/normalization dataset categories
```bash
python3 src/cleanup_dataset.py
```
Normalizes category names, merges fragmented categories, fixes parsing errors. Run after building a new dataset.

### Convert COCO to YOLO format
```bash
python3 src/coco_to_yolo.py
```
Converts COCO annotations to YOLO format with train/val split (8:2). Merges categories with < 10 annotations into "其他缺陷".

### Visualize annotations
```bash
python3 src/visualize_dataset.py --num 10
```
Options: `--dataset` (default `dataset/annotations.json`), `--output` (default `dataset/visualizations`), `--num` (sample count).

## Architecture

```
src/
├── build_coco_dataset.py   # Dataset builder - scans 巡检报告, detects red circles, outputs COCO format
├── cleanup_dataset.py      # Category normalizer - merges fragments, fixes parsing errors
├── coco_to_yolo.py         # COCO → YOLO format converter with train/val split
└── visualize_dataset.py   # Annotation visualizer - draws bounding boxes on images

dataset/
├── annotations.json        # COCO format: images, annotations, categories (72 categories)
├── defect_severity.json    # defect_type → severity mapping (一般/严重/危急缺陷)
├── images/                 # Renamed images (DND_xxxxxxxx.jpg)
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

**Dataset builder flow**: Scan inspection folders → match annotated image to original → parse filename for defect metadata → detect red circles via HSV → output COCO format with renamed images.

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
- `build_coco_dataset.py` includes filename parsing that strips parenthetical details and filters malformed categories
- Run `cleanup_dataset.py` after building to ensure category consistency
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

## Project Structure

```
VibeDND/
├── src/
│   ├── build_coco_dataset.py    # Build COCO dataset from raw reports
│   ├── cleanup_dataset.py       # Normalize/merge categories
│   ├── coco_to_yolo.py         # Convert COCO → YOLO format
│   └── visualize_dataset.py     # Visualize annotations
├── dataset/
│   ├── annotations.json         # COCO format (72 categories)
│   ├── defect_severity.json    # Severity mapping
│   ├── images/                 # Original images (DND_xxxxxxxx.jpg)
│   └── yolo/                   # YOLO format (51 categories after merging)
│       ├── dataset.yaml
│       ├── images/train, val   # Symlinks
│       └── labels/train, val   # .txt files
├── venv/                       # Python virtual environment
└── CLAUDE.md                   # This file
```
