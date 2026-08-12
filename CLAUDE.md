# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VibeDND is a COCO-format dataset for **配网无人机巡检缺陷检测** (power distribution network drone inspection defect detection): 3665 images (4000x3000), 72 raw defect categories.

## 当前路线(2026-08,两阶段 pipeline)

**任务路线权威文档: `docs/两阶段路线.md`** — 先读它再动手。

核心决策:
- **自有标注已全部废弃**(2026-08-12 用户确认, 含验收 GT; 移至 `experiments/archive/annotations/`); 验收协议待重定义(见路线文档)
- 新 pipeline: 公开数据集预训练(InsPLAD/EPRI/UPID/IDID) → 部件检测器 + 状态分类器级联(部件状态型 11 类) + 直接检测分支(场景异物型 5 类)
- 参考调研: `docs/配电网缺陷检测开源数据集调研.md`, `docs/reference.pdf` (Energies 2026 综述)

## 历史结论(2026-07, R1-R7 实验)

- 旧路线冠军: `runs/detect/r3_yolo11n_full/weights/best.pt` — 冻结 val mAP50=0.088 (TTA 0.096)
- 已证伪: 模型容量/长训/高分辨率/切片/紧框重标注(YOLO-World)/教师补标/copy-paste/SAM 精修 — 全部无效, 细节见 `experiments/log.md` + `experiments/final_report.md`
- 失败实验代码与衍生标注已归档至 `experiments/archive/`(可再生, 勿恢复用于新路线)
- R8 预训练验证: yolo11n 在 InsPLAD 上 mAP50=0.916 (`runs/detect/r8_insplad_yolo11n2/`), 部件特征跨域有效(特写裁剪下 0.81 置信度认出中国配网绝缘子)

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### COCO → YOLO conversion
```bash
python3 src/coco_to_yolo.py [--annotations <json>] [--output <dir>]       # 通用转换(自带 8:2 划分)
python3 src/build_yolo_from_split.py --train-annotations <json> --val-annotations <json> \
  --split <split.json> --train-split-ids --output <dir>                   # 按指定划分构建
```

### InsPLAD (external pretraining dataset)
```bash
python3 src/insplad_to_yolo.py   # dataset/external/insplad/det → dataset/yolo_insplad/ (自带官方 train/val)
```

### Visualize annotations
```bash
python3 src/visualize_dataset.py --num 10
```

## Architecture

```
src/
├── coco_to_yolo.py           # COCO → YOLO 通用转换
├── build_yolo_from_split.py  # 按指定 split JSON 构建 YOLO 数据集(symlink)
├── insplad_to_yolo.py        # InsPLAD-det COCO → YOLO
├── sliced_eval.py            # SAHI 式切片推理验收(高分辨率评估工具)
└── visualize_dataset.py      # 标注可视化

(旧管线脚本 build_coco_dataset / build_reduced_dataset / clean_red_circles 已归档至 experiments/archive/scripts/)

dataset/
├── defect_severity.json      # 缺陷严重度映射
├── images/                   # 清洗后原图 DND_xxxxxxxx.jpg (38G)
├── images_contaminated_backup/  # 清洗前备份 (18G)
├── external/insplad/         # InsPLAD 原始包 + det/ 解压 (15G)
└── yolo_insplad/             # InsPLAD YOLO 格式(官方 train/val)

experiments/
├── log.md, final_report.md   # R1-R8 实验日志与终版报告
└── archive/                  # 失败实验代码+衍生标注(归档, 勿用于新路线)

docs/
├── 两阶段路线.md              # ★ 当前任务路线权威文档
├── 配电网缺陷检测开源数据集调研.md
└── reference.pdf             # Energies 2026 电力巡检公开数据集综述
```

## Key Patterns

- Image dimensions: 4000x3000; images cleaned of red-circle contamination (Telea inpaint, ring-only filtering)
- HSV red detection: ranges `([0,100,100],[10,255,255])` and `([160,100,100],[180,255,255])`
- ultralytics 的 `device=` 参数会覆盖 `CUDA_VISIBLE_DEVICES`, 直接用 `device=N` 即可
- ultralytics SAM 框提示需显式 `conf=0.01`(mask 分数低于默认 0.25 会被静默丢弃)
- Chinese characters must be preserved in all paths and filenames

## YOLO Training

```bash
source venv/bin/activate   # 或 ~/miniconda/envs/yolo
yolo detect train model=yolo11n.pt data=<yaml> epochs=100 imgsz=1280 batch=16 device=0
```

## Sync to Server

```bash
rsync -avz --exclude='dataset/images' --exclude='dataset/external' --exclude='venv' --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='runs' -e "ssh -p 1172" ./ mac247:/home/huyue/huyue-project/VibeDND
```
