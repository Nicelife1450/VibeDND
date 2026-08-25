#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPRI 部件检测器 → DND 全量伪标注 (里程碑 3 收尾)

- 对 dataset/images/DND_*.jpg 全量推理(imgsz=1280, conf=0.25 全收)
- 高置信(≥0.5)自动收入 dataset/yolo_dnd_pseudo/ (symlink 图像 + YOLO 标签)
- 全部检出(含置信度)存 experiments/pseudo_labels_raw/ 供后续调阈值
- 打印每类统计 + 零检出图像清单
"""

from collections import Counter
from pathlib import Path

from ultralytics import YOLO

MODEL = 'runs/detect/r9_epri_component/weights/best.pt'
NAMES = ['insulator', 'pole', 'crossarm', 'cutout', 'transformer']
ACCEPT = 0.5
OUT = Path('dataset/yolo_dnd_pseudo')
RAW = Path('experiments/pseudo_labels_raw')
RAW.mkdir(exist_ok=True, parents=True)

model = YOLO(MODEL)
imgs = sorted(Path('dataset/images').glob('DND_*.jpg'))
print(f'共 {len(imgs)} 张 DND 原图')

(OUT / 'images').mkdir(parents=True, exist_ok=True)
(OUT / 'labels').mkdir(parents=True, exist_ok=True)

stats = Counter()
zero_det = []
import os

for i, img in enumerate(imgs):
    r = model.predict(str(img), imgsz=1280, conf=0.25, verbose=False, device=0)[0]
    raw_lines, acc_lines = [], []
    for b in r.boxes:
        c = int(b.cls)
        conf = float(b.conf)
        x, y, w, h = b.xywhn[0].tolist()
        raw_lines.append(f'{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.4f}')
        stats[('raw', c)] += 1
        if conf >= ACCEPT:
            acc_lines.append(f'{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}')
            stats[('acc', c)] += 1
    (RAW / f'{img.stem}.txt').write_text('\n'.join(raw_lines))
    if not raw_lines:
        zero_det.append(img.name)
    # 伪标注集: 只收有≥0.5 检出的图
    if acc_lines:
        dst = OUT / 'images' / img.name
        if not dst.exists():
            os.symlink(img.resolve(), dst)
        (OUT / 'labels' / f'{img.stem}.txt').write_text('\n'.join(acc_lines))
    if (i + 1) % 500 == 0:
        print(f'  {i+1}/{len(imgs)} ...')

yaml_lines = [f'path: {OUT.resolve()}', 'train: images', 'val: images', '', 'names:']
yaml_lines += [f'  {i}: {n}' for i, n in enumerate(NAMES)]
(OUT / 'dataset.yaml').write_text('\n'.join(yaml_lines) + '\n')

n_acc_img = len(list((OUT / 'labels').iterdir()))
print(f'\n=== 完成 ===')
print(f'有 ≥{ACCEPT} 检出的图像: {n_acc_img}/{len(imgs)} ({n_acc_img/len(imgs)*100:.0f}%)')
print(f'零检出图像: {len(zero_det)} 张 → experiments/pseudo_zero_det.txt')
Path('experiments/pseudo_zero_det.txt').write_text('\n'.join(zero_det))
print(f'\n{"类":<12}{"≥0.25":>8}{"≥0.5":>8}')
for i, n in enumerate(NAMES):
    print(f'{n:<12}{stats[("raw", i)]:>8}{stats[("acc", i)]:>8}')
print(f'\n输出: {OUT}/ (raw 含置信度: {RAW}/)')
