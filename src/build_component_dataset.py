#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPRI(3 线路) → 配网部件检测数据集 v2 (dataset/yolo_component/)

统一 5 类:
  0 insulator    绝缘子(EPRI 多边形→紧框)
  1 pole         电杆(EPRI 整杆多边形)
  2 crossarm     横担
  3 cutout       跌落式熔断器
  4 transformer  配电变压器

设计:
- EPRI 折线类(conductor/other_wire/guy_wires)不转检测框(全图对角线框无意义), background_structure 丢弃
- 划分按线路地理隔离: circuit5+7 → train, circuit10 → val
- obscured/truncated 属性不剔除(保留全量)
- ICARUS 杆顶已按用户决策删除(2026-08-13, 俯视杆顶与我方场景不符)
"""

import ast
import csv
import os
from collections import Counter
from pathlib import Path

ROOT = Path('dataset')
OUT = ROOT / 'yolo_component'
NAMES = ['insulator', 'pole', 'crossarm', 'cutout', 'transformer']
CLS = {'insulator': 0, 'pole': 1, 'crossarm': 2, 'cutouts': 3, 'transformers': 4}

stats = Counter()
img_stats = Counter()


def link_img(src: Path, tag: str, split: str) -> str:
    dst = OUT / 'images' / split / f'{tag}_{src.name.replace(" ", "_").replace("(", "").replace(")", "")}'
    if not dst.exists():
        os.symlink(src.resolve(), dst)
    return dst.stem


def add_epri():
    csv.field_size_limit(10 ** 9)
    rows = {r[1]: r[0] for r in csv.reader(open(ROOT / 'external/epri/Overhead-Distribution-Labels.csv'))
            if len(r) > 1 and r[1] != 'External ID'}
    for circuit, split in (('circuit3', 'train'), ('circuit4', 'train'), ('circuit5', 'train'),
                           ('circuit6', 'train'), ('circuit7', 'train'), ('circuit8', 'train'),
                           ('circuit10', 'val'), ('circuit13b', 'val')):
        d = ROOT / 'external/epri' / circuit
        for f in sorted(os.listdir(d)):
            if f not in rows:
                continue
            stem = link_img(d / f, 'epri', split)
            lines = []
            for o in ast.literal_eval(rows[f])['objects']:
                if o['value'] not in CLS or 'polygon' not in o:
                    continue
                # 图像尺寸从多边形外推不行, 需要真实宽高
                lines.append(o)  # 先收集, 尺寸统一二次处理
            _write_epri_labels(d / f, stem, split, lines)
            img_stats[('epri', split)] += 1


def _write_epri_labels(img_path, stem, split, objs):
    from PIL import Image
    W, H = Image.open(img_path).size
    lines = []
    for o in objs:
        xs = [p['x'] for p in o['polygon']]
        ys = [p['y'] for p in o['polygon']]
        x1, x2, y1, y2 = max(0, min(xs)), min(W, max(xs)), max(0, min(ys)), min(H, max(ys))
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            continue
        c = CLS[o['value']]
        lines.append(f'{c} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {w/W:.6f} {h/H:.6f}')
        stats[('epri', c)] += 1
    (OUT / 'labels' / split / f'{stem}.txt').write_text('\n'.join(lines))


for d in ('images', 'labels'):
    for s in ('train', 'val'):
        (OUT / d / s).mkdir(parents=True, exist_ok=True)

add_epri()

yaml_lines = [f'path: {OUT.resolve()}', 'train: images/train', 'val: images/val', '', 'names:']
yaml_lines += [f'  {i}: {n}' for i, n in enumerate(NAMES)]
(OUT / 'dataset.yaml').write_text('\n'.join(yaml_lines) + '\n')

n_tr = sum(v for (s, sp), v in img_stats.items() if sp == 'train')
n_va = sum(v for (s, sp), v in img_stats.items() if sp == 'val')
print(f'图像: train {n_tr} / val {n_va}')
print(f'\n{"类":<14}{"实例":>8}')
for i, n in enumerate(NAMES):
    print(f'{i} {n:<12}{stats[("epri", i)]:>8}')
print(f'\n输出: {OUT}/dataset.yaml')
