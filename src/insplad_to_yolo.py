#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InsPLAD-det (COCO) -> YOLO 数据集转换
输入:dataset/external/insplad/det/{train,val}/*.jpg + annotations/instances_{train,val}.json
输出:dataset/yolo_insplad/ (symlink 图像 + YOLO labels + dataset.yaml), 并打印类别统计
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict

SRC = Path('dataset/external/insplad/det')
OUT = Path('dataset/yolo_insplad')


def convert(split):
    with open(SRC / 'annotations' / f'instances_{split}.json') as f:
        coco = json.load(f)
    cats = sorted(coco['categories'], key=lambda c: c['id'])
    cat_to_yolo = {c['id']: i for i, c in enumerate(cats)}
    img_info = {i['id']: i for i in coco['images']}
    anns_of = defaultdict(list)
    for a in coco['annotations']:
        anns_of[a['image_id']].append(a)

    img_out, lbl_out = OUT / 'images' / split, OUT / 'labels' / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    n = 0
    for img_id, info in img_info.items():
        src = (SRC / split / info['file_name']).resolve()
        if not src.exists():
            continue
        dst = img_out / info['file_name']
        if not dst.exists():
            os.symlink(src, dst)
        W, H = info['width'], info['height']
        lines = []
        for a in anns_of.get(img_id, []):
            x, y, w, h = a['bbox']
            xc, yc = (x + w / 2) / W, (y + h / 2) / H
            lines.append(f'{cat_to_yolo[a["category_id"]]} {xc:.6f} {yc:.6f} {w / W:.6f} {h / H:.6f}')
            counts[a['category_id']] += 1
        (lbl_out / f'{Path(info["file_name"]).stem}.txt').write_text('\n'.join(lines))
        n += 1
    return cats, counts, n, img_info


cats_tr, cnt_tr, n_tr, info_tr = convert('train')
_, cnt_val, n_val, info_val = convert('val')

yaml_lines = [f'path: {OUT.resolve()}', 'train: images/train', 'val: images/val', '', 'names:']
yaml_lines += [f'  {i}: {c["name"]}' for i, c in enumerate(cats_tr)]
(OUT / 'dataset.yaml').write_text('\n'.join(yaml_lines) + '\n')

print(f'train: {n_tr} 张 | val: {n_val} 张 | {len(cats_tr)} 类')
print(f'\n{"类别":<28} {"train":>6} {"val":>6}')
for c in cats_tr:
    print(f'{c["name"]:<28} {cnt_tr[c["id"]]:>6} {cnt_val[c["id"]]:>6}')

# 分辨率分布
import numpy as np
ws = [i['width'] for i in info_tr.values()]
hs = [i['height'] for i in info_tr.values()]
print(f'\n分辨率: {min(ws)}x{min(hs)} ~ {max(ws)}x{max(hs)}, 中位 {int(np.median(ws))}x{int(np.median(hs))}')
print(f'输出: {OUT}/dataset.yaml')
