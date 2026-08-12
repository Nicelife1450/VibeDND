#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练集切片脚本 (SAHI 式 tiling)
功能:将 train 图像切成重叠切片, 小目标相对放大; val 保持完整原图
输出:dataset/tiles/TILE_xxxxxxxx.jpg + dataset/annotations_tiles.json
"""

import json
import cv2
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def tile_image(img, anns, tile, stride, min_vis=0.4, min_side=6):
    """切一张图, 返回 [(tile_img, [新标注...]), ...]。bbox 裁剪到片内。"""
    h, w = img.shape[:2]
    results = []
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            # 边缘不足 tile 大小的片, 平移使切片大小一致
            ty0, tx0 = y1 - tile, x1 - tile
            if ty0 < 0 or tx0 < 0:
                # 图比 tile 小, 直接整图
                ty0, tx0, y1, x1 = 0, 0, h, w
            new_anns = []
            for a in anns:
                bx, by, bw, bh = a['bbox']
                ix1 = max(bx, tx0)
                iy1 = max(by, ty0)
                ix2 = min(bx + bw, x1)
                iy2 = min(by + bh, y1)
                iw, ih = ix2 - ix1, iy2 - iy1
                if iw <= 0 or ih <= 0:
                    continue
                if iw * ih < min_vis * bw * bh:  # 可见面积不足
                    continue
                if iw < min_side or ih < min_side:
                    continue
                nb = [ix1 - tx0, iy1 - ty0, iw, ih]
                new_anns.append({**a, 'bbox': nb, 'area': iw * ih})
            tile_img = img[ty0:y1, tx0:x1]
            results.append((tile_img, new_anns))
    return results


def main():
    parser = argparse.ArgumentParser(description='训练集切片 (SAHI tiling)')
    parser.add_argument('--annotations', type=str, default='dataset/annotations_reduced.json')
    parser.add_argument('--split', type=str, default='dataset/split_reduced.json')
    parser.add_argument('--tile', type=int, default=1280)
    parser.add_argument('--stride', type=int, default=1024)
    parser.add_argument('--keep-empty', type=float, default=0.3, help='无标注切片保留比例')
    parser.add_argument('--min-vis', type=float, default=0.4, help='bbox 可见面积比例下限')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='dataset/annotations_tiles.json')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ann_path = Path(args.annotations)
    img_dir = ann_path.parent / 'images'
    tile_dir = ann_path.parent / 'tiles'
    tile_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    with open(args.split, 'r', encoding='utf-8') as f:
        split = json.load(f)
    train_ids = set(split['train'])

    img_info = {img['id']: img for img in coco['images']}
    img_to_anns = defaultdict(list)
    for a in coco['annotations']:
        img_to_anns[a['image_id']].append(a)

    out_images, out_annotations = [], []
    next_img_id = 1
    next_ann_id = 1
    n_empty_kept = 0
    n_skipped = 0

    train_list = sorted(train_ids)
    for idx, img_id in enumerate(train_list, 1):
        info = img_info[img_id]
        img = cv2.imread(str(img_dir / info['file_name']))
        if img is None:
            print(f'无法读取: {info["file_name"]}')
            n_skipped += 1
            continue
        tiles = tile_image(img, img_to_anns.get(img_id, []),
                           args.tile, args.stride, args.min_vis)
        for tile_img, tile_anns in tiles:
            if not tile_anns and random.random() > args.keep_empty:
                continue
            if not tile_anns:
                n_empty_kept += 1
            fname = f'TILE_{next_img_id:08d}.jpg'
            cv2.imwrite(str(tile_dir / fname), tile_img)
            th, tw = tile_img.shape[:2]
            out_images.append({
                'id': next_img_id,
                'file_name': fname,
                'original_name': f'tile_of_{info["file_name"]}',
                'width': tw,
                'height': th,
            })
            for a in tile_anns:
                out_annotations.append({
                    'id': next_ann_id,
                    'image_id': next_img_id,
                    'category_id': a['category_id'],
                    'bbox': [float(v) for v in a['bbox']],
                    'area': float(a['area']),
                    'iscrowd': 0,
                })
                next_ann_id += 1
            next_img_id += 1
        if idx % 200 == 0:
            print(f'  {idx}/{len(train_list)} 张原图 -> {len(out_images)} 切片')

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'info': coco.get('info', {}),
            'images': out_images,
            'annotations': out_annotations,
            'categories': coco['categories'],
        }, f, ensure_ascii=False, indent=2)

    print(f'\n切片完成! tile={args.tile} stride={args.stride}')
    print(f'切片图像: {len(out_images)} (含空片 {n_empty_kept}, 跳过 {n_skipped})')
    print(f'切片标注: {len(out_annotations)}')
    print(f'输出: {args.output}')


if __name__ == '__main__':
    main()
