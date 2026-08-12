#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按冻结划分构建 YOLO 数据集
功能:train 用切片 COCO, val 用完整原图 COCO (按 split JSON 的 val 列表), 输出 YOLO 格式
图像使用 symlink, 不复制。
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict


def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    return ((x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h)


def write_split(coco, img_ids, img_search_dirs, out_img_dir, out_lbl_dir, cat_to_yolo):
    img_info = {i['id']: i for i in coco['images']}
    anns_of = defaultdict(list)
    for a in coco['annotations']:
        anns_of[a['image_id']].append(a)

    n_written = 0
    for img_id in img_ids:
        info = img_info[img_id]
        src = None
        for d in img_search_dirs:
            p = d / info['file_name']
            if p.exists():
                src = p.resolve()
                break
        if src is None:
            print(f'Warning: 找不到图像 {info["file_name"]}')
            continue
        dst = out_img_dir / info['file_name']
        if not dst.exists():
            os.symlink(src, dst)

        lines = []
        for a in anns_of.get(img_id, []):
            yc = cat_to_yolo[a['category_id']]
            xc, yc_, w, h = coco_to_yolo_bbox(a['bbox'], info['width'], info['height'])
            lines.append(f'{yc} {xc:.6f} {yc_:.6f} {w:.6f} {h:.6f}')
        (out_lbl_dir / f'{Path(info["file_name"]).stem}.txt').write_text('\n'.join(lines))
        n_written += 1
    return n_written


def main():
    parser = argparse.ArgumentParser(description='按冻结划分构建 YOLO 数据集')
    parser.add_argument('--train-annotations', type=str, default='dataset/annotations_tiles.json',
                        help='训练用 COCO (切片)')
    parser.add_argument('--val-annotations', type=str, default='dataset/annotations_reduced.json',
                        help='验收用 COCO (完整原图)')
    parser.add_argument('--split', type=str, default='dataset/split_reduced.json')
    parser.add_argument('--train-split-ids', action='store_true',
                        help='train 也按 split["train"] 选图(未切片消融实验用)')
    parser.add_argument('--train-extra-augmented', action='store_true',
                        help='train 额外纳入不在 split 任何一侧的图像(如 AUG_CP_* 增强图)')
    parser.add_argument('--output', type=str, default='dataset/yolo_reduced')
    args = parser.parse_args()

    with open(args.train_annotations, 'r', encoding='utf-8') as f:
        train_coco = json.load(f)
    with open(args.val_annotations, 'r', encoding='utf-8') as f:
        val_coco = json.load(f)
    with open(args.split, 'r', encoding='utf-8') as f:
        split = json.load(f)

    # 类别一致性检查
    train_cats = {c['id']: c['name'] for c in train_coco['categories']}
    val_cats = {c['id']: c['name'] for c in val_coco['categories']}
    assert train_cats == val_cats, 'train/val 类别不一致!'
    sorted_ids = sorted(train_cats)
    cat_to_yolo = {cid: i for i, cid in enumerate(sorted_ids)}
    names = [train_cats[cid] for cid in sorted_ids]

    out = Path(args.output)
    dirs = {}
    for part in ['train', 'val']:
        for kind in ['images', 'labels']:
            d = out / kind / part
            d.mkdir(parents=True, exist_ok=True)
            dirs[(kind, part)] = d

    base = Path(args.train_annotations).parent
    if args.train_split_ids:
        train_ids = list(split['train'])
        if args.train_extra_augmented:
            in_split = set(split['train']) | set(split['val'])
            extra = [i['id'] for i in train_coco['images'] if i['id'] not in in_split]
            train_ids += extra
            print(f'额外增强图: {len(extra)} 张')
        train_dirs = [base / 'images', base / 'augmented_images']
    else:
        train_ids = [i['id'] for i in train_coco['images']]
        train_dirs = [base / 'tiles']
    n_train = write_split(
        train_coco, train_ids, train_dirs,
        dirs[('images', 'train')], dirs[('labels', 'train')], cat_to_yolo)
    n_val = write_split(
        val_coco, split['val'],
        [base / 'images'],
        dirs[('images', 'val')], dirs[('labels', 'val')], cat_to_yolo)

    yaml_lines = [
        f'path: {out.resolve()}',
        'train: images/train',
        'val: images/val',
        '',
        'names:',
    ]
    yaml_lines += [f'  {i}: {n}' for i, n in enumerate(names)]
    (out / 'dataset.yaml').write_text('\n'.join(yaml_lines) + '\n', encoding='utf-8')

    print(f'train: {n_train} 张 (切片) | val: {n_val} 张 (完整原图) | {len(names)} 类')
    print(f'输出: {out}/dataset.yaml')


if __name__ == '__main__':
    main()
