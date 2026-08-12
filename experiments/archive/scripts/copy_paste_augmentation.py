#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copy-Paste 增强脚本（针对稀有类别）
功能：从稀有类别提取 patch 并粘贴到随机背景上，直接扩充样本数量
"""

import json
import cv2
import numpy as np
import random
from pathlib import Path
from collections import defaultdict, Counter
import argparse


def compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
    y2 = min(box_a[1] + box_a[3], box_b[1] + box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    if area_a + area_b - inter <= 0:
        return 0.0
    return inter / (area_a + area_b - inter)


def extract_patch(img, bbox):
    x, y, w, h = [int(v) for v in bbox]
    x = max(0, x)
    y = max(0, y)
    h_img, w_img = img.shape[:2]
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    if w <= 0 or h <= 0:
        return None
    return img[y:y+h, x:x+w]


def color_jitter_patch(patch, alpha_range=(0.8, 1.2), beta_range=(-20, 20)):
    alpha = random.uniform(*alpha_range)
    beta = random.uniform(*beta_range)
    patch = cv2.convertScaleAbs(patch, alpha=alpha, beta=beta)
    return patch


def paste_patch(background, patch, center_xy):
    bh, bw = background.shape[:2]
    ph, pw = patch.shape[:2]
    cx, cy = center_xy

    # 随机缩放 0.8x - 1.2x
    scale = random.uniform(0.8, 1.2)
    if scale != 1.0:
        pw = int(pw * scale)
        ph = int(ph * scale)
        patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)

    x1 = max(0, cx - pw // 2)
    y1 = max(0, cy - ph // 2)
    x2 = min(bw, x1 + pw)
    y2 = min(bh, y1 + ph)

    patch_clipped = patch[:y2-y1, :x2-x1]
    background[y1:y2, x1:x2] = patch_clipped
    return background, [x1, y1, x2 - x1, y2 - y1]


def find_valid_paste_location(bg_shape, patch_shape, existing_bboxes, max_tries=20, iou_thresh=0.2):
    bh, bw = bg_shape[:2]
    ph, pw = patch_shape[:2]
    margin_x = int(bw * 0.05)
    margin_y = int(bh * 0.05)
    if pw > bw - 2 * margin_x or ph > bh - 2 * margin_y:
        return None, None

    for _ in range(max_tries):
        cx = random.randint(margin_x + pw // 2, bw - margin_x - pw // 2)
        cy = random.randint(margin_y + ph // 2, bh - margin_y - ph // 2)
        new_box = [cx - pw // 2, cy - ph // 2, pw, ph]

        if all(compute_iou(new_box, eb) < iou_thresh for eb in existing_bboxes):
            return cx, cy
    return None, None


def main():
    parser = argparse.ArgumentParser(description='Copy-Paste 稀有类别增强')
    parser.add_argument('--annotations', type=str, default='dataset/annotations.json')
    parser.add_argument('--rare-threshold', type=int, default=50)
    parser.add_argument('--target-per-category', type=int, default=200)
    parser.add_argument('--output', type=str, default='dataset/annotations_copypaste.json')
    parser.add_argument('--split', type=str, default=None,
                        help='划分 JSON; 提供后 patch 来源与背景都只用 train 图像(防 val 泄漏)')
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    ann_path = Path(args.annotations)
    img_dir = ann_path.parent / 'images'
    aug_img_dir = ann_path.parent / 'augmented_images'
    aug_img_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    train_only = None
    if args.split:
        with open(args.split, 'r', encoding='utf-8') as f:
            train_only = set(json.load(f)['train'])
        print(f'划分约束: patch/背景仅来自 {len(train_only)} 张 train 图像')

    cat_counts = Counter(a['category_id'] for a in coco['annotations']
                         if train_only is None or a['image_id'] in train_only)
    rare_cat_ids = {cid for cid, c in cat_counts.items() if c < args.rare_threshold}

    print(f"发现 {len(rare_cat_ids)} 个稀有类别 (<{args.rare_threshold} 标注)")

    img_id_to_info = {img['id']: img for img in coco['images']}
    img_to_anns = defaultdict(list)
    cat_to_anns = defaultdict(list)
    for ann in coco['annotations']:
        img_to_anns[ann['image_id']].append(ann)
        cat_to_anns[ann['category_id']].append(ann)

    # 预提取稀有类别的 patch
    patches = defaultdict(list)  # cat_id -> list of (patch_img, orig_bbox)
    for cid in rare_cat_ids:
        for ann in cat_to_anns.get(cid, []):
            if train_only is not None and ann['image_id'] not in train_only:
                continue
            img_info = img_id_to_info[ann['image_id']]
            img_path = img_dir / img_info['file_name']
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            patch = extract_patch(img, ann['bbox'])
            if patch is not None and patch.size > 0:
                patches[cid].append((patch, ann['bbox']))

    # 背景图像池：优先选择标注较少的图像
    bg_images = [img for img in coco['images']
                 if train_only is None or img['id'] in train_only]

    next_img_id = max(img['id'] for img in coco['images']) + 1
    next_ann_id = max(a['id'] for a in coco['annotations']) + 1

    total_added_images = 0
    total_added_anns = 0

    for cid in rare_cat_ids:
        current_count = cat_counts[cid]
        needed = max(0, args.target_per_category - current_count)
        if needed == 0 or not patches[cid]:
            continue

        print(f"类别 {id_to_name(cid, coco)} (id={cid}): 当前 {current_count}, 需要补充 {needed}")

        for i in range(needed):
            patch_img, orig_bbox = random.choice(patches[cid])
            bg_info = random.choice(bg_images)
            bg_path = img_dir / bg_info['file_name']
            bg = cv2.imread(str(bg_path))
            if bg is None:
                continue

            # 获取背景上的已有标注框（COCO 格式）
            existing_bboxes = [ann['bbox'] for ann in img_to_anns.get(bg_info['id'], [])]

            # 颜色抖动
            patch_img_jittered = color_jitter_patch(patch_img.copy())

            # 松散 ROI patch 可能过大, 先缩到背景的 60% 以内
            bh, bw = bg.shape[:2]
            ph, pw = patch_img_jittered.shape[:2]
            fit = min(1.0, 0.6 * bw / pw, 0.6 * bh / ph)
            if fit < 1.0:
                patch_img_jittered = cv2.resize(
                    patch_img_jittered, (int(pw * fit), int(ph * fit)),
                    interpolation=cv2.INTER_AREA)

            ph, pw = patch_img_jittered.shape[:2]
            cx, cy = find_valid_paste_location(bg.shape, (ph, pw), existing_bboxes)
            if cx is None:
                continue

            bg_aug, new_bbox = paste_patch(bg.copy(), patch_img_jittered, (cx, cy))

            aug_filename = f"AUG_CP_{next_img_id:08d}.jpg"
            aug_path = aug_img_dir / aug_filename
            cv2.imwrite(str(aug_path), bg_aug)

            coco['images'].append({
                'id': next_img_id,
                'file_name': aug_filename,
                'original_name': f"copypaste_from_{bg_info['file_name']}",
                'width': bg_info['width'],
                'height': bg_info['height'],
                'line_name': bg_info.get('line_name', 'augmented'),
                'pole_id': bg_info.get('pole_id', 'augmented')
            })

            coco['annotations'].append({
                'id': next_ann_id,
                'image_id': next_img_id,
                'category_id': cid,
                'bbox': [float(v) for v in new_bbox],
                'area': float(new_bbox[2] * new_bbox[3]),
                'iscrowd': 0
            })

            next_img_id += 1
            next_ann_id += 1
            total_added_images += 1
            total_added_anns += 1

    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"\nCopy-Paste 增强完成！")
    print(f"新增图像: {total_added_images}")
    print(f"新增标注: {total_added_anns}")
    print(f"输出文件: {output_path}")


def id_to_name(cat_id, coco_data):
    for cat in coco_data['categories']:
        if cat['id'] == cat_id:
            return cat['name']
    return str(cat_id)


if __name__ == '__main__':
    main()
