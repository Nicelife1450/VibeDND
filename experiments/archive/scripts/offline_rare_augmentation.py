#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线重度增强脚本（针对稀有类别）
功能：对包含稀有类别的训练图像进行离线增强，生成3倍变体，直接扩充到 dataset/images/
"""

import json
import cv2
import numpy as np
import random
from pathlib import Path
from collections import defaultdict, Counter
import argparse


def random_brightness_contrast(img, alpha_range=(0.7, 1.3), beta_range=(-30, 30)):
    alpha = random.uniform(*alpha_range)
    beta = random.uniform(*beta_range)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img


def hsv_jitter(img, h_limit=10, s_limit=40, v_limit=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    h = (h + random.randint(-h_limit, h_limit)) % 180
    s = np.clip(s + random.randint(-s_limit, s_limit), 0, 255)
    v = np.clip(v + random.randint(-v_limit, v_limit), 0, 255)
    hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2] = h, s, v
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def gauss_noise(img, var_limit=(10, 50)):
    var = random.randint(*var_limit)
    sigma = var ** 0.5
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    img = img.astype(np.float32) + noise
    return np.clip(img, 0, 255).astype(np.uint8)


def shift_scale_rotate(img, bboxes, shift_limit=0.1, scale_limit=0.2, rotate_limit=15):
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    shift_x = random.uniform(-shift_limit, shift_limit) * w
    shift_y = random.uniform(-shift_limit, shift_limit) * h
    scale = 1 + random.uniform(-scale_limit, scale_limit)
    angle = random.uniform(-rotate_limit, rotate_limit)

    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += shift_x
    M[1, 2] += shift_y

    img_warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))

    new_bboxes = []
    for bbox in bboxes:
        x, y, bw, bh = bbox
        corners = np.array([
            [x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]
        ], dtype=np.float32)
        corners = np.hstack([corners, np.ones((4, 1))])
        warped = (M @ corners.T).T
        nx, ny = warped[:, 0].min(), warped[:, 1].min()
        nx2, ny2 = warped[:, 0].max(), warped[:, 1].max()
        nx = max(0, nx)
        ny = max(0, ny)
        nx2 = min(w, nx2)
        ny2 = min(h, ny2)
        nbw, nbh = nx2 - nx, ny2 - ny
        if nbw > 5 and nbh > 5:
            new_bboxes.append([nx, ny, nbw, nbh])
        else:
            new_bboxes.append([x, y, bw, bh])

    return img_warped, new_bboxes


def horizontal_flip(img, bboxes):
    h, w = img.shape[:2]
    img = cv2.flip(img, 1)
    new_bboxes = []
    for bbox in bboxes:
        x, y, bw, bh = bbox
        nx = w - x - bw
        new_bboxes.append([nx, y, bw, bh])
    return img, new_bboxes


def apply_augmentation(img, bboxes):
    aug_order = random.sample(['bc', 'hsv', 'noise', 'ssr', 'flip'], k=random.randint(2, 5))
    for aug in aug_order:
        if aug == 'bc':
            img = random_brightness_contrast(img)
        elif aug == 'hsv':
            img = hsv_jitter(img)
        elif aug == 'noise':
            img = gauss_noise(img)
        elif aug == 'ssr':
            img, bboxes = shift_scale_rotate(img, bboxes)
        elif aug == 'flip':
            if random.random() < 0.5:
                img, bboxes = horizontal_flip(img, bboxes)
    return img, bboxes


def main():
    parser = argparse.ArgumentParser(description='离线稀有类别增强')
    parser.add_argument('--annotations', type=str, default='dataset/annotations.json')
    parser.add_argument('--rare-threshold', type=int, default=50)
    parser.add_argument('--variants', type=int, default=3, help='每张稀有图像生成的变体数量')
    parser.add_argument('--output', type=str, default='dataset/annotations_augmented_rare.json')
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    ann_path = Path(args.annotations)
    img_dir = ann_path.parent / 'images'

    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    cat_counts = Counter(a['category_id'] for a in coco['annotations'])
    rare_cat_ids = {cid for cid, c in cat_counts.items() if c < args.rare_threshold}

    print(f"发现 {len(rare_cat_ids)} 个稀有类别 (<{args.rare_threshold} 标注)")

    img_to_anns = defaultdict(list)
    for ann in coco['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    img_id_to_info = {img['id']: img for img in coco['images']}

    next_img_id = max(img['id'] for img in coco['images']) + 1
    next_ann_id = max(a['id'] for a in coco['annotations']) + 1

    total_added_images = 0
    total_added_anns = 0

    for img_id, anns in img_to_anns.items():
        if not any(ann['category_id'] in rare_cat_ids for ann in anns):
            continue

        img_info = img_id_to_info[img_id]
        img_path = img_dir / img_info['file_name']
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue

        bboxes = [ann['bbox'] for ann in anns]
        labels = [ann['category_id'] for ann in anns]

        for aug_idx in range(args.variants):
            aug_img, aug_bboxes = apply_augmentation(img.copy(), [b[:] for b in bboxes])

            new_filename = f"AUG_RARE_{next_img_id:08d}.jpg"
            new_path = img_dir / new_filename
            cv2.imwrite(str(new_path), aug_img)

            coco['images'].append({
                'id': next_img_id,
                'file_name': new_filename,
                'original_name': f'aug_{aug_idx}_of_{img_info["file_name"]}',
                'width': img_info['width'],
                'height': img_info['height'],
                'line_name': img_info.get('line_name', ''),
                'pole_id': img_info.get('pole_id', '')
            })

            for bbox, label in zip(aug_bboxes, labels):
                x, y, w, h = bbox
                coco['annotations'].append({
                    'id': next_ann_id,
                    'image_id': next_img_id,
                    'category_id': label,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'area': float(w * h),
                    'iscrowd': 0
                })
                next_ann_id += 1
                total_added_anns += 1

            next_img_id += 1
            total_added_images += 1

    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"\n增强完成！")
    print(f"新增图像: {total_added_images}")
    print(f"新增标注: {total_added_anns}")
    print(f"输出文件: {output_path}")


if __name__ == '__main__':
    main()
