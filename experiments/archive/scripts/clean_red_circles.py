#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红圈污染清洗脚本
功能:对红圈污染图像做 inpainting 去除红圈, 污染原图备份到 dataset/images_contaminated_backup/
输入:dataset/contaminated_images.json (由扫描生成)
"""

import json
import cv2
import numpy as np
import shutil
import argparse
from pathlib import Path
from multiprocessing import Pool


def inpaint_red(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | \
           cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    # 连通域形状分析(用真实像素面积): 手绘圈是细环(extent 低), 实心红块(缺陷本体)保留
    keep = np.zeros_like(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 20:
            keep[labels == i] = 255  # 小噪点直接清
            continue
        extent = area / (w * h)  # 细环 ≈0.05-0.3, 实心块 >0.5
        if extent < 0.4:
            keep[labels == i] = 255
    if keep.sum() == 0:
        return img
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8), iterations=2)
    # 只对包含手绘圈的局部 ROI 做 inpaint (全图 inpaint 在 18MP 上要 ~5s)
    out = img.copy()
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img.shape[:2]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        pad = 15
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        roi = img[y0:y1, x0:x1]
        roi_mask = keep[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = cv2.inpaint(roi, roi_mask, 5, cv2.INPAINT_TELEA)
    return out


def process_one(job):
    """job = (src_path, bak_path)。从备份还原(或建立备份)后 inpaint 覆盖写回。"""
    src, bak = job
    src, bak = Path(src), Path(bak)
    try:
        if bak.exists():
            shutil.copy2(bak, src)
        else:
            shutil.copy2(src, bak)
        img = cv2.imread(str(src))
        if img is None:
            return False
        cv2.imwrite(str(src), inpaint_red(img))
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description='红圈污染清洗 (inpainting)')
    parser.add_argument('--contaminated', type=str, default='dataset/contaminated_images.json')
    parser.add_argument('--images-dir', type=str, default='dataset/images')
    parser.add_argument('--backup-dir', type=str, default='dataset/images_contaminated_backup')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()

    with open(args.contaminated, 'r', encoding='utf-8') as f:
        contaminated = json.load(f)

    images_dir = Path(args.images_dir)
    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 建立 image_id -> file_name 映射
    with open(images_dir.parent / 'annotations_reduced.json', 'r', encoding='utf-8') as f:
        coco = json.load(f)
    id_to_file = {str(i['id']): i['file_name'] for i in coco['images']}

    jobs = []
    for iid in contaminated:
        fname = id_to_file.get(iid)
        if fname:
            jobs.append((str(images_dir / fname), str(backup_dir / fname)))

    with Pool(args.workers) as pool:
        results = pool.map(process_one, jobs, chunksize=8)

    print(f'\n清洗完成: {sum(results)} 张, 失败/跳过 {len(results) - sum(results)}')
    print(f'污染原图备份: {backup_dir}/')


if __name__ == '__main__':
    main()
