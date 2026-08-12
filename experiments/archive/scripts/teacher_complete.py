#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
教师补全标注脚本
功能:用教师模型对 train 图像做切片推理, 把与现有 GT 不重叠的高置信检测
     补为新标注(修复巡检报告"只圈上报缺陷"导致的标注不完整问题)
输出:dataset/annotations_completed.json
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from sliced_eval import tiles_of, nms_xyxy


def iou_xyxy(a, b):
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-9)


def main():
    parser = argparse.ArgumentParser(description='教师补全标注')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--annotations', type=str, default='dataset/annotations_reduced.json')
    parser.add_argument('--split', type=str, default='dataset/split_reduced.json')
    parser.add_argument('--conf', type=float, default=0.15)
    parser.add_argument('--gt-iou', type=float, default=0.3, help='与已有 GT IoU 超过此值则视为已标注')
    parser.add_argument('--tile', type=int, default=1280)
    parser.add_argument('--stride', type=int, default=1024)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--output', type=str, default='dataset/annotations_completed.json')
    args = parser.parse_args()

    from ultralytics import YOLO

    with open(args.annotations, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    with open(args.split, 'r', encoding='utf-8') as f:
        train_ids = set(json.load(f)['train'])

    sorted_ids = sorted(c['id'] for c in coco['categories'])
    cat_names = {c['id']: c['name'] for c in coco['categories']}
    img_dir = Path(args.annotations).parent / 'images'
    info = {i['id']: i for i in coco['images'] if i['id'] in train_ids}

    gt_by_img = defaultdict(list)
    for a in coco['annotations']:
        if a['image_id'] in train_ids:
            x, y, w, h = a['bbox']
            gt_by_img[a['image_id']].append([x, y, x + w, y + h])

    model = YOLO(args.model)
    new_annotations = list(coco['annotations'])
    next_ann_id = max(a['id'] for a in coco['annotations']) + 1
    added_per_cat = defaultdict(int)

    train_list = sorted(info)
    for vi, img_id in enumerate(train_list, 1):
        img = cv2.imread(str(img_dir / info[img_id]['file_name']))
        if img is None:
            continue
        H, W = img.shape[:2]
        crops, offsets = [], []
        for tx0, ty0, tw, th in tiles_of(H, W, args.tile, args.stride):
            crops.append(img[ty0:ty0+th, tx0:tx0+tw])
            offsets.append((tx0, ty0))
        img_dets = defaultdict(lambda: [[], []])
        for b0 in range(0, len(crops), args.batch):
            batch = crops[b0:b0+args.batch]
            results = model(batch, conf=args.conf, imgsz=args.imgsz,
                            device=args.device, verbose=False)
            for bi, r in enumerate(results):
                if r.boxes is None:
                    continue
                tx0, ty0 = offsets[b0 + bi]
                for box in r.boxes:
                    cls = int(box.cls)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    img_dets[cls][0].append([x1+tx0, y1+ty0, x2+tx0, y2+ty0])
                    img_dets[cls][1].append(float(box.conf))
        gts = gt_by_img.get(img_id, [])
        for cls, (boxes, scores) in img_dets.items():
            boxes = np.array(boxes)
            scores = np.array(scores)
            for ki in nms_xyxy(boxes, scores, 0.5):
                det = boxes[ki].tolist()
                # 与任何现有 GT 重叠则跳过(已被人工标注)
                if any(iou_xyxy(det, g) >= args.gt_iou for g in gts):
                    continue
                x1, y1, x2, y2 = det
                new_annotations.append({
                    'id': next_ann_id,
                    'image_id': img_id,
                    'category_id': sorted_ids[cls],
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'area': (x2 - x1) * (y2 - y1),
                    'iscrowd': 0,
                    'teacher': True,
                })
                added_per_cat[sorted_ids[cls]] += 1
                next_ann_id += 1
        if vi % 200 == 0:
            print(f'  {vi}/{len(train_list)}, 已补 {next_ann_id - len(coco["annotations"]) - 1} 条')

    print(f'\n教师补全:')
    for cid in sorted(added_per_cat, key=lambda c: -added_per_cat[c]):
        print(f'  {cat_names[cid]:<16} +{added_per_cat[cid]}')
    print(f'总补标: {sum(added_per_cat.values())}')

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'info': coco.get('info', {}),
            'images': coco['images'],
            'annotations': new_annotations,
            'categories': coco['categories'],
        }, f, ensure_ascii=False, indent=2)
    print(f'输出: {args.output}')


if __name__ == '__main__':
    main()
