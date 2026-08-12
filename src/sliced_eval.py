#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAHI 式切片推理验收脚本
功能:完整原图切成重叠切片逐片推理, 坐标映射回原图, 类内 NMS 合并,
     对冻结 val 计算 mAP50 / mAP50-95 (COCO 101 点插值, 自实现无额外依赖)
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict


def tiles_of(h, w, tile, stride):
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            ty0, tx0 = max(0, y1 - tile), max(0, x1 - tile)
            yield tx0, ty0, x1 - tx0, y1 - ty0


def nms_xyxy(boxes, scores, iou_thr):
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def ap_101(recalls, precisions):
    """COCO 101 点插值 AP"""
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = precisions[recalls >= t].max() if (recalls >= t).any() else 0.0
        ap += p
    return ap / 101


def evaluate(all_dets, all_gts, iou_thr):
    """all_dets/all_gts: {cat: [(img_id, conf/score ignored, xyxy)]}; 返回该 IoU 下各类 AP"""
    aps = {}
    for cat, gts in all_gts.items():
        dets = sorted(all_dets.get(cat, []), key=lambda d: -d[1])
        npos = len(gts)
        gt_matched = defaultdict(set)  # img_id -> 已匹配 gt 索引
        gt_by_img = defaultdict(list)
        for gi, (img_id, box) in enumerate(gts):
            gt_by_img[img_id].append((gi, box))
        tp, fp = [], []
        for img_id, conf, box in dets:
            best_iou, best_gi = 0.0, -1
            for gi, gbox in gt_by_img.get(img_id, []):
                if gi in gt_matched[img_id]:
                    continue
                xx1, yy1 = max(box[0], gbox[0]), max(box[1], gbox[1])
                xx2, yy2 = min(box[2], gbox[2]), min(box[3], gbox[3])
                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                ua = (box[2]-box[0])*(box[3]-box[1]) + (gbox[2]-gbox[0])*(gbox[3]-gbox[1]) - inter
                iou = inter / (ua + 1e-9)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_thr and best_gi >= 0:
                gt_matched[img_id].add(best_gi)
                tp.append(1); fp.append(0)
            else:
                tp.append(0); fp.append(1)
        if npos == 0 or not tp:
            continue
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        recalls = tp_c / npos
        precisions = tp_c / (tp_c + fp_c + 1e-9)
        aps[cat] = ap_101(recalls, precisions)
    return aps


def main():
    parser = argparse.ArgumentParser(description='SAHI 切片推理验收')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--annotations', type=str, default='dataset/annotations_reduced.json')
    parser.add_argument('--split', type=str, default='dataset/split_reduced.json')
    parser.add_argument('--tile', type=int, default=1280)
    parser.add_argument('--stride', type=int, default=1024)
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--nms-iou', type=float, default=0.5)
    parser.add_argument('--imgsz', type=int, default=1280, help='切片推理输入尺寸(与训练一致)')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    from ultralytics import YOLO

    with open(args.annotations, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    with open(args.split, 'r', encoding='utf-8') as f:
        val_ids = set(json.load(f)['val'])

    # YOLO 类别索引 -> COCO category_id
    sorted_ids = sorted(c['id'] for c in coco['categories'])
    cat_names = {c['id']: c['name'] for c in coco['categories']}

    img_dir = Path(args.annotations).parent / 'images'
    info = {i['id']: i for i in coco['images'] if i['id'] in val_ids}

    # GT (xyxy, 按类)
    all_gts = defaultdict(list)
    for a in coco['annotations']:
        if a['image_id'] in val_ids:
            x, y, w, h = a['bbox']
            all_gts[a['category_id']].append((a['image_id'], [x, y, x + w, y + h]))

    model = YOLO(args.model)
    all_dets = defaultdict(list)  # cat -> [(img_id, conf, xyxy)]

    val_list = sorted(info)
    for vi, img_id in enumerate(val_list, 1):
        img = cv2.imread(str(img_dir / info[img_id]['file_name']))
        if img is None:
            continue
        H, W = img.shape[:2]
        crops, offsets = [], []
        for tx0, ty0, tw, th in tiles_of(H, W, args.tile, args.stride):
            crops.append(img[ty0:ty0+th, tx0:tx0+tw])
            offsets.append((tx0, ty0))
        img_dets = defaultdict(lambda: [[], []])  # yolo_cls -> [boxes, scores]
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
        for cls, (boxes, scores) in img_dets.items():
            boxes = np.array(boxes)
            scores = np.array(scores)
            for ki in nms_xyxy(boxes, scores, args.nms_iou):
                all_dets[sorted_ids[cls]].append((img_id, float(scores[ki]), boxes[ki].tolist()))
        if vi % 50 == 0:
            print(f'  {vi}/{len(val_list)}')

    # mAP50 与 mAP50-95
    ap50 = evaluate(all_dets, all_gts, 0.5)
    maps = []
    for thr in np.arange(0.5, 1.0, 0.05):
        aps = evaluate(all_dets, all_gts, round(thr, 2))
        maps.append(np.mean(list(aps.values())) if aps else 0.0)

    print(f'\n{"类别":<16} {"AP50":>7}')
    for cid in sorted(cat_names):
        if cid in ap50:
            print(f'{cat_names[cid]:<16} {ap50[cid]:7.3f}')
    print(f'\nmAP50   = {np.mean(list(ap50.values())):.4f}')
    print(f'mAP50-95= {np.mean(maps):.4f}')


if __name__ == '__main__':
    main()
