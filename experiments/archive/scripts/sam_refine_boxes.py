#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM 框提示精修脚本
功能:把松散 ROI 标注框作为 box prompt 喂给 SAM(类别无关分割),
     取 mask 外接矩形作为紧框。
接受规则(保守, 质量不达标一律回退原 ROI):
  - mask 像素数 >= min-pixels (排除退化碎片)
  - mask extent = 像素数/外接框面积 >= min-extent (排除电线/细长误分割)
  - 紧框限制在 ROI 外扩 30% 范围内
注意:ultralytics SAM 框提示的 mask 分数普遍低于默认 conf=0.25, 必须显式 conf=0.01
输出:annotations_sam_tight_shard{k}.json (仅 annotations 列表, 便于多卡分片后合并)
"""

import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict


def select_tight_box(mask, roi, min_pixels=5000, min_extent=0.2, pad_ratio=0.3):
    """mask: HxW bool; roi: [x,y,w,h]。合格返回紧框 [x,y,w,h], 否则 None"""
    ys, xs = np.where(mask)
    if len(xs) < min_pixels:
        return None
    bx, by = xs.min(), ys.min()
    bw, bh = xs.max() - bx + 1, ys.max() - by + 1
    extent = len(xs) / (bw * bh)
    if extent < min_extent:
        return None
    rx, ry, rw, rh = roi
    px, py = rw * pad_ratio, rh * pad_ratio
    x0, y0 = max(0.0, rx - px), max(0.0, ry - py)
    x1, y1 = rx + rw + px, ry + rh + py
    nx, ny = max(float(bx), x0), max(float(by), y0)
    nx2, ny2 = min(float(bx + bw), x1), min(float(by + bh), y1)
    if nx2 - nx < 10 or ny2 - ny < 10:
        return None
    return [nx, ny, nx2 - nx, ny2 - ny]


def main():
    parser = argparse.ArgumentParser(description='SAM 框提示精修')
    parser.add_argument('--annotations', type=str, default='dataset/annotations_reduced.json')
    parser.add_argument('--model', type=str, default='sam2_l.pt')
    parser.add_argument('--conf', type=float, default=0.01)
    parser.add_argument('--min-pixels', type=int, default=5000)
    parser.add_argument('--min-extent', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    from ultralytics import SAM

    with open(args.annotations, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    img_dir = Path(args.annotations).parent / 'images'

    anns_of = defaultdict(list)
    for a in coco['annotations']:
        anns_of[a['image_id']].append(a)
    info = {i['id']: i for i in coco['images']}

    img_ids = sorted(info)[args.shard::args.num_shards]
    print(f'分片 {args.shard}/{args.num_shards}: {len(img_ids)} 张图')

    model = SAM(args.model)
    out_anns = []
    n_refined = n_fallback = 0

    for vi, img_id in enumerate(img_ids, 1):
        img = cv2.imread(str(img_dir / info[img_id]['file_name']))
        if img is None:
            continue
        anns = anns_of.get(img_id, [])
        if not anns:
            continue
        boxes = []
        for a in anns:
            x, y, w, h = a['bbox']
            boxes.append([x, y, x + w, y + h])
        try:
            res = model(img, bboxes=boxes, conf=args.conf,
                        device=args.device, verbose=False)[0]
        except Exception as e:
            print(f'  img {img_id} 推理失败: {e}')
            res = None
        for ai, a in enumerate(anns):
            nb = None
            if res is not None and res.masks is not None and len(res.masks) > ai:
                m = res.masks.data[ai].cpu().numpy()
                if m.shape != img.shape[:2]:
                    m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
                nb = select_tight_box(m > 0.5, a['bbox'],
                                      args.min_pixels, args.min_extent)
            na = dict(a)
            if nb is not None:
                na['bbox'] = nb
                na['area'] = nb[2] * nb[3]
                na['refined'] = True
                n_refined += 1
            else:
                n_fallback += 1
            out_anns.append(na)
        if vi % 100 == 0:
            print(f'  {vi}/{len(img_ids)} 精修 {n_refined} 回退 {n_fallback}', flush=True)

    out = args.output or f'dataset/annotations_sam_tight_shard{args.shard}.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(out_anns, f, ensure_ascii=False)
    print(f'完成: 精修 {n_refined}, 回退 {n_fallback}, 输出 {out}')


if __name__ == '__main__':
    main()
