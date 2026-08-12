#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
紧框重标注脚本
功能:用 YOLO-World 开放词汇检测在现有 ROI 标注框内生成紧框
     每个 ROI 按类别的设备 prompt 匹配最佳检测框; 无匹配时保留原 ROI
输出:dataset/annotations_tight.json + 每类收紧成功率统计
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

# 类别 -> YOLO-World 英文 prompt 组(描述缺陷所在的设备/对象)
CLASS_PROMPTS = {
    '绑扎线不规范': ['binding wire', 'tie wire', 'wire wrap', 'metal wire', 'twisted wire', 'wire binding'],
    '绝缘子污秽': ['ceramic insulator', 'insulator', 'porcelain insulator'],
    '横担锈蚀': ['crossarm', 'steel crossarm', 'metal crossarm', 'angle steel'],
    '距树木距离不够': ['tree crown', 'tree branches', 'treetop', 'vegetation'],
    '其他缺陷': [],  # 无法定义 prompt, 保持原 ROI
    '绝缘线绝缘层破损': ['power line', 'cable', 'wire', 'electrical wire'],
    '套管破损': ['bushing', 'ceramic bushing', 'transformer bushing'],
    '绝缘罩脱落': ['insulator cover', 'protective cover', 'plastic cover', 'insulation hood', 'silicone cover'],
    '避雷器破损': ['surge arrester', 'lightning arrester', 'arrester'],
    '杆基杂物堆积': ['debris pile', 'garbage pile', 'trash pile', 'weeds'],
    '绝缘子固定不牢固': ['ceramic insulator', 'insulator', 'porcelain insulator'],
    '绝缘子破损': ['ceramic insulator', 'insulator', 'porcelain insulator'],
    '绝缘护套缺陷': ['cable sheath', 'insulation sleeve', 'cable jacket', 'wire insulation'],
    '杆塔异物鸟巢': ['bird nest', 'nest', 'twig nest', 'bird nest on pole'],
    '杆塔未设防护设施': ['utility pole', 'concrete pole', 'power pole'],
    '接地引下线连接不良': ['grounding wire', 'ground wire', 'copper wire', 'down lead'],
}


def box_center_in(box, roi):
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    return roi[0] <= cx <= roi[0] + roi[2] and roi[1] <= cy <= roi[1] + roi[3]


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def main():
    parser = argparse.ArgumentParser(description='YOLO-World 紧框重标注')
    parser.add_argument('--annotations', type=str, default='dataset/annotations_reduced.json')
    parser.add_argument('--model', type=str, default='yolov8x-worldv2.pt')
    parser.add_argument('--conf', type=float, default=0.02, help='检测置信度阈值(召回优先)')
    parser.add_argument('--min-iou', type=float, default=0.05, help='检测框与 ROI 的最小 IoU')
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--output', type=str, default='dataset/annotations_tight.json')
    args = parser.parse_args()

    from ultralytics import YOLO

    with open(args.annotations, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    id_to_name = {c['id']: c['name'] for c in coco['categories']}

    # 全部 prompt 去重, 建立 prompt -> 索引 与 类别 -> prompt 索引组
    all_prompts = sorted({p for ps in CLASS_PROMPTS.values() for p in ps})
    prompt_idx = {p: i for i, p in enumerate(all_prompts)}
    cat_prompt_ids = {}
    for cname, ps in CLASS_PROMPTS.items():
        cat_prompt_ids[cname] = {prompt_idx[p] for p in ps}

    print(f'加载 {args.model}, {len(all_prompts)} 个 prompt')
    model = YOLO(args.model)
    model.set_classes(all_prompts)

    img_dir = Path(args.annotations).parent / 'images'
    anns_of = defaultdict(list)
    for a in coco['annotations']:
        anns_of[a['image_id']].append(a)

    # 每张图推理一次, 然后 ROI 匹配
    stats_total = defaultdict(int)
    stats_matched = defaultdict(int)
    new_annotations = []
    unmatched = []  # (new_annotations 索引, image_path, roi, cname)
    next_ann_id = 1

    images = coco['images']
    for idx, info in enumerate(images, 1):
        img_path = img_dir / info['file_name']
        anns = anns_of.get(info['id'], [])
        if not anns:
            continue
        results = model(str(img_path), conf=args.conf, imgsz=args.imgsz,
                        device=args.device, verbose=False)[0]
        # 按 prompt 索引分组检测框
        dets_by_prompt = defaultdict(list)  # prompt_idx -> [(conf, xywh)]
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls = int(box.cls)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets_by_prompt[cls].append((float(box.conf), [x1, y1, x2 - x1, y2 - y1]))

        for a in anns:
            cname = id_to_name[a['category_id']]
            roi = a['bbox']
            stats_total[cname] += 1
            best = None
            for pi in cat_prompt_ids.get(cname, ()):  # 该类 prompt 组的检测框
                for conf, dbox in dets_by_prompt.get(pi, ()):  # 匹配: 中心在 ROI 内或 IoU 达标
                    if box_center_in(dbox, roi) or box_iou(dbox, roi) >= args.min_iou:
                        if best is None or conf > best[0]:
                            best = (conf, dbox)
            if best is not None:
                nb = best[1]
                stats_matched[cname] += 1
            else:
                nb = roi  # 回退保留原 ROI
            new_annotations.append({
                'id': next_ann_id,
                'image_id': a['image_id'],
                'category_id': a['category_id'],
                'bbox': [float(v) for v in nb],
                'area': float(nb[2] * nb[3]),
                'iscrowd': 0,
            })
            if best is None and cat_prompt_ids.get(cname):
                unmatched.append((len(new_annotations) - 1, str(img_path), roi, cname))
            next_ann_id += 1

        if idx % 200 == 0:
            print(f'  {idx}/{len(images)}')

    # 第二遍: 未匹配 ROI 的局部放大推理(小目标相对变大)
    if unmatched:
        print(f'\n第二遍: {len(unmatched)} 个未匹配 ROI 局部放大推理...')
        import cv2
        pad_ratio = 0.3
        for ui, (ann_i, img_path, roi, cname) in enumerate(unmatched, 1):
            img = cv2.imread(img_path)
            if img is None:
                continue
            H, W = img.shape[:2]
            x, y, w, h = roi
            pw, ph = w * pad_ratio, h * pad_ratio
            x0, y0 = int(max(0, x - pw)), int(max(0, y - ph))
            x1, y1 = int(min(W, x + w + pw)), int(min(H, y + h + ph))
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            results = model(crop, conf=args.conf, imgsz=args.imgsz,
                            device=args.device, verbose=False)[0]
            best = None
            if results.boxes is not None:
                for box in results.boxes:
                    cls = int(box.cls)
                    if cls not in cat_prompt_ids[cname]:
                        continue
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    dbox = [bx1 + x0, by1 + y0, bx2 - bx1, by2 - by1]
                    if box_center_in(dbox, roi) or box_iou(dbox, roi) >= args.min_iou:
                        conf = float(box.conf)
                        if best is None or conf > best[0]:
                            best = (conf, dbox)
            if best is not None:
                nb = best[1]
                new_annotations[ann_i]['bbox'] = [float(v) for v in nb]
                new_annotations[ann_i]['area'] = float(nb[2] * nb[3])
                stats_matched[cname] += 1
            if ui % 500 == 0:
                print(f'  第二遍 {ui}/{len(unmatched)}')

    print(f'\n{"类别":<16} {"收紧成功":>8} {"总数":>6} {"比例":>6}')
    for cname in sorted(stats_total, key=lambda c: -stats_total[c]):
        t, m = stats_total[cname], stats_matched[cname]
        print(f'{cname:<16} {m:>8} {t:>6} {m/t*100:>5.1f}%')

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'info': coco.get('info', {}),
            'images': coco['images'],
            'annotations': new_annotations,
            'categories': coco['categories'],
        }, f, ensure_ascii=False, indent=2)
    print(f'\n输出: {args.output}')


if __name__ == '__main__':
    main()
