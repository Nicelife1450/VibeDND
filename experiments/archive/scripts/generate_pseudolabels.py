#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
伪标签生成脚本
功能：使用训练好的教师模型为未标注的原始图像生成高置信度伪标签
"""

import json
import cv2
import shutil
from pathlib import Path
from collections import defaultdict
import argparse


def find_unannotated_images(coco_path, inspection_dir):
    """找出巡检报告中有但不在 COCO 数据集中的原始图像"""
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    annotated_names = {img['original_name'] for img in coco['images']}

    unannotated = []
    inspection_path = Path(inspection_dir)
    if not inspection_path.exists():
        print(f"Warning: 巡检报告目录不存在: {inspection_path}")
        return unannotated

    for year_dir in inspection_path.iterdir():
        if not year_dir.is_dir():
            continue
        for line_dir in year_dir.iterdir():
            if not line_dir.is_dir():
                continue
            original_dir = line_dir / "缺陷原图"
            if original_dir.exists():
                for img_path in original_dir.rglob("*.jpg"):
                    if img_path.name not in annotated_names:
                        unannotated.append(img_path)
    return unannotated


def main():
    parser = argparse.ArgumentParser(description='生成伪标签')
    parser.add_argument('--model', type=str, required=True, help='教师模型权重路径 (.pt)')
    parser.add_argument('--coco', type=str, default='dataset/annotations.json')
    parser.add_argument('--inspection-dir', type=str, default='巡检报告')
    parser.add_argument('--conf', type=float, default=0.6, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.5, help='NMS IoU 阈值')
    parser.add_argument('--output', type=str, default='dataset/pseudo_annotations.json')
    parser.add_argument('--max-images', type=int, default=5000, help='最大处理图像数')
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics 未安装，请先安装: pip install ultralytics")
        return

    print("查找未标注图像...")
    unannotated = find_unannotated_images(args.coco, args.inspection_dir)
    print(f"找到 {len(unannotated)} 张未标注原始图像")

    if len(unannotated) == 0:
        print("没有未标注图像，退出")
        return

    if len(unannotated) > args.max_images:
        import random
        random.seed(42)
        unannotated = random.sample(unannotated, args.max_images)
        print(f"随机采样至 {args.max_images} 张")

    print(f"加载教师模型: {args.model}")
    model = YOLO(args.model)

    # 加载原始 categories 用于映射
    with open(args.coco, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    categories = coco['categories']

    # 构建 YOLO class index -> COCO category_id 映射
    # YOLO 输出的是合并后的 0-indexed 类别，需要还原到原始 COCO 类别 ID
    # 这里我们先读取 YOLO 对应的类别名称，再匹配回 COCO
    yolo_names = model.names  # dict: {0: 'name', ...}
    name_to_cat_id = {cat['name']: cat['id'] for cat in categories}

    pseudo_images = []
    pseudo_annotations = []
    next_img_id = 1
    next_ann_id = 1

    output_img_dir = Path(args.output).parent / 'pseudo_images'
    output_img_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n开始生成伪标签 (conf={args.conf}, iou={args.iou})...")
    for idx, img_path in enumerate(unannotated, 1):
        results = model(str(img_path), conf=args.conf, iou=args.iou, verbose=False)[0]
        if len(results.boxes) == 0:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        pseudo_filename = f"PSEUDO_{next_img_id:08d}.jpg"
        pseudo_img_path = output_img_dir / pseudo_filename
        shutil.copy2(str(img_path), str(pseudo_img_path))

        pseudo_images.append({
            'id': next_img_id,
            'file_name': pseudo_filename,
            'original_name': img_path.name,
            'width': w,
            'height': h,
            'line_name': 'pseudo',
            'pole_id': 'pseudo'
        })

        for box in results.boxes:
            yolo_cls = int(box.cls)
            cls_name = yolo_names.get(yolo_cls, '')
            cat_id = name_to_cat_id.get(cls_name)
            if cat_id is None:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw, bh = x2 - x1, y2 - y1
            score = float(box.conf)

            pseudo_annotations.append({
                'id': next_ann_id,
                'image_id': next_img_id,
                'category_id': cat_id,
                'bbox': [x1, y1, bw, bh],
                'area': bw * bh,
                'iscrowd': 0,
                'score': score
            })
            next_ann_id += 1

        next_img_id += 1

        if idx % 100 == 0:
            print(f"  已处理 {idx}/{len(unannotated)} 张，生成 {len(pseudo_images)} 张伪标签图像")

    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'images': pseudo_images,
            'annotations': pseudo_annotations,
            'categories': categories
        }, f, ensure_ascii=False, indent=2)

    print(f"\n伪标签生成完成！")
    print(f"伪标签图像数: {len(pseudo_images)}")
    print(f"伪标签标注数: {len(pseudo_annotations)}")
    print(f"输出文件: {output_path}")


if __name__ == '__main__':
    main()
