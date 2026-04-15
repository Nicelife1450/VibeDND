#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO 格式转换为 YOLO 格式
功能：将 COCO annotations.json 转换为 YOLO 格式，支持训练/验证划分
"""

import json
import os
import random
import shutil
from pathlib import Path
from collections import defaultdict


# 合并阈值：少于 N 个标注的类别合并到"其他缺陷"
MERGE_THRESHOLD = 10


def coco_to_yolo_bbox(bbox, img_width, img_height):
    """
    COCO bbox [x, y, w, h] -> YOLO bbox [x_center, y_center, width, height]
    所有值归一化到 0-1
    """
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height


def main():
    random.seed(42)

    # 路径配置
    base_dir = Path('dataset')
    images_dir = base_dir / 'images'
    coco_ann_file = base_dir / 'annotations.json'

    output_dir = base_dir / 'yolo'
    output_images_train = output_dir / 'images' / 'train'
    output_images_val = output_dir / 'images' / 'val'
    output_labels_train = output_dir / 'labels' / 'train'
    output_labels_val = output_dir / 'labels' / 'val'

    # 创建目录
    for d in [output_images_train, output_images_val, output_labels_train, output_labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    # 加载 COCO 数据
    print("Loading COCO annotations...")
    with open(coco_ann_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 构建 ID -> category 映射
    id_to_cat = {cat['id']: cat for cat in coco_data['categories']}
    id_to_img = {img['id']: img for img in coco_data['images']}

    # 统计每个类别的标注数量
    cat_counts = defaultdict(int)
    for ann in coco_data['annotations']:
        cat_counts[ann['category_id']] += 1

    # 确定需要合并的类别
    categories_to_merge = set()
    for cid, count in cat_counts.items():
        if count < MERGE_THRESHOLD:
            categories_to_merge.add(cid)

    print(f"Categories with < {MERGE_THRESHOLD} annotations to merge: {len(categories_to_merge)}")

    # "其他缺陷" 的 ID
    other_defect_id = None
    for cat in coco_data['categories']:
        if cat['name'] == '其他缺陷':
            other_defect_id = cat['id']
            break

    if other_defect_id is None:
        print("Warning: '其他缺陷' category not found, will use first available ID")
        other_defect_id = max(id_to_cat.keys()) + 1

    print(f"'其他缺陷' category ID: {other_defect_id}")

    # 构建新的类别列表（合并后的）
    # 保留所有标注数 >= threshold 的类别，以及"其他缺陷"
    new_categories = []
    new_cat_names = []
    cat_id_mapping = {}  # old_id -> new_id (for merged categories)

    next_id = 0
    for cat in coco_data['categories']:
        if cat['id'] == other_defect_id:
            continue  # 先跳过，稍后添加
        if cat['id'] not in categories_to_merge:
            cat_id_mapping[cat['id']] = next_id
            new_categories.append({
                'id': next_id,
                'name': cat['name'],
                'supercategory': cat['supercategory']
            })
            new_cat_names.append(cat['name'])
            next_id += 1

    # 添加"其他缺陷"
    cat_id_mapping[other_defect_id] = next_id
    new_categories.append({
        'id': next_id,
        'name': '其他缺陷',
        'supercategory': 'defect'
    })
    new_cat_names.append('其他缺陷')
    next_id += 1

    # 对所有被合并的类别 -> 其他缺陷
    for cid in categories_to_merge:
        if cid != other_defect_id:
            cat_id_mapping[cid] = cat_id_mapping[other_defect_id]

    print(f"Original categories: {len(coco_data['categories'])}")
    print(f"New categories: {len(new_categories)}")
    print(f"Categories merged: {len(categories_to_merge)}")

    # 构建 image_id -> annotations 映射
    img_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    # 划分训练/验证集 (8:2)
    all_images = coco_data['images'].copy()
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")

    # 创建软链和标签文件
    def process_images(images, labels_dir, images_output_dir, images_input_dir):
        count = 0
        for img_info in images:
            img_id = img_info['id']
            file_name = img_info['file_name']
            img_width = img_info['width']
            img_height = img_info['height']

            # 创建软链
            src = images_input_dir / file_name
            dst = images_output_dir / file_name
            if not dst.exists():
                os.symlink(src.resolve(), dst)

            # 创建标签文件
            label_file = labels_dir / f"{Path(file_name).stem}.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                for ann in img_to_anns.get(img_id, []):
                    old_cat_id = ann['category_id']
                    new_cat_id = cat_id_mapping[old_cat_id]

                    # 跳过"其他缺陷"类别中的标注（因为它们本来就是要合并的）
                    # 但如果原始类别就是其他缺陷，保留
                    if old_cat_id in categories_to_merge and old_cat_id != other_defect_id:
                        # 这些被合并了，但不在其他缺陷中出现
                        pass

                    x_center, y_center, width, height = coco_to_yolo_bbox(
                        ann['bbox'], img_width, img_height
                    )

                    # 限制在 0-1 范围内
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))

                    f.write(f"{new_cat_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            count += 1

        return count

    print("\nProcessing training set...")
    train_count = process_images(train_images, output_labels_train, output_images_train, images_dir)
    print(f"Training labels created: {train_count}")

    print("\nProcessing validation set...")
    val_count = process_images(val_images, output_labels_val, output_images_val, images_dir)
    print(f"Validation labels created: {val_count}")

    # 创建 dataset.yaml
    yaml_content = f"""# YOLO Dataset Configuration
path: {output_dir.resolve()}
train: images/train
val: images/val

# Classes ({len(new_categories)} categories after merging rare categories)
names:
"""
    for i, cat_name in enumerate(new_cat_names):
        yaml_content += f"  {i}: {cat_name}\n"

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\nDataset YAML saved: {yaml_path}")

    # 打印类别统计
    print("\n" + "=" * 60)
    print("Category Statistics After Merging:")
    print("=" * 60)

    # 重新统计
    new_cat_counts = defaultdict(int)
    for ann in coco_data['annotations']:
        old_cat_id = ann['category_id']
        new_cat_id = cat_id_mapping[old_cat_id]
        new_cat_counts[new_cat_id] += 1

    for i, name in enumerate(new_cat_names):
        count = new_cat_counts.get(i, 0)
        marker = " (merged)" if name == "其他缺陷" else ""
        print(f"  {i:3d}: {count:5d} | {name}{marker}")

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"YOLO dataset location: {output_dir.resolve()}")
    print(f"\nTo train YOLOv8m:")
    print(f"  yolo detect train model=yolov8m.pt data={yaml_path} epochs=100 imgsz=1280 batch=8")


if __name__ == '__main__':
    main()
