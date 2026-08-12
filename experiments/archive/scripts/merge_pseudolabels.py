#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并伪标签脚本
功能：将教师模型生成的伪标签与原始标注合并为一个新的 COCO 数据集
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='合并伪标签与原始标注')
    parser.add_argument('--coco', type=str, default='dataset/annotations.json',
                        help='原始 COCO 标注文件')
    parser.add_argument('--pseudo', type=str, default='dataset/pseudo_annotations.json',
                        help='伪标签 COCO 文件')
    parser.add_argument('--output', type=str, default='dataset/annotations_merged.json',
                        help='合并后的输出文件')
    args = parser.parse_args()

    with open(args.coco, 'r', encoding='utf-8') as f:
        labeled = json.load(f)
    with open(args.pseudo, 'r', encoding='utf-8') as f:
        pseudo = json.load(f)

    max_img_id = max(img['id'] for img in labeled['images'])
    max_ann_id = max(a['id'] for a in labeled['annotations'])

    for img in pseudo['images']:
        img['id'] += max_img_id
    for ann in pseudo['annotations']:
        ann['id'] += max_ann_id
        ann['image_id'] += max_img_id

    labeled['images'].extend(pseudo['images'])
    labeled['annotations'].extend(pseudo['annotations'])

    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)

    print(f"合并完成！")
    print(f"原始图像: {len(labeled['images']) - len(pseudo['images'])}")
    print(f"伪标签图像: {len(pseudo['images'])}")
    print(f"总图像数: {len(labeled['images'])}")
    print(f"总标注数: {len(labeled['annotations'])}")
    print(f"输出文件: {output_path}")


if __name__ == '__main__':
    main()
