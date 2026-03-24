#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集可视化脚本
功能：可视化COCO格式数据集中的标注框
"""

import json
import cv2
import random
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def visualize_annotations(dataset_path, output_dir, num_samples: int = 10):
    """
    可视化数据集标注
    
    Args:
        dataset_path: annotations.json文件路径
        output_dir: 输出目录
        num_samples: 可视化样本数量
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    images_dir = dataset_path.parent / 'images'
    
    cat_id_to_name = {cat['id']: cat['name'] for cat in dataset['categories']}
    img_id_to_info = {img['id']: img for img in dataset['images']}
    
    img_to_anns = {}
    for ann in dataset['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    sample_img_ids = random.sample(list(img_to_anns.keys()), 
                                   min(num_samples, len(img_to_anns)))
    
    print(f"可视化 {len(sample_img_ids)} 张图像...")
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 24)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", 24)
        except:
            font = ImageFont.load_default()
    
    for i, img_id in enumerate(sample_img_ids):
        img_info = img_id_to_info[img_id]
        img_path = images_dir / img_info['file_name']
        
        if not img_path.exists():
            print(f"图像不存在: {img_path}")
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        anns = img_to_anns[img_id]
        for ann in anns:
            x, y, w, h = ann['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            draw.rectangle([x, y, x+w, y+h], outline=(0, 255, 0), width=3)
            
            cat_name = cat_id_to_name[ann['category_id']]
            label = f"{cat_name}"
            
            bbox = draw.textbbox((x, y), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            draw.rectangle([x, y-text_h-10, x+text_w+10, y], fill=(0, 255, 0))
            
            draw.text((x+5, y-text_h-5), label, fill=(0, 0, 0), font=font)
        
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        output_path = output_dir / f"vis_{i+1:03d}_{img_info['file_name']}"
        cv2.imwrite(str(output_path), img_bgr)
        print(f"保存: {output_path}")
    
    print(f"\n可视化完成！输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='可视化COCO格式数据集')
    parser.add_argument('--dataset', type=str, default='dataset/annotations.json',
                       help='annotations.json文件路径')
    parser.add_argument('--output', type=str, default='dataset/visualizations',
                       help='输出目录')
    parser.add_argument('--num', type=int, default=10,
                       help='可视化样本数量')
    
    args = parser.parse_args()
    
    visualize_annotations(args.dataset, args.output, args.num)


if __name__ == '__main__':
    main()
