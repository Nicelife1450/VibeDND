#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
减类数据集构建脚本
功能:合并近义类别、过滤稀有类别(<100 标注)、按保留类分层划分 train/val (8:2, seed 42)
输出:dataset/annotations_reduced.json + dataset/split_reduced.json (冻结验收集)
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter

# 近义类合并表: 原始类名 -> 合并后类名
MERGE_MAP = {
    '通道距树木距离不够': '距树木距离不够',
    '线路距树木距离不够': '距树木距离不够',
    '导线距树木距离不够': '距树木距离不够',
    '线夹绝缘罩脱落': '绝缘罩脱落',
    '绝缘罩脱落': '绝缘罩脱落',
    '绝缘护套损坏': '绝缘护套缺陷',
    '线夹绝缘护套缺失': '绝缘护套缺陷',
    '杆塔异物、鸟巢': '杆塔异物鸟巢',
    '杆塔本体异物、鸟巢': '杆塔异物鸟巢',
    '杆塔鸟巢': '杆塔异物鸟巢',
    '杆塔异物': '杆塔异物鸟巢',
    '道路边的杆塔未设防护设施': '杆塔未设防护设施',
    '道路边的杆塔未设保护设施': '杆塔未设防护设施',
}

# 独立保留的类(合并后标注 >= 100 的类会自动纳入, 此表仅用于注释与核对)
# 绑扎线不规范 847 / 绝缘子污秽 754 / 横担锈蚀 716 / 其他缺陷 505 / 绝缘线绝缘层破损 348
# 套管破损 317 / 避雷器破损 302 / 杆基杂物堆积 160 / 绝缘子固定不牢固 153 / 绝缘子破损 138
# 接地引下线连接不良 119


def main():
    parser = argparse.ArgumentParser(description='构建减类数据集 + 分层划分')
    parser.add_argument('--annotations', type=str, default='dataset/annotations.json')
    parser.add_argument('--min-count', type=int, default=100, help='合并后保留类别的最小标注数')
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='dataset/annotations_reduced.json')
    parser.add_argument('--split-output', type=str, default='dataset/split_reduced.json')
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.annotations, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    id_to_name = {c['id']: c['name'] for c in coco['categories']}

    # 1. 计算合并后各类标注数
    merged_name_of = {}  # 原始 category_id -> 合并后类名
    for cid, name in id_to_name.items():
        merged_name_of[cid] = MERGE_MAP.get(name, name)

    merged_counts = Counter(merged_name_of[a['category_id']] for a in coco['annotations'])
    kept_names = sorted([n for n, c in merged_counts.items() if c >= args.min_count])
    print(f'合并后共 {len(merged_counts)} 类, 保留 {len(kept_names)} 类 (>= {args.min_count} 标注):')
    for n in kept_names:
        print(f'  {merged_counts[n]:5d}  {n}')
    dropped = {n: c for n, c in merged_counts.items() if c < args.min_count}
    print(f'丢弃 {len(dropped)} 个稀有类, 共 {sum(dropped.values())} 条标注转为负样本')

    # 2. 重建 categories (新 ID 1..N) 并过滤/重映射 annotations
    new_id_of_name = {n: i + 1 for i, n in enumerate(kept_names)}
    new_categories = [{'id': new_id_of_name[n], 'name': n, 'supercategory': 'defect'}
                      for n in kept_names]

    new_annotations = []
    next_ann_id = 1
    for a in coco['annotations']:
        mname = merged_name_of[a['category_id']]
        if mname not in new_id_of_name:
            continue
        new_annotations.append({
            'id': next_ann_id,
            'image_id': a['image_id'],
            'category_id': new_id_of_name[mname],
            'bbox': a['bbox'],
            'area': a['area'],
            'iscrowd': a.get('iscrowd', 0),
        })
        next_ann_id += 1

    # 3. 多标签分层划分: 全局 val 封顶 val_fraction, 类间轮询分配保证各类公平获得 val 配额
    img_to_cats = defaultdict(set)
    for a in new_annotations:
        img_to_cats[a['image_id']].add(a['category_id'])
    cat_to_imgs = defaultdict(list)
    for img_id, cats in img_to_cats.items():
        for c in cats:
            cat_to_imgs[c].append(img_id)

    cat_total = Counter(a['category_id'] for a in new_annotations)
    val_target = {c: max(10, round(cat_total[c] * args.val_fraction)) for c in cat_total}
    val_cat_count = Counter()

    all_img_ids = [img['id'] for img in coco['images']]
    n_target_val = round(len(all_img_ids) * args.val_fraction)

    val_images, assigned = set(), set()
    pools = {}
    for c in cat_total:
        imgs = list(cat_to_imgs[c])
        random.shuffle(imgs)
        pools[c] = imgs
    classes_asc = sorted(cat_total, key=lambda c: cat_total[c])

    progress = True
    while len(val_images) < n_target_val and progress:
        progress = False
        for c in classes_asc:
            if len(val_images) >= n_target_val:
                break
            if val_cat_count[c] >= val_target[c]:
                continue
            # 从该类的池中取一张未分配的图
            while pools[c] and pools[c][-1] in assigned:
                pools[c].pop()
            if not pools[c]:
                continue
            i = pools[c].pop()
            val_images.add(i)
            assigned.add(i)
            for cc in img_to_cats[i]:
                val_cat_count[cc] += 1
            progress = True

    # 若全局 val 仍不足(理论上不会), 随机补足
    rest = [i for i in all_img_ids if i not in assigned]
    random.shuffle(rest)
    extra = max(0, n_target_val - len(val_images))
    val_images.update(rest[:extra])
    train_images = set(all_img_ids) - val_images

    # 4. 打印每类 train/val 实例数表
    train_cat_count = Counter(a['category_id'] for a in new_annotations if a['image_id'] in train_images)
    val_cat_count = Counter(a['category_id'] for a in new_annotations if a['image_id'] in val_images)
    name_of_id = {v: k for k, v in new_id_of_name.items()}
    print(f'\n{"类别":<20} {"总数":>6} {"train":>6} {"val":>6}')
    bad = []
    for cid in sorted(new_id_of_name.values()):
        n, t, v = cat_total[cid], train_cat_count[cid], val_cat_count[cid]
        print(f'{name_of_id[cid]:<20} {n:>6} {t:>6} {v:>6}')
        if v < 10:
            bad.append(name_of_id[cid])
    if bad:
        print(f'WARNING: 以下类 val 实例 < 10: {bad}')

    # 5. 输出
    out = {
        'info': coco.get('info', {}),
        'images': coco['images'],
        'annotations': new_annotations,
        'categories': new_categories,
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    with open(args.split_output, 'w', encoding='utf-8') as f:
        json.dump({
            'seed': args.seed,
            'val_fraction': args.val_fraction,
            'train': sorted(train_images),
            'val': sorted(val_images),
        }, f, ensure_ascii=False, indent=2)

    print(f'\n总图像: {len(all_img_ids)} | train: {len(train_images)} | val: {len(val_images)}')
    print(f'保留标注: {len(new_annotations)}')
    print(f'输出: {args.output}')
    print(f'划分: {args.split_output} (验收集冻结)')


if __name__ == '__main__':
    main()
