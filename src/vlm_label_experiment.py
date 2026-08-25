#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM 部件级标注实验: Qwen2.5-VL-7B 直接检测 DND 图像部件, 与 EPRI 伪标注对比

样本: 跨域验证 12 张 + 零检出 4 张 = 16 张
输出:
  experiments/vlm_exp/render/{i:02d}_{stem}.jpg  渲染图(蓝=VLM, 红=EPRI伪标注)
  experiments/vlm_exp/report.txt                  每图检出数 + 类匹配 IoU 统计
"""

import json
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

NAMES = ['insulator', 'pole', 'crossarm', 'cutout', 'transformer']
CN = {'绝缘子': 'insulator', '电杆': 'pole', '杆塔': 'pole', '横担': 'crossarm',
      '熔断器': 'cutout', '跌落式熔断器': 'cutout', '变压器': 'transformer',
      'insulator': 'insulator', 'pole': 'pole', 'crossarm': 'crossarm',
      'cutout': 'cutout', 'transformer': 'transformer'}

OUT = Path('experiments/vlm_exp')
(OUT / 'render').mkdir(parents=True, exist_ok=True)

# 样本: 与 EPRI 跨域验证同一批
sample_stems = [Path(p).stem for p in
                Path('/home/huyue/.claude/jobs/37c1933f/tmp/xdomain_sample.txt').read_text().splitlines()]
zero = Path('experiments/pseudo_zero_det.txt').read_text().splitlines()[:0]  # 占位
import random
random.seed(11)
zero4 = [Path(s).stem for s in random.sample(Path('experiments/pseudo_zero_det.txt').read_text().splitlines(), 4)]
stems = sample_stems + zero4
print(f'样本 {len(stems)} 张')

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-7B-Instruct', torch_dtype=torch.bfloat16,
    device_map='cuda:0', attn_implementation='eager')  # 1600px 上限下 eager 不会 OOM; sdpa 需 torch>=2.1.1
processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')

MAX_SIDE = 1024  # eager 注意力 fp32 softmax 峰值 ~5.7GB@1600px, 1024px 下 ~1GB 可放下
TMP = Path('/home/huyue/.claude/jobs/37c1933f/tmp/vlm_resized')
TMP.mkdir(parents=True, exist_ok=True)

PROMPT = ('这是一张配电网无人机巡检照片。请检测图中的电力部件：绝缘子(insulator)、电杆(pole)、'
          '横担(crossarm)、跌落式熔断器(cutout)、变压器(transformer)。'
          '对每一个检测到的部件，输出一行 JSON：{"label": 类别英文, "bbox_2d": [x1, y1, x2, y2]}，'
          '坐标为图像像素坐标。只输出 JSON 行，不要其他内容。没有检测到的类别不要输出。')


def parse_boxes(text, W, H):
    """解析 JSON 行 + 容错 Qwen 特殊标记格式, 坐标归一化到 0-1"""
    boxes = []
    # JSON 行
    for m in re.finditer(r'\{[^{}]*"bbox_2d"[^{}]*\}', text):
        try:
            d = json.loads(m.group(0))
            label = str(d.get('label', '')).lower().strip()
            bb = d['bbox_2d']
            cls = next((v for k, v in CN.items() if k in label), None)
            if cls and len(bb) == 4:
                boxes.append((cls, [float(v) for v in bb]))
        except Exception:
            pass
    # <|box_start|>(x1,y1),(x2,y2)<|box_end|> 格式
    for m in re.finditer(r'object_ref_start\|>([^<]+)<\|object_ref_end\|><\|box_start\|>\((\d+),\s*(\d+)\),\((\d+),\s*(\d+)\)', text):
        cls = next((v for k, v in CN.items() if k in m.group(1).lower()), None)
        if cls:
            boxes.append((cls, [float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))]))
    # 坐标尺度自适应: >2 视为像素(可能基于缩放图, 按比例夹取)
    out = []
    for cls, (x1, y1, x2, y2) in boxes:
        if max(x1, y1, x2, y2) <= 2:      # 已归一化
            pass
        elif max(x1, y1, x2, y2) <= 1000:  # 0-1000 制
            x1, y1, x2, y2 = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
        else:                              # 像素坐标(按缩放图, 直接按原图比例除)
            x1, x2, y1, y2 = x1 / W, x2 / W, y1 / H, y2 / H
        x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
        y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
        if (x2 - x1) * (y2 - y1) > 1e-5:
            out.append((cls, x1, y1, x2, y2))
    return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


report = []
for i, stem in enumerate(stems):
    img_path = Path('dataset/images') / f'{stem}.jpg'
    # 预缩放到长边 1600(可控坐标系 + 控制 vision token 数)
    im0 = Image.open(img_path).convert('RGB')
    W0, H0 = im0.size
    im0.thumbnail((MAX_SIDE, MAX_SIDE))
    rimg = TMP / f'{stem}.jpg'
    im0.save(rimg, quality=92)
    W, H = im0.size
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': str(rimg)},
        {'type': 'text', 'text': PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors='pt').to('cuda:0')
    out_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    resp = processor.batch_decode(out_ids[:, inputs['input_ids'].shape[1]:],
                                  skip_special_tokens=True)[0]
    vlm_boxes = parse_boxes(resp, W, H)

    # EPRI 伪标注 (raw, conf>=0.25)
    epri_boxes = []
    raw_f = Path('experiments/pseudo_labels_raw') / f'{stem}.txt'
    if raw_f.exists():
        for ln in raw_f.read_text().splitlines():
            p = ln.split()
            c, x, y, w, h = int(p[0]), *map(float, p[1:5])
            epri_boxes.append((NAMES[c], x - w / 2, y - h / 2, x + w / 2, y + h / 2))

    # 类匹配 IoU>=0.3
    match_ve = sum(1 for vc, *vb in vlm_boxes
                   if any(ec == vc and iou(vb, eb) >= 0.3 for ec, *eb in epri_boxes))
    match_ev = sum(1 for ec, *eb in epri_boxes
                   if any(vc == ec and iou(vb, eb) >= 0.3 for vc, *vb in vlm_boxes))
    report.append(f'{stem}: VLM {len(vlm_boxes)} 框 / EPRI {len(epri_boxes)} 框 | '
                  f'VLM命中EPRI {match_ve}/{len(vlm_boxes)}, EPRI被命中 {match_ev}/{len(epri_boxes)}')
    print(report[-1])

    # 渲染: 蓝=VLM, 红=EPRI (框已归一化, 直接乘原图尺寸)
    im = Image.open(img_path).convert('RGB')
    OW, OH = im.size
    im.thumbnail((1300, 1300))
    sc = im.size[0] / OW
    dr = ImageDraw.Draw(im)
    for cls, x1, y1, x2, y2 in vlm_boxes:
        dr.rectangle([x1 * OW * sc, y1 * OH * sc, x2 * OW * sc, y2 * OH * sc], outline='blue', width=3)
        dr.text((x1 * OW * sc + 2, y1 * OH * sc + 2), f'V:{cls[:4]}', fill='blue')
    for cls, x1, y1, x2, y2 in epri_boxes:
        dr.rectangle([x1 * OW * sc, y1 * OH * sc, x2 * OW * sc, y2 * OH * sc], outline='red', width=2)
    im.save(OUT / 'render' / f'{i:02d}_{stem}.jpg', quality=88)

(OUT / 'report.txt').write_text('\n'.join(report))
print(f'\n渲染图 → {OUT}/render/ | 报告 → {OUT}/report.txt')
