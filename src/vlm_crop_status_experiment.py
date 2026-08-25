#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM 实验二: Qwen2.5-VL 作为级联 Stage-2 状态分类器(部件裁剪图 → 缺陷判断)

从 EPRI 伪标注(conf>=0.5)裁剪部件图, 逐张问 VLM 状态(正常/污秽/破损/锈蚀/其他异常+理由)。
裁剪图小 → VLM 的细粒度判断才是它的战场; 全图检测是它的弱项(实验一已证)。
输出: experiments/vlm_exp2/crops/{i:02d}_{cls}_{stem}.jpg + verdicts.txt
"""

import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

NAMES = ['insulator', 'pole', 'crossarm', 'cutout', 'transformer']
OUT = Path('experiments/vlm_exp2')
(OUT / 'crops').mkdir(parents=True, exist_ok=True)
TMP = Path('/home/huyue/.claude/jobs/37c1933f/tmp/vlm_crops')
TMP.mkdir(parents=True, exist_ok=True)

# 挑 10 个高置信部件框(绝缘子为主, 加横担/杆), 跨不同图像
import random
random.seed(5)
cands = []
for f in sorted(Path('experiments/pseudo_labels_raw').iterdir()):
    for ln in f.read_text().splitlines():
        p = ln.split()
        c, conf = int(p[0]), float(p[5])
        if conf >= 0.55 and c in (0, 1, 2):
            x, y, w, h = map(float, p[1:5])
            if w * h > 0.0005:  # 太小的框裁出来没内容
                cands.append((f.stem, c, x, y, w, h, conf))
random.shuffle(cands)
picks = []
seen = set()
for stem, c, x, y, w, h, conf in cands:
    key = (stem, c)
    if key in seen:
        continue
    seen.add(key)
    picks.append((stem, c, x, y, w, h, conf))
    if len(picks) >= 10:
        break

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-7B-Instruct', torch_dtype=torch.bfloat16,
    device_map='cuda:0', attn_implementation='eager')
processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')

CN_CLS = {'insulator': '绝缘子', 'pole': '电杆', 'crossarm': '横担'}

verdicts = []
for i, (stem, c, x, y, w, h, conf) in enumerate(picks):
    cls = NAMES[c]
    im = Image.open(Path('dataset/images') / f'{stem}.jpg').convert('RGB')
    W, H = im.size
    # 扩 30% 边距裁剪
    pw, ph = w * W, h * H
    x1 = max(0, x * W - 0.3 * pw); x2 = min(W, x * W + 1.3 * pw)
    y1 = max(0, y * H - 0.3 * ph); y2 = min(H, y * H + 1.3 * ph)
    crop = im.crop((x1, y1, x2, y2))
    crop.thumbnail((1024, 1024))
    cpath = TMP / f'{i:02d}_{cls}_{stem}.jpg'
    crop.save(cpath, quality=92)

    prompt = (f'这是从中国配电网无人机巡检照片中裁剪的{CN_CLS[cls]}部件图。'
              '请仔细观察并判断该部件的状态，从以下选项中选择最符合的一项：'
              '正常 / 污秽 / 破损 / 锈蚀 / 异物附着 / 其他异常。'
              '先用一个选项词回答，然后用一句话说明判断依据（看到了什么）。')
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': str(cpath)},
        {'type': 'text', 'text': prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors='pt').to('cuda:0')
    out_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    resp = processor.batch_decode(out_ids[:, inputs['input_ids'].shape[1]:],
                                  skip_special_tokens=True)[0].strip()
    verdicts.append(f'[{i:02d}] {cls} ({stem}, conf={conf:.2f}): {resp}')
    print(verdicts[-1])
    # 保存带标签的裁剪图
    d = ImageDraw.Draw(crop)
    d.rectangle([0, 0, crop.size[0], 22], fill='white')
    d.text((3, 4), resp.split('。')[0][:60], fill='black')
    crop.save(OUT / 'crops' / f'{i:02d}_{cls}_{stem}.jpg', quality=92)

(OUT / 'verdicts.txt').write_text('\n\n'.join(verdicts))
print(f'\n→ {OUT}/crops/ + verdicts.txt')
