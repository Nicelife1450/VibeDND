#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配网无人机图像缺陷检测数据集构建脚本
功能：扫描巡检报告目录，提取缺陷图像和标注，生成COCO格式数据集
"""

import os
import re
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Category Normalization (merged from cleanup_dataset.py) ====================

def strip_parentheses(name: str) -> str:
    """移除类别名称中的括号内容"""
    name = re.sub(r'（[^）]*）', '', name)
    name = re.sub(r'\([^)]*\)', '', name)
    return name.strip()


def normalize_defect_name(name: str) -> str:
    """规范化缺陷类别名称"""
    name = name.strip()
    name = strip_parentheses(name)

    # 修复重复词汇
    if '横担横担' in name:
        name = name.replace('横担横担', '横担')
    if '导线导线' in name:
        name = name.replace('导线导线', '导线')
    if '基础基础' in name:
        name = name.replace('基础基础', '基础')
    if '横担锈蚀锈蚀' in name:
        name = name.replace('横担锈蚀锈蚀', '横担锈蚀')

    # 修复缺失前缀
    if name.startswith('基杂物堆积') or name.startswith('杂物堆积'):
        name = '杆基杂物堆积'
    if name.startswith('杆横担'):
        name = name.replace('杆横担', '横担')

    # 修复拼写错误
    if '松驰' in name:
        name = name.replace('松驰', '松弛')
    if '住上' in name:
        name = name.replace('住上', '柱上')
    if '通道树距木' in name:
        name = name.replace('通道树距木', '通道距树木')

    # 修复柱上开关类
    if '柱上' in name and '开关' in name:
        name = re.sub(r'柱上[Ss][Ff]\d*开关', '柱上开关', name)
        name = re.sub(r'柱上(\d+)开关', r'柱上开关', name)
        if '套管' in name:
            name = name.replace('柱上开关套管', '柱上开关破损')
        if '绝缘罩脱落' in name or '无绝缘罩' in name:
            name = name.replace('柱上开关', '柱上开关破损')

    return name.strip()


def should_merge_to_other(name: str) -> tuple:
    """判断是否应合并到其他缺陷"""
    if re.match(r'^\d{2}-\d{4}', name):
        return True, '其他缺陷'
    if re.search(r'线#\d+.*线#\d+', name):
        return True, '通道距树木距离不够'
    if re.search(r'.+\d{2}-\d{4}-\d{2,4}.+距树木距离不够', name):
        return True, '导线距树木距离不够'
    if '线#' in name and '距树木距离不够' in name:
        return True, '通道距树木距离不够'
    return False, None


SPECIFIC_MAPPINGS = {
    '横担锈蚀（起皮和严重麻点，锈蚀面积超过2分之1）': '横担锈蚀',
    '横担锈蚀（起皮和严重麻点，锈蚀面积超过12）': '横担锈蚀',
    '横担锈蚀（起皮和严重麻点，锈蚀面积超过2分之1）弯曲': '横担锈蚀',
    '横担锈蚀（锈蚀面积超过12）': '横担锈蚀',
    '横担锈蚀锈蚀': '横担锈蚀',
    '横担锈蚀锈蚀（锈蚀面积超过12）': '横担锈蚀',
    '横担锈蚀锈蚀（锈蚀面积超过2分之一）': '横担锈蚀',
    '横担弯曲变形': '横担弯曲',
    '横担横担弯曲、倾斜、变形': '横担弯曲',
    '横担横担弯曲、倾斜、变形（固定不牢固）': '横担弯曲',
    '横担横担弯曲、倾斜、变形（电缆头固定不规范）': '横担弯曲',
    '横担松动、主件脱落（支撑角铁脱落）': '横担松动、主件脱落',
    '横担松动、主件脱落（横担位移，与导线不在垂直角度）': '横担松动、主件脱落',
    '横担倾斜': '横担弯曲',
    '横担变形': '横担弯曲',
    '杆基杂物堆积（杆塔被建筑包围，违章搭盖）': '杆基杂物堆积',
    '杆基杂物堆积（杆塔被铁皮）': '杆基杂物堆积',
    '杆基杂物堆积（杆塔台变被树包围）': '杆基杂物堆积',
    '杆基杂物堆积（违章搭盖）': '杆基杂物堆积',
    '杂物堆积（巡视通道受阻）': '杆基杂物堆积',
    '杂物堆积': '杆基杂物堆积',
    '通道杂物堆积（电缆头刀闸操作位置有树木杂草）': '通道杂物堆积',
    '绑扎线不规范（使用裸绑扎线）': '绑扎线不规范',
    '绑扎线不规范（绑扎为裸铝线）': '绑扎线不规范',
    '绑扎线不规范（松动）': '绑扎线不规范',
    '绑扎线不规范（松弛）': '绑扎线不规范',
    '绑扎线不规范（脱落）': '绑扎线不规范',
    '导线绑扎线不规范': '绑扎线不规范',
    '绑扎带导线未绑扎': '绑扎线不规范',
    '导线未绑扎': '绑扎线不规范',
    '帮扎带导线未绑扎': '绑扎线不规范',
    '绑扎带绑扎线不规范': '绑扎线不规范',
    '绑扎带绑扎线不规范（绑扎固定配件锈蚀严重）': '绑扎线不规范',
    '绑扎带绑扎线不规范（绑扎线无绝缘层）': '绑扎线不规范',
    '绑扎带导线未绑扎': '绑扎线不规范',
    '通道树距木距离不够': '通道距树木距离不够',
    '绝缘子污秽（放电痕迹）': '绝缘子污秽',
    '绝缘子固定不牢固（倾斜）': '绝缘子固定不牢固',
    '绝缘子破损（损坏）': '绝缘子破损',
    '绝缘子釉表面脱落': '绝缘子破损',
    '绝缘子破损（同杆另一回路）': '绝缘子破损',
    '绝缘子倾斜': '绝缘子固定不牢固',
    '绝缘线绝缘层破损（裸导线）': '绝缘线绝缘层破损',
    '绝缘线绝缘层破损（现场为裸导线）': '绝缘线绝缘层破损',
    '绝缘线绝缘层破损（破皮）': '绝缘线绝缘层破损',
    '绝缘线绝缘层破损（引流线破皮）': '绝缘线绝缘层破损',
    '绝缘线绝缘层破损（老化)': '绝缘线绝缘层破损',
    '绝缘线绝缘层破损（有发热发损现象）': '绝缘线绝缘层破损',
    '绝缘线绝缘层破损（有放电痕迹）': '绝缘线绝缘层破损',
    '导线绝缘线绝缘层破损': '绝缘线绝缘层破损',
    '线路大号侧通道距树木距离不够（树碰线）': '通道距树木距离不够',
    '线路大号侧通道距树木距离不够(树枝距裸导线距离0.5米)': '通道距树木距离不够',
    '线路大号侧通道距树木距离不够（树枝距绝缘导线距离0.2米）': '通道距树木距离不够',
    '线路大号侧通道距树木距离不够（树压线）': '通道距树木距离不够',
    '线路小号侧通道距树木距离不够（树碰线）': '通道距树木距离不够',
    '线路大号侧通道距树木距离不够': '通道距树木距离不够',
    '线路小号侧通道距树木距离不够': '通道距树木距离不够',
    '小号侧通道距树木距离不够': '通道距树木距离不够',
    '大号侧通道距树木距离不够': '通道距树木距离不够',
    '导线距树木距离不够': '导线距树木距离不够',
    '线路大号侧距树木距离不够（树碰线）': '线路距树木距离不够',
    '线路大号侧距树木距离不够（距树枝距离0.2米）': '线路距树木距离不够',
    '线路大号侧距树木距离不够（距树枝距离0.3米）': '线路距树木距离不够',
    '线路大号侧距树木距离不够（距树枝距离0.5米）': '线路距树木距离不够',
    '线路大号侧距树木距离不够（距树枝距离0.1米）': '线路距树木距离不够',
    '线路大号侧距树木距离不够（距树木0.5米）': '线路距树木距离不够',
    '线路大号侧距树木距离不够': '线路距树木距离不够',
    '线路小号侧距树木距离不够（树碰线）': '线路距树木距离不够',
    '线路小号侧距树木距离不够（距树枝距离0.2米）': '线路距树木距离不够',
    '线路小号侧距树木距离不够（距树枝距离0.3米）': '线路距树木距离不够',
    '线路小号侧距树木距离不够（距树枝距离0.5米）': '线路距树木距离不够',
    '线路小号侧距树木距离不够（导线距树枝0.2米）': '线路距树木距离不够',
    '线路小号侧距树木距离不够（距离树枝0.2米）': '线路距树木距离不够',
    '线路小号侧距树木距离不够': '线路距树木距离不够',
    '边相距树木距离不够（距树枝距离0.4米）': '线路距树木距离不够',
    '边相距树木距离不够（距树枝距离0.5米）': '线路距树木距离不够',
    '大线路大号侧距树木距离不够（树碰线）': '线路距树木距离不够',
    '距树木距离不够（藤蔓爬上杆顶）': '通道距树木距离不够',
    '距树木距离不够（藤上杆）': '通道距树木距离不够',
    '通道距树木距离不够（藤上杆）': '通道距树木距离不够',
    '距树木距离不够': '通道距树木距离不够',
    '绝缘罩脱落（线夹裸露）': '绝缘罩脱落',
    '绝缘罩脱落(电缆头接线柱处无绝缘罩)': '绝缘罩脱落',
    '线夹绝缘罩脱落(电缆头接线柱处无绝缘罩)': '线夹绝缘罩脱落',
    '柱上开关绝缘罩脱落': '绝缘罩脱落',
    '线夹主件松动或脱落（柱上开关线夹断一相）': '线夹松动',
    '套管破损（电缆头无绝缘护套）': '套管破损',
    '套管破损（柱上开关无绝缘护套）': '套管破损',
    '套管破损（高压接线柱无绝缘护套）': '套管破损',
    '套管破损（缺失）': '套管破损',
    '柱上真空开关套管破损（缺失）': '套管破损',
    '柱上真空开关套管破损': '套管破损',
    '真空断路器套管破损（缺失）': '套管破损',
    '高、低压套管破损（高低压无绝缘护套）': '套管破损',
    '配变高、低压套管（缺失）': '套管破损',
    '配变高、低压套管（缺失无绝缘包扎）': '套管破损',
    '套管缺绝缘罩': '套管破损',
    '套管缺绝缘套管（互感器）': '套管破损',
    '套管缺绝缘套管': '套管破损',
    '避雷器破损（锈蚀）': '避雷器破损',
    '避雷器破损': '避雷器破损',
    '避雷器老旧': '避雷器破损',
    '避雷器烧毁一相': '避雷器破损',
    '避雷器损坏一相': '避雷器破损',
    '避雷器缺失一相': '避雷器破损',
    '避雷器缺失两相': '避雷器破损',
    '避雷器无引线': '避雷器破损',
    '接地引下线连接不良(上引线缺失)': '接地引下线连接不良',
    '接地引下线连接不良(上引线缺失，引下线断)': '接地引下线连接不良',
    '接地引下线连接不良（引上线未接入）': '接地引下线连接不良',
    '接地引下线连接不良（断）': '接地引下线连接不良',
    '刀闸锈蚀（老旧）': '刀闸锈蚀',
    '拉线松弛（接线松）': '拉线松弛',
    '松弛（拉线）': '拉线松弛',
    '鸟巢': '杆塔鸟巢',
    '杆上有鸟巢': '杆塔鸟巢',
    '杆塔本体鸟巢': '杆塔鸟巢',
    '杆塔异物': '杆塔异物',
    '导线上有异物': '导线异物',
    '异物（枯枝）': '杆塔异物',
    '异物（藤上杆）': '杆塔异物',
    '异物（拉线有藤包围）': '杆塔异物',
    '异物（杆上有一导线）': '杆塔异物',
    '破损（避雷器）': '破损',
    '破损（绝缘套管缺失）': '破损',
    '破损（柱上开关电缆头无绝缘罩无绝缘包扎：电气裸露）': '破损',
    '破损（柱上开关无绝缘护套）': '破损',
    '破损（电缆头无绝缘护套）': '破损',
    '破损（低压侧套管缺失）': '破损',
    '破损（住上开关无绝缘护套）': '破损',
    '杆塔本体异物、岛巢（岛巢）': '杆塔异物、鸟巢',
    '杆塔本体异物、岛巢（废弃的绝缘子未拆除）': '杆塔异物、鸟巢',
    '杆塔本体异物、岛巢（岛巢已碰到导线）': '杆塔异物、鸟巢',
    '杆塔本体异物、岛巢（废旧绝缘子放在横担上）': '杆塔异物、鸟巢',
    '杆塔本体异物、岛巢（干枯的爬藤）': '杆塔异物、鸟巢',
    '杆塔本体异物、岛巢（废弃的绝缘子或避雷器未拆除）': '杆塔异物、鸟巢',
    '绝缘护套损坏（导线为裸导线）': '绝缘护套损坏',
    '金属氧化物避雷器本体污秽': '避雷器破损',
    '金属氧化物避雷器破损（缺失）': '避雷器破损',
    '金属氧化物避雷器锈蚀': '避雷器破损',
    '故障指示器安装不牢靠（老旧）': '故障指示器安装不牢靠',
    '标识错误（现场杆号与图纸不符）': '标识错误',
    '标示错误（现场杆号与图纸不符）': '标识错误',
    '标识错误（现场与系统图不符，系统09杆无用户需异动）': '标识错误',
    '标识错误（图纸为刀闸，现场是跌落式）': '标识错误',
    '连接金具球头锈蚀严重(并钩线夹有烧焦痕迹)': '连接金具锈蚀',
    '连接金具球头锈蚀严重（同杆另一回路并钩线夹有烧焦痕迹）': '连接金具锈蚀',
    '连接金具球头锈蚀严重': '连接金具锈蚀',
    '连接金具锈蚀严重': '连接金具锈蚀',
    '安装不牢靠（防雷装置与穿刺体错位）': '安装不牢靠',
    '安装不牢靠（防雷装置一相缺失）': '安装不牢靠',
    '安装不牢靠（防雷装置两相缺失）': '安装不牢靠',
    '安装不牢靠（防雷装置中相缺失）': '安装不牢靠',
    '固定不牢固（倾斜）': '固定不牢固',
    '锈蚀（起皮和严重麻点，锈蚀面积超过）': '锈蚀',
    '锈蚀（起皮和严重麻点，锈蚀面积超过12）': '锈蚀',
    '锈蚀（起皮和严重麻点，锈蚀面积超过2分之1）': '锈蚀',
    '锈蚀（刀闸老旧）': '锈蚀',
    '锈蚀（销户未拆除）': '锈蚀',
    '锈蚀（导线有灼烧的痕迹）': '锈蚀',
    '钢绞线松弛（接线松）': '钢绞线松弛',
    '高低压套管缺绝缘罩(变压器器身生绣严重)': '高低压套管缺绝缘罩',
    '高压接线柱无绝缘护套': '高低压套管缺绝缘罩',
    '高低压接线柱无绝缘护套': '高低压套管缺绝缘罩',
    '柱上开关无绝缘护套': '柱上开关破损',
    '柱上SF6开关套管（缺失）': '柱上开关破损',
    '柱上SF6开关线夹绝缘罩脱落': '柱上开关破损',
    '柱上开关套管破损(缺失)': '柱上开关破损',
    '基础沉降（电缆井盖缺失）': '基础沉降',
    '距建筑物距离不够（杆被建筑物包围通道被堵）': '距建筑物距离不够',
    '导线距建筑物距离不够': '距建筑物距离不够',
    '通道距建筑物距离不够': '距建筑物距离不够',
    '柱上SF6开关套管': '柱上开关破损',
    '柱上SF6开关线夹绝缘罩脱落': '线夹绝缘罩脱落',
    '柱上真空开关破损': '柱上开关破损',
    '柱上真空开关无绝缘罩': '柱上开关破损',
    '柱上真空开关套管破损': '柱上开关破损',
    '高、低压套管破损': '套管破损',
    '配电变压器高、低压套管': '套管破损',
    '配变边相高压套管': '套管破损',
    '高低压绕组缺绝缘罩': '套管破损',
    '高低压套管缺绝缘罩': '套管破损',
    '柱上6开关套管': '柱上开关破损',
    '柱上６开关套管': '柱上开关破损',
    '柱上开关套管': '柱上开关破损',
    '柱上6开关线夹绝缘罩脱落': '线夹绝缘罩脱落',
    '柱上６开关线夹绝缘罩脱落': '线夹绝缘罩脱落',
    '线夹严重锈蚀': '线夹锈蚀',
    '线夹松动': '线夹绝缘罩脱落',
    '线夹绝缘护罩脱落': '线夹绝缘罩脱落',
    '线夹绝缘护套脱落': '线夹绝缘罩脱落',
    '线夹护套缺失': '线夹绝缘护套缺失',
    '横担抱箍锈蚀': '横担锈蚀',
    '横担松动、主件脱落': '横担弯曲',
    '导线锈蚀': '锈蚀',
    '导线灯笼现象': '导线异物',
    '导线脱落': '导线异物',
    '杆塔上有遗留工器具': '杆塔异物',
    '杆塔本体杆塔上有遗留工器具、金具': '杆塔异物',
    '杆塔基础沉降': '基础沉降',
    '无标识或缺少标识': '标识错误',
    '绑扎线不规范（绑扎线松弛)': '绑扎线不规范',
    '绝缘罩固定不牢固': '绝缘罩脱落',
    '绝缘罩污秽': '绝缘子污秽',
    '防雷金具锈蚀': '避雷器破损',
    '防雷金具安装不牢靠': '安装不牢靠',
    '拉线防护设施不满足要求': '拉线松弛',
    '松动': '横担弯曲',
    '柱上开关绝缘护套缺失': '柱上开关破损',
    '柱上开关边相跑线距树木距离不够': '线路距树木距离不够',
    '柱上开关电缆头无绝缘罩无绝缘包扎：电气裸露）': '柱上开关破损',
    '杆被树枝包围线路距树木距离不够': '线路距树木距离不够',
    '杆距树木距离不够': '线路距树木距离不够',
    '通道距树木距离不够(#015-#016之间距树木距离小于0.7米【已接触】）': '通道距树木距离不够',
    '全杆基杂物堆积': '杆基杂物堆积',
    '隔离开关本体锈蚀': '隔离开关锈蚀',
}


def normalize_category(name: str) -> str:
    """规范化类别名称"""
    original = name.strip()

    if original in SPECIFIC_MAPPINGS:
        return SPECIFIC_MAPPINGS[original]

    should_merge, target = should_merge_to_other(original)
    if should_merge:
        return target

    normalized = normalize_defect_name(original)

    if not normalized or len(normalized) < 2:
        return '其他缺陷'

    return normalized


# =================================================================================================


class Config:
    BASE_DIR = Path(__file__).parent.parent
    INSPECTION_DIR = BASE_DIR / "巡检报告"
    OUTPUT_DIR = BASE_DIR / "dataset"
    IMAGES_DIR = OUTPUT_DIR / "images"
    ANNOTATIONS_FILE = OUTPUT_DIR / "annotations.json"
    SEVERITY_FILE = OUTPUT_DIR / "defect_severity.json"
    LOG_FILE = OUTPUT_DIR / "build_log.txt"

    RED_DETECTION = {
        'hsv_range1': ([0, 100, 100], [10, 255, 255]),
        'hsv_range2': ([160, 100, 100], [180, 255, 255]),
        'min_area': 1000,
        'min_circularity': 0.5
    }

    COPY_IMAGES = True


class DefectDatasetBuilder:
    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = {}
        self.defect_severity = {}
        self.image_id = 0
        self.annotation_id = 0
        self.category_id = 0

        self.stats = {
            'total_images': 0,
            'success_images': 0,
            'failed_images': 0,
            'total_annotations': 0,
            'no_original': 0,
            'no_red_circle': 0,
            'parse_error': 0
        }

    def scan_inspection_folders(self) -> List[Path]:
        inspection_folders = []
        for year_dir in Config.INSPECTION_DIR.iterdir():
            if not year_dir.is_dir():
                continue
            for line_dir in year_dir.iterdir():
                if not line_dir.is_dir():
                    continue
                if (line_dir / "缺陷圈图").exists():
                    inspection_folders.append(line_dir)
        logger.info(f"发现 {len(inspection_folders)} 个巡检线路文件夹")
        return inspection_folders

    def find_original_image(self, annotated_img_path: Path, original_base: Path) -> Optional[Path]:
        filename = annotated_img_path.name
        for root, dirs, files in os.walk(original_base):
            if filename in files:
                return Path(root) / filename
        return None

    def parse_filename(self, filename: str) -> Optional[Dict]:
        try:
            name_without_ext = filename.rsplit('.', 1)[0]

            severity_keywords = ['一般缺陷', '严重缺陷', '危急缺陷']

            severity = None
            severity_pos = -1
            for keyword in severity_keywords:
                pos = name_without_ext.rfind(keyword)
                if pos != -1 and pos > severity_pos:
                    severity = keyword
                    severity_pos = pos

            if not severity:
                logger.warning(f"未找到严重程度关键词: {filename}")
                return None

            rest = name_without_ext[:severity_pos].rstrip('_-')

            # 处理连续分隔符的情况
            import re
            rest = re.sub(r'[_-]{2,}', '_', rest)

            # 尝试多种分隔符组合
            separators = ['_', '-']
            defect_desc = None
            pole_id = None
            line_name = None

            for sep1 in separators:
                parts = rest.rsplit(sep1, 1)
                if len(parts) >= 2 and parts[1].strip():
                    potential_defect = parts[1]
                    rest2 = parts[0]

                    for sep2 in separators:
                        parts2 = rest2.rsplit(sep2, 1)
                        if len(parts2) >= 2 and parts2[1].strip():
                            potential_pole = parts2[1]
                            potential_line = parts2[0]

                            # 验证：杆号应该包含数字，且不应该是纯数字
                            if any(c.isdigit() for c in potential_pole) and not potential_pole.isdigit():
                                defect_desc = potential_defect
                                pole_id = potential_pole
                                line_name = potential_line
                                break

                    if defect_desc:
                        break

            if not defect_desc or not pole_id or not line_name:
                logger.warning(f"无法解析文件名结构: {filename}")
                return None

            # 规范化 defect_desc: 移除括号内容
            defect_desc = re.sub(r'（[^）]*）', '', defect_desc)
            defect_desc = re.sub(r'\([^)]*\)', '', defect_desc)
            defect_desc = defect_desc.strip()

            # 如果 defect_desc 仍然太短或为空，使用原始值
            if len(defect_desc) < 2:
                defect_desc = rest.rsplit('_', 1)[-1] if '_' in rest else rest

            # 过滤掉明显不是缺陷类型的解析结果
            # 例如：05-1202-28支05支2 这种杆号被误识别为缺陷
            if re.match(r'^\d{2}-\d{4}', defect_desc):
                logger.warning(f"缺陷描述像是杆号片段: {defect_desc}, 文件名: {filename}")
                return None

            return {
                'line_name': line_name,
                'pole_id': pole_id,
                'defect_type': defect_desc,
                'severity': severity
            }
        except Exception as e:
            logger.warning(f"文件名解析失败: {filename}, 错误: {e}")
            return None

    def detect_red_circles(self, image_path: Path) -> List[Dict]:
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"无法读取图像: {image_path}")
                return []

            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lower1, upper1 = Config.RED_DETECTION['hsv_range1']
            lower2, upper2 = Config.RED_DETECTION['hsv_range2']

            mask1 = cv2.inRange(img_hsv, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(img_hsv, np.array(lower2), np.array(upper2))
            red_mask = cv2.bitwise_or(mask1, mask2)

            kernel = np.ones((3, 3), np.uint8)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bboxes = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < Config.RED_DETECTION['min_area']:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < Config.RED_DETECTION['min_circularity']:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                bboxes.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'area': int(area)
                })

            return bboxes

        except Exception as e:
            logger.warning(f"红色圈圈检测失败: {image_path}, 错误: {e}")
            return []

    def get_or_create_category(self, defect_type: str) -> int:
        normalized_type = normalize_category(defect_type)
        if normalized_type not in self.categories:
            self.category_id += 1
            self.categories[normalized_type] = {
                'id': self.category_id,
                'name': normalized_type,
                'supercategory': 'defect'
            }
        return self.categories[normalized_type]['id']

    def process_image(self, annotated_img_path: Path, original_img_path: Path,
                      defect_info: Dict) -> bool:
        try:
            bboxes = self.detect_red_circles(annotated_img_path)

            if not bboxes:
                self.stats['no_red_circle'] += 1
                logger.warning(f"未检测到红色圈圈: {annotated_img_path.name}")
                return False

            self.image_id += 1

            img = cv2.imread(str(original_img_path))
            if img is None:
                logger.warning(f"无法读取原图: {original_img_path}")
                return False

            height, width = img.shape[:2]

            new_filename = f"DND_{self.image_id:08d}.jpg"

            self.images.append({
                'id': self.image_id,
                'file_name': new_filename,
                'original_name': original_img_path.name,
                'width': width,
                'height': height,
                'line_name': defect_info['line_name'],
                'pole_id': defect_info['pole_id']
            })

            normalized_defect_type = normalize_category(
                defect_info['defect_type'])
            category_id = self.get_or_create_category(normalized_defect_type)

            # 存储严重程度，冲突时保留更严重的
            severity_order = {'危急缺陷': 3, '严重缺陷': 2, '一般缺陷': 1}
            if normalized_defect_type not in self.defect_severity:
                self.defect_severity[normalized_defect_type] = defect_info['severity']
            else:
                existing_severity = self.defect_severity[normalized_defect_type]
                if severity_order.get(defect_info['severity'], 0) > severity_order.get(existing_severity, 0):
                    self.defect_severity[normalized_defect_type] = defect_info['severity']

            for bbox_info in bboxes:
                self.annotation_id += 1
                self.annotations.append({
                    'id': self.annotation_id,
                    'image_id': self.image_id,
                    'category_id': category_id,
                    'bbox': bbox_info['bbox'],
                    'area': bbox_info['area'],
                    'iscrowd': 0
                })
                self.stats['total_annotations'] += 1

            if Config.COPY_IMAGES:
                dest_path = Config.IMAGES_DIR / new_filename
                if not dest_path.exists():
                    shutil.copy2(str(original_img_path), str(dest_path))

            return True

        except Exception as e:
            logger.error(f"处理图像失败: {annotated_img_path}, 错误: {e}")
            return False

    def build_dataset(self):
        logger.info("=" * 60)
        logger.info("开始构建COCO格式数据集")
        logger.info("=" * 60)

        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        Config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        inspection_folders = self.scan_inspection_folders()

        for folder in inspection_folders:
            logger.info(f"\n处理线路: {folder.name}")

            annotated_dir = folder / "缺陷圈图"
            original_base = folder / "缺陷原图"

            if not annotated_dir.exists():
                logger.warning(f"缺陷圈图目录不存在: {annotated_dir}")
                continue

            annotated_images = list(annotated_dir.glob("*.jpg")) + \
                list(annotated_dir.glob("*.JPG"))

            for annotated_img in annotated_images:
                self.stats['total_images'] += 1

                original_img = self.find_original_image(
                    annotated_img, original_base)
                if not original_img:
                    self.stats['no_original'] += 1
                    logger.warning(f"未找到原图: {annotated_img.name}")
                    continue

                defect_info = self.parse_filename(annotated_img.name)
                if not defect_info:
                    self.stats['parse_error'] += 1
                    continue

                success = self.process_image(
                    annotated_img, original_img, defect_info)

                if success:
                    self.stats['success_images'] += 1
                else:
                    self.stats['failed_images'] += 1

                if self.stats['total_images'] % 10 == 0:
                    logger.info(f"已处理 {self.stats['total_images']} 张图像")

        self.save_dataset()
        self.print_statistics()

    def save_dataset(self):
        coco_format = {
            'info': {
                'description': '配网无人机巡检缺陷检测数据集',
                'version': '1.0',
                'year': datetime.now().year,
                'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'licenses': [],
            'images': self.images,
            'annotations': self.annotations,
            'categories': list(self.categories.values())
        }

        with open(Config.ANNOTATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(coco_format, f, ensure_ascii=False, indent=2)
        logger.info(f"\n保存标注文件: {Config.ANNOTATIONS_FILE}")

        with open(Config.SEVERITY_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.defect_severity, f, ensure_ascii=False, indent=2)
        logger.info(f"保存严重程度映射: {Config.SEVERITY_FILE}")

    def print_statistics(self):
        logger.info("\n" + "=" * 60)
        logger.info("数据集构建统计")
        logger.info("=" * 60)
        logger.info(f"总图像数: {self.stats['total_images']}")
        logger.info(f"成功处理: {self.stats['success_images']}")
        logger.info(f"失败处理: {self.stats['failed_images']}")
        logger.info(f"  - 未找到原图: {self.stats['no_original']}")
        logger.info(f"  - 未检测到圈圈: {self.stats['no_red_circle']}")
        logger.info(f"  - 文件名解析失败: {self.stats['parse_error']}")
        logger.info(f"总标注数: {self.stats['total_annotations']}")
        logger.info(f"缺陷类型数: {len(self.categories)}")
        logger.info(f"输出目录: {Config.OUTPUT_DIR}")
        logger.info("=" * 60)


def main():
    builder = DefectDatasetBuilder()
    builder.build_dataset()


if __name__ == '__main__':
    main()
