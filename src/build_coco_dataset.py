#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配网无人机图像缺陷检测数据集构建脚本
功能：扫描巡检报告目录，提取缺陷图像和标注，生成COCO格式数据集
"""

import os
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
        if defect_type not in self.categories:
            self.category_id += 1
            self.categories[defect_type] = {
                'id': self.category_id,
                'name': defect_type,
                'supercategory': 'defect'
            }
        return self.categories[defect_type]['id']

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

            category_id = self.get_or_create_category(
                defect_info['defect_type'])

            self.defect_severity[defect_info['defect_type']
                                 ] = defect_info['severity']

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
