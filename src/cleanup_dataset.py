#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集清理脚本
功能：规范化类别名称，合并碎片化类别，修复解析错误
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def strip_parentheses(name: str) -> str:
    """移除类别名称中的括号内容"""
    # 移除中文括号内容 （...）
    name = re.sub(r'（[^）]*）', '', name)
    # 移除英文括号内容 (...)
    name = re.sub(r'\([^)]*\)', '', name)
    return name.strip()


def normalize_defect_name(name: str) -> str:
    """
    规范化缺陷类别名称
    1. 移除括号内容
    2. 修复重复词汇
    3. 修复前缀问题
    4. 修复拼写错误
    """
    original = name
    name = name.strip()

    # 移除括号内容（已决定合并到基础类别）
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

    # 修复柱上开关类 - SF6, SF, 6等变体
    if '柱上' in name and '开关' in name:
        name = re.sub(r'柱上[Ss][Ff]\d*开关', '柱上开关', name)
        name = re.sub(r'柱上(\d+)开关', r'柱上开关', name)
        # 如果是"柱上开关套管"类型，应该是"柱上开关破损"
        if '套管' in name:
            name = name.replace('柱上开关套管', '柱上开关破损')
        if '绝缘罩脱落' in name or '无绝缘罩' in name:
            name = name.replace('柱上开关', '柱上开关破损')

    # 移除尾部空格
    name = name.strip()

    return name


def is_fragment_category(name: str) -> bool:
    """判断是否为碎片化的类别（杆号片段）"""
    # 匹配类似 05-1202-28支05支2, 05-0704-01 等模式
    if re.match(r'^\d{2}-\d{4}-\d{2}', name):
        return True
    # 仙霞线#004 这种线路位置信息
    if re.match(r'.+线#\d+-.+线#\d+', name):
        return True
    return False


def should_merge_to_other(name: str) -> tuple:
    """判断是否应合并到其他缺陷"""
    # 碎片化的杆号类别 (05-xxxx, 08-xxxx)
    if re.match(r'^\d{2}-\d{4}', name):
        return True, '其他缺陷'

    # 仙霞线#004-仙霞线#005 这种带具体位置的
    if re.search(r'线#\d+.*线#\d+', name):
        return True, '通道距树木距离不够'

    # 雪峰线15-0401-106导线距树木距离不够 这种带具体杆号的
    if re.search(r'.+\d{2}-\d{4}-\d{2,4}.+距树木距离不够', name):
        return True, '导线距树木距离不够'

    # 带具体位置的通道树距木距离不够
    if '线#' in name and '距树木距离不够' in name:
        return True, '通道距树木距离不够'

    return False, None


# 定义规范化映射（用于无法自动规范化的特殊情况）
SPECIFIC_MAPPINGS = {
    # 横担锈蚀家族
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

    # 杆基杂物堆积家族
    '杆基杂物堆积（杆塔被建筑包围，违章搭盖）': '杆基杂物堆积',
    '杆基杂物堆积（杆塔被铁皮）': '杆基杂物堆积',
    '杆基杂物堆积（杆塔台变被树包围）': '杆基杂物堆积',
    '杆基杂物堆积（违章搭盖）': '杆基杂物堆积',
    '杂物堆积（巡视通道受阻）': '杆基杂物堆积',
    '杂物堆积': '杆基杂物堆积',
    '通道杂物堆积（电缆头刀闸操作位置有树木杂草）': '通道杂物堆积',

    # 绑扎线不规范家族
    '绑扎线不规范（使用裸绑扎线）': '绑扎线不规范',
    '绑扎线不规范（绑扎为裸铝线）': '绑扎线不规范',
    '绑扎线不规范（松动）': '绑扎线不规范',
    '绑扎线不规范（松弛）': '绑扎线不规范',
    '绑扎线不规范（脱落）': '绑扎线不规范',
    '导线绑扎线不规范': '绑扎线不规范',
    '绑扎带导线未绑扎': '绑扎线不规范',
    '导线未绑扎': '绑扎线不规范',
    '帮扎带导线未绑扎': '绑扎线不规范',
    '帮扎带导线未绑扎': '绑扎线不规范',  # 拼写错误

    # 绑扎带家族
    '绑扎带绑扎线不规范': '绑扎线不规范',
    '绑扎带绑扎线不规范（绑扎固定配件锈蚀严重）': '绑扎线不规范',
    '绑扎带绑扎线不规范（绑扎线无绝缘层）': '绑扎线不规范',
    '绑扎带导线未绑扎': '绑扎线不规范',

    # 通道树距木距离不够 - 树/木 typo
    '通道树距木距离不够': '通道距树木距离不够',

    # 绝缘子家族
    '绝缘子污秽（放电痕迹）': '绝缘子污秽',
    '绝缘子固定不牢固（倾斜）': '绝缘子固定不牢固',
    '绝缘子破损（损坏）': '绝缘子破损',
    '绝缘子釉表面脱落': '绝缘子破损',
    '绝缘子破损（同杆另一回路）': '绝缘子破损',
    '绝缘子倾斜': '绝缘子固定不牢固',

    # 绝缘线绝缘层破损家族
    '绝缘线绝缘层破损（裸导线）': '绝缘线绝缘层破损',
    '绝缘线绝缘层破损（现场为裸导线）': '绝缘线绝缘层破损',
    '绝缘线绝缘层破损（破皮）': '绝缘线绝缘层破损',
    '绝缘线绝缘层破损（引流线破皮）': '绝缘线绝缘层破损',
    '绝缘线绝缘层破损（老化)': '绝缘线绝缘层破损',
    '绝缘线绝缘层破损（有发热发损现象）': '绝缘线绝缘层破损',
    '绝缘线绝缘层破损（有放电痕迹）': '绝缘线绝缘层破损',
    '导线绝缘线绝缘层破损': '绝缘线绝缘层破损',

    # 通道距树木距离不够家族
    '线路大号侧通道距树木距离不够（树碰线）': '通道距树木距离不够',
    '线路大号侧通道距树木距离不够(树枝距裸导线距离0.5米)': '通道距树木距离不够',
    '线路大号侧通道距树木距离不够（树枝距绝缘导线距离0.2米）': '通道距树木距离不够',
    '线路大号侧通道距树木距离不够（树压线）': '通道距树木距离不够',
    '线路小号侧通道距树木距离不够（树碰线）': '通道距树木距离不够',
    '线路大号侧通道距树木距离不够': '通道距树木距离不够',
    '线路小号侧通道距树木距离不够': '通道距树木距离不够',
    '小号侧通道距树木距离不够': '通道距树木距离不够',
    '大号侧通道距树木距离不够': '通道距树木距离不够',

    # 导线距树木距离不够家族
    '导线距树木距离不够': '导线距树木距离不够',

    # 线路大小号侧距树木距离不够家族 - 合并到线路距树木距离不够
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

    # 距树木距离不够（带藤蔓等）
    '距树木距离不够（藤蔓爬上杆顶）': '通道距树木距离不够',
    '距树木距离不够（藤上杆）': '通道距树木距离不够',
    '通道距树木距离不够（藤上杆）': '通道距树木距离不够',
    '距树木距离不够': '通道距树木距离不够',

    # 破损家族
    '绝缘罩脱落（线夹裸露）': '绝缘罩脱落',
    '绝缘罩脱落(电缆头接线柱处无绝缘罩)': '绝缘罩脱落',
    '线夹绝缘罩脱落(电缆头接线柱处无绝缘罩)': '线夹绝缘罩脱落',
    '柱上开关绝缘罩脱落': '绝缘罩脱落',
    '线夹主件松动或脱落（柱上开关线夹断一相）': '线夹松动',

    # 套管破损家族
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

    # 避雷器家族
    '避雷器破损（锈蚀）': '避雷器破损',
    '避雷器破损': '避雷器破损',
    '避雷器老旧': '避雷器破损',
    '避雷器烧毁一相': '避雷器破损',
    '避雷器损坏一相': '避雷器破损',
    '避雷器缺失一相': '避雷器破损',
    '避雷器缺失两相': '避雷器破损',
    '避雷器无引线': '避雷器破损',

    # 接地引下线连接不良
    '接地引下线连接不良(上引线缺失)': '接地引下线连接不良',
    '接地引下线连接不良(上引线缺失，引下线断)': '接地引下线连接不良',
    '接地引下线连接不良（引上线未接入）': '接地引下线连接不良',
    '接地引下线连接不良（断）': '接地引下线连接不良',

    # 刀闸锈蚀
    '刀闸锈蚀（老旧）': '刀闸锈蚀',

    # 拉线松弛
    '拉线松弛（接线松）': '拉线松弛',
    '松弛（拉线）': '拉线松弛',

    # 鸟巢
    '鸟巢': '杆塔鸟巢',
    '杆上有鸟巢': '杆塔鸟巢',
    '杆塔本体鸟巢': '杆塔鸟巢',

    # 异物
    '杆塔异物': '杆塔异物',
    '导线上有异物': '导线异物',
    '异物（枯枝）': '杆塔异物',
    '异物（藤上杆）': '杆塔异物',
    '异物（拉线有藤包围）': '杆塔异物',
    '异物（杆上有一导线）': '杆塔异物',

    # 破损（一般）
    '破损（避雷器）': '破损',
    '破损（绝缘套管缺失）': '破损',
    '破损（柱上开关电缆头无绝缘罩无绝缘包扎：电气裸露）': '破损',
    '破损（柱上开关无绝缘护套）': '破损',
    '破损（电缆头无绝缘护套）': '破损',
    '破损（低压侧套管缺失）': '破损',
    '破损（住上开关无绝缘护套）': '破损',

    # 杆塔本体异物、岛巢
    '杆塔本体异物、岛巢（岛巢）': '杆塔异物、鸟巢',
    '杆塔本体异物、岛巢（废弃的绝缘子未拆除）': '杆塔异物、鸟巢',
    '杆塔本体异物、岛巢（岛巢已碰到导线）': '杆塔异物、鸟巢',
    '杆塔本体异物、岛巢（废旧绝缘子放在横担上）': '杆塔异物、鸟巢',
    '杆塔本体异物、岛巢（干枯的爬藤）': '杆塔异物、鸟巢',
    '杆塔本体异物、岛巢（废弃的绝缘子或避雷器未拆除）': '杆塔异物、鸟巢',

    # 绝缘护套损坏
    '绝缘护套损坏（导线为裸导线）': '绝缘护套损坏',

    # 金属氧化物避雷器
    '金属氧化物避雷器本体污秽': '避雷器破损',
    '金属氧化物避雷器破损（缺失）': '避雷器破损',
    '金属氧化物避雷器锈蚀': '避雷器破损',

    # 故障指示器
    '故障指示器安装不牢靠（老旧）': '故障指示器安装不牢靠',

    # 标识错误
    '标识错误（现场杆号与图纸不符）': '标识错误',
    '标示错误（现场杆号与图纸不符）': '标识错误',
    '标识错误（现场与系统图不符，系统09杆无用户需异动）': '标识错误',
    '标识错误（图纸为刀闸，现场是跌落式）': '标识错误',

    # 连接金具
    '连接金具球头锈蚀严重(并钩线夹有烧焦痕迹)': '连接金具锈蚀',
    '连接金具球头锈蚀严重（同杆另一回路并钩线夹有烧焦痕迹）': '连接金具锈蚀',
    '连接金具球头锈蚀严重': '连接金具锈蚀',
    '连接金具锈蚀严重': '连接金具锈蚀',

    # 安装不牢靠
    '安装不牢靠（防雷装置与穿刺体错位）': '安装不牢靠',
    '安装不牢靠（防雷装置一相缺失）': '安装不牢靠',
    '安装不牢靠（防雷装置两相缺失）': '安装不牢靠',
    '安装不牢靠（防雷装置中相缺失）': '安装不牢靠',
    '固定不牢固（倾斜）': '固定不牢固',

    # 锈蚀
    '锈蚀（起皮和严重麻点，锈蚀面积超过）': '锈蚀',
    '锈蚀（起皮和严重麻点，锈蚀面积超过12）': '锈蚀',
    '锈蚀（起皮和严重麻点，锈蚀面积超过2分之1）': '锈蚀',
    '锈蚀（刀闸老旧）': '锈蚀',
    '锈蚀（销户未拆除）': '锈蚀',
    '锈蚀（导线有灼烧的痕迹）': '锈蚀',

    # 钢绞线
    '钢绞线松弛（接线松）': '钢绞线松弛',

    # 高低压套管
    '高低压套管缺绝缘罩(变压器器身生绣严重)': '高低压套管缺绝缘罩',
    '高压接线柱无绝缘护套': '高低压套管缺绝缘罩',
    '高低压接线柱无绝缘护套': '高低压套管缺绝缘罩',

    # 柱上开关
    '柱上开关无绝缘护套': '柱上开关破损',
    '柱上SF6开关套管（缺失）': '柱上开关破损',
    '柱上SF6开关线夹绝缘罩脱落': '柱上开关破损',
    '柱上开关套管破损(缺失)': '柱上开关破损',

    # 基础沉降
    '基础沉降（电缆井盖缺失）': '基础沉降',

    # 距建筑物距离
    '距建筑物距离不够（杆被建筑物包围通道被堵）': '距建筑物距离不够',
    '导线距建筑物距离不够': '距建筑物距离不够',
    '通道距建筑物距离不够': '距建筑物距离不够',

    # 柱上开关/套管相关
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

    # 线夹相关
    '线夹严重锈蚀': '线夹锈蚀',
    '线夹松动': '线夹绝缘罩脱落',
    '线夹绝缘护罩脱落': '线夹绝缘罩脱落',
    '线夹绝缘护套脱落': '线夹绝缘罩脱落',
    '线夹护套缺失': '线夹绝缘护套缺失',

    # 横担相关
    '横担抱箍锈蚀': '横担锈蚀',
    '横担松动、主件脱落': '横担弯曲',

    # 导线相关
    '导线锈蚀': '锈蚀',
    '导线灯笼现象': '导线异物',
    '导线脱落': '导线异物',

    # 杆塔异物相关
    '杆塔上有遗留工器具': '杆塔异物',
    '杆塔本体杆塔上有遗留工器具、金具': '杆塔异物',

    # 基础相关
    '杆塔基础沉降': '基础沉降',

    # 标识相关
    '无标识或缺少标识': '标识错误',

    # 绑扎线相关
    '绑扎线不规范（绑扎线松弛)': '绑扎线不规范',

    # 绝缘罩相关
    '绝缘罩固定不牢固': '绝缘罩脱落',
    '绝缘罩污秽': '绝缘子污秽',

    # 防雷金具相关
    '防雷金具锈蚀': '避雷器破损',
    '防雷金具安装不牢靠': '安装不牢靠',

    # 拉线相关
    '拉线防护设施不满足要求': '拉线松弛',

    # 松动类
    '松动': '横担弯曲',

    # 柱上开关类
    '柱上开关绝缘护套缺失': '柱上开关破损',
    '柱上开关边相跑线距树木距离不够': '线路距树木距离不够',
    '柱上开关电缆头无绝缘罩无绝缘包扎：电气裸露）': '柱上开关破损',

    # 其他具体位置相关的距离问题
    '杆被树枝包围线路距树木距离不够': '线路距树木距离不够',
    '杆距树木距离不够': '线路距树木距离不够',
    '通道距树木距离不够(#015-#016之间距树木距离小于0.7米【已接触】）': '通道距树木距离不够',

    # 全杆基杂物
    '全杆基杂物堆积': '杆基杂物堆积',

    # 隔离开关相关
    '隔离开关本体锈蚀': '隔离开关锈蚀',
}


def normalize_category(name: str) -> str:
    """规范化类别名称"""
    original = name.strip()

    # 1. 先检查精确映射
    if original in SPECIFIC_MAPPINGS:
        return SPECIFIC_MAPPINGS[original]

    # 2. 检查是否为碎片类别
    should_merge, target = should_merge_to_other(original)
    if should_merge:
        return target

    # 3. 应用通用规范化
    normalized = normalize_defect_name(original)

    # 4. 如果规范化后为空或太短，返回其他缺陷
    if not normalized or len(normalized) < 2:
        return '其他缺陷'

    return normalized


def cleanup_dataset():
    """清理数据集"""
    dataset_path = Path('dataset/annotations.json')
    severity_path = Path('dataset/defect_severity.json')
    output_path = dataset_path

    print("Loading dataset...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 统计信息
    stats = {
        'total_annotations': len(data['annotations']),
        'original_categories': len(data['categories']),
        'normalized_names': defaultdict(int),
        'fragment_categories': 0,
        'empty_categories': 0,
    }

    # 创建 ID -> 原始category 映射
    id_to_category = {cat['id']: cat for cat in data['categories']}

    # 规范化每个 annotation 的 category_id
    old_to_new_cat_id = {}  # old_id -> new_canonical_name
    new_categories = {}  # canonical_name -> {'id': new_id, 'count': 0}

    for ann in data['annotations']:
        old_cat_id = ann['category_id']
        old_cat_name = id_to_category[old_cat_id]['name']

        # 规范化类别名称
        new_cat_name = normalize_category(old_cat_name)

        # 统计
        stats['normalized_names'][new_cat_name] += 1

        if old_cat_id not in old_to_new_cat_id:
            old_to_new_cat_id[old_cat_id] = new_cat_name

        if new_cat_name == '其他缺陷':
            stats['fragment_categories'] += 1

        if not new_cat_name:
            stats['empty_categories'] += 1

    # 创建新的紧凑类别 ID
    unique_categories = sorted(set(old_to_new_cat_id.values()))
    new_cat_id_map = {name: idx + 1 for idx, name in enumerate(unique_categories)}

    # 更新 annotations 的 category_id
    for ann in data['annotations']:
        old_cat_id = ann['category_id']
        old_cat_name = id_to_category[old_cat_id]['name']
        new_cat_name = old_to_new_cat_id[old_cat_id]
        ann['category_id'] = new_cat_id_map[new_cat_name]

    # 构建新的 categories 列表
    new_categories_list = []
    for cat_name in unique_categories:
        new_id = new_cat_id_map[cat_name]
        count = stats['normalized_names'][cat_name]
        new_categories_list.append({
            'id': new_id,
            'name': cat_name,
            'supercategory': 'defect'
        })

    data['categories'] = new_categories_list

    # 更新 defect_severity.json
    print("Loading severity mapping...")
    with open(severity_path, 'r', encoding='utf-8') as f:
        severity_data = json.load(f)

    new_severity = {}
    for old_name, severity in severity_data.items():
        new_name = normalize_category(old_name)
        # 如果多个旧名称映射到同一个新名称，保留一个
        if new_name not in new_severity:
            new_severity[new_name] = severity
        elif severity != new_severity[new_name]:
            # 严重程度冲突，保留更严重的
            severity_order = {'危急缺陷': 3, '严重缺陷': 2, '一般缺陷': 1}
            if severity_order.get(severity, 0) > severity_order.get(new_severity[new_name], 0):
                new_severity[new_name] = severity

    # 保存清理后的数据集
    print(f"Saving cleaned dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 保存新的 severity mapping
    print(f"Saving cleaned severity mapping to {severity_path}...")
    with open(severity_path, 'w', encoding='utf-8') as f:
        json.dump(new_severity, f, ensure_ascii=False, indent=2)

    # 打印统计信息
    print("\n" + "=" * 60)
    print("Dataset Cleanup Summary")
    print("=" * 60)
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"Original categories: {stats['original_categories']}")
    print(f"Cleaned categories: {len(unique_categories)}")
    print(f"Fragment categories merged: {stats['fragment_categories']}")
    print()

    print("Top 20 categories by count:")
    print("-" * 50)
    sorted_cats = sorted(stats['normalized_names'].items(),
                         key=lambda x: x[1], reverse=True)
    for name, count in sorted_cats[:20]:
        print(f"  {count:5d} | {name}")

    print("\n" + "=" * 60)
    print("Cleanup complete!")


if __name__ == '__main__':
    cleanup_dataset()
