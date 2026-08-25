# 实验记录

**验收标准**: yolo11n 在 `dataset/split_reduced.json` 冻结 val(733 张完整原图)上 mAP50-95 ≥ 0.40

## 数据定义(冻结)

- `dataset/annotations_reduced.json`: 16 类(5 组合并 + 10 独立 + 其他缺陷), 6218 标注, 3665 图
- `dataset/split_reduced.json`: train 2932 / val 733, seed 42, 分层(每类 val ≥35 实例)
- `dataset/annotations_tiles.json`: 训练切片, tile=1280 stride=1024, 空片保留 30%
- 环境: `~/miniconda/envs/yolo` (ultralytics 8.3.203, torch 2.0.1+cu117)

## Round 1(3 卡并行)— 已终止(发现数据污染)

| GPU | 实验 | 配置 | best mAP50-95 | best mAP50 | 备注 |
|---|---|---|---|---|---|
| 0 | r1_yolo11n_tiled | yolo11n, tiled, imgsz=1280, batch=16 | 0.049@ep45 | 0.069 | patience 早停 ep95 |
| 1 | r1_yolo11m_tiled | yolo11m, tiled, batch=8 | 0.041@ep23 | 0.049 | 发现污染后终止 |
| 2 | r1_yolo11n_full | yolo11n, 未切片, batch=16 | 0.087@ep84 | 0.136 | 发现污染后终止 ep98 |

### Round 1 关键发现: 红圈污染

- per-class AP: `其他缺陷` AP50-95=0.79(异常), 其余 15 类全部 ≤0.11
- 全数据集扫描: **2187/3665 (59.7%) 图像残留红圈**(`缺陷圈图`混入,红像素>0.05%)
  - `其他缺陷` 169/171 污染 → 模型学到"红圈→类别"捷径, 虚假高 AP
  - 污染比例越低的类 AP 越真实但越低
- 结论: R1 所有绝对数值无效; 数据形态(切片与否)不是主要矛盾, **标注/图像质量才是**

### 处置

1. `src/clean_red_circles.py`: HSV 红像素 mask + dilate + Telea inpainting, 污染原图备份至 `dataset/images_contaminated_backup/`
2. 重建切片 + 删 YOLO labels 缓存
3. Round 2 在干净数据上重训(同 R1 三组配置)

## Round 2(部分清洗数据 — 作废)

| GPU | 实验 | best mAP50-95 | best mAP50 | 备注 |
|---|---|---|---|---|
| 0 | r2_yolo11n_tiled | 0.049@ep82 | 0.074 | 早停 ep132 |
| 1 | r2_yolo11m_tiled | 0.057@ep39 | - | 发现清洗不彻底后终止 |
| 2 | r2_yolo11n_full | 0.087@ep94 | 0.133 | 与 R1 持平 → 暴露清洗不彻底 |

### Round 2 关键发现: 清洗 mask 缺陷

- R2 与 R1 数字几乎相同 → 怀疑清洗未生效; 抽查证实**大量手绘圈残留**
- 根因: 轮廓版 mask 用 `RETR_EXTERNAL` + `contourArea`, 细环外轮廓=实心圆盘 → 形状判据失效
- 修复: `connectedComponentsWithStats` 真实像素面积 + extent<0.4 判据; 只对圈所在 ROI 做 inpaint(提速 2.4x); 24 进程并行
- 修复后全量验证: 红像素 p50=0.06%(残留为实心红物体, 合理)
- 另注: `其他缺陷` 清洗前后 AP 均 0.80 → 其"可学性"来自残留红圈捷径, 非真实能力

## Round 3(真·干净数据)— 决定性结果

| GPU | 实验 | best mAP50-95 | best mAP50 | 备注 |
|---|---|---|---|---|
| 0 | r3_yolo11n_tiled | 0.011@ep90 | 0.034 | 早停 ep140 |
| 1 | r3_yolo11m_tiled | 0.004@ep39 | 0.010 | 终止 |
| 2 | r3_yolo11n_full | 0.040@ep148 | 0.088 | 跑满 150ep, 仍在缓慢上升 |

### 结论(已与用户对齐)

- 清洗后 mAP 不升反降 → **R1/R2 的"学习信号"几乎全部来自红圈残留捷径**; `其他缺陷` AP 0.80→0.10 证实
- 真实基线 yolo11n = **0.04**; 瓶颈=标注形态(松散 ROI 圈 + 细微属性缺陷), 非模型容量(yolo11m 同样起不来)
- per-class 真实可学性: 杆塔异物鸟巢 0.14 > 避雷器破损 0.10 > 横担锈蚀 0.09, 其余 12 类 <0.05
- **用户决策: 走紧框重标注 pipeline** — YOLO-World (yolov8x-worldv2) 在 ROI 内生成设备紧框, 重建 train+val, 验收口径随之更新
- 依赖: ultralytics/CLIP(用户批准安装)

## Round 4(紧框重标注)— 否定结论

| GPU | 实验 | best mAP50-95 | 备注 |
|---|---|---|---|
| 0 | r4_yolo11n_tiled | 0.006@ep63 | 早停 ep113 |
| 1 | r4_yolo11m_tiled | 0.004@ep12 | 终止 |
| 2 | r4_yolo11n_full | 0.019@ep104 | 终止 |

- 紧框数据表现**更差**: YOLO-World 在低 conf(0.003-0.07)下的紧框噪声大(错框/半框), 且紧框(设备级)与回退 ROI(区域级)语义混杂
- 附加实验: r3_yolo11n_full best 在 imgsz=2560 推理 → mAP 0.013(**尺度锁定**: 1280 压缩训练的模型只能 1280 推理; 切片训练需 SAHI 式切片推理验收, 未实现)
- 四轮总结论: 瓶颈 = 缺陷视觉显著性低 + 标注噪声, 清洗/切片/紧框/容量 四个杠杆全部否定

历史参照: yolov8m 51类 imgsz=1280 39ep(未完成): mAP50-95=0.061, mAP50=0.108

## Round 6(教师补全 + copy-paste)— 否定结论

**数据**: r5_yolo11m 教师切片推理补全 train 标注 +9471(树木 5004 条经植被过滤后 3319)→ copy-paste 5 弱类 +1407 图 → train 4339 张 / 15411 标注; val 不变(733 纯人工 GT)

| GPU | 实验 | best mAP50 | 备注 |
|---|---|---|---|
| 0 | r6_yolo11n_full | 0.0488@ep117 (TTA 0.055) | 早停 ep138; **远低于 r3 的 0.088** |
| 1 | r6_yolo11m_full | 0.0576@ep93 | 早停 ep143; 容量救不了噪声标签 |
| 2 | r6_yolo11n_tiled | 0.0401@ep64 | 趋势明确后终止, 释放 GPU |

### 结论

- 弱监督补标(教师自训练)+ copy-paste **不升反降**: 教师噪声标签(即使过滤)稀释了人工标注质量
- 与 R4(紧框重标注失败)结论一致: **任何引入噪声标签的弱监督路线在该数据集上都是负收益**
- 瓶颈依然是: 缺陷视觉显著性低 + 人工标注本身松散, 而非标注数量不足

## 终止判定(plateau 规则触发)

- R5(长训/高分辨率): 0.0796, 未超最优 → strike 1
- R6(教师补全+copy-paste): 0.0576, 未超最优 → strike 2
- 连续 2 轮无 >1pt 提升, 按 PLAN.md 停止迭代, 交付最优结果 + 分析报告
- **最终交付**: r3_yolo11n_full, 冻结 val mAP50=0.088 (TTA 0.096), 详见 experiments/final_report.md

## Round 7(SAM 框提示精修)— 用户批准的无监督路线实验

**动机**: R6 后用户提出无监督方向; 采用 SAM(sam2_l)类别无关分割精修松散 ROI, 避免 R4(YOLO-World 类别条件检测)的语义噪声

**关键调试**: ultralytics SAM 框提示的 mask 分数普遍 < 默认 conf=0.25 → 返回空; 需显式 conf=0.01(SAM1/SAM2 同)

**精修规则**(保守, 不达标回退原 ROI): mask ≥5000 像素, extent ≥0.2(排除电线细长误分割, 抽查验证有效), 紧框限制在 ROI 外扩 30% 内

**结果**: 6218 标注中 3775 精修(61%), 2443 回退; train/val 同步精修保持语义一致
- 精修率: 绝缘子污秽 81%, 绑扎线 76%, 鸟巢 75%, 横担锈蚀仅 27%, 防护设施 34%
- 抽查: 紧框质量良好(绝缘子串/线夹级), 误分割基本被 extent 规则拦截

**训练**: r7_yolo11n_sam(GPU0) + r7_yolo11m_sam(GPU1), 150ep, imgsz=1280
**注意**: val 同步精修 → 验收口径与 R3-R6(松散 ROI val)不直接可比, 用于验证"紧框语义是否更可学"

### R7 结果 — 否定结论

| 实验 | val 口径 | best mAP50 |
|---|---|---|
| r7_yolo11n_sam(紧框训练) | 紧框 | 0.0627@ep97 (跑满 150ep) |
| r3_yolo11n_full(松散框训练, 交叉对照) | 紧框 | 0.0681 |
| (参照) r3_yolo11n_full | 松散框 | 0.0879 |

- **紧框训练在自己的口径下仍低于松散框训练** → 框几何形态不是瓶颈, SAM 精修无收益
- r7_yolo11m_sam 在趋势明确后终止(容量从未改变过结论)
- **三次独立验证一致**(R4 紧框重标注 / R6 弱监督补标 / R7 SAM 精修): 任何"改标注"路线均为负或零收益
- 最终结论收紧: 瓶颈 = 缺陷视觉显著性低 + 类间语义混淆(属性级缺陷), 唯一出路仍是高质量人工标注(紧框+补全漏标+可能需重定义缺陷类别体系)

## R8-prep(InsPLAD 单独训练+验证)— 决定性对照

**数据**: InsPLAD-det (Mendeley 5n3fjgvfyz, CC BY-NC 3.0), 7981 train / 2626 val, 18 类电力部件, 1920x1080, 抽检标注为高质量紧框
**转换**: `src/insplad_to_yolo.py` → `dataset/yolo_insplad/`

| 实验 | best mAP50 | mAP50-95 |
|---|---|---|
| r8_insplad_yolo11n2 (100ep 跑满) | **0.916@ep97** | 0.751 |

### 结论

1. **同管线同模型: InsPLAD 0.916 vs VibeDND 0.088 (10 倍差距)** — 瓶颈在数据的最终铁证, 环境/代码/训练流程全部排除
2. 跨域测试: 整图广角(4000x3000)conf=0.05 零检出(尺度/视角差异是绝对的); **特写裁剪下 glass insulator conf=0.81 直接认出中国配网绝缘子** → 部件级特征跨域有效, 预训练有戏, 但必须全量微调适配尺度
3. 注意 InsPLAD 图像风格: 大量部件特写+浅景深, 与我方广角场景差异大; 预训练价值取决于 fine-tune 的尺度重适配

## 2026-08-12 预训练集 v1 合并(里程碑 2)

- 下载: ICARUS(Zenodo 7781388, 12,334 张电杆) + Tomaszewski(Kaggle 镜像, 1,600) + CPLID(GitHub, 848, VOC XML) + ID-2024(GDrive, 1,631); UPID 官方 GDrive 已失效; EPRI 仅得标签 CSV(29,620 条 9 类多边形), 图像 540GB 需填表单, 用户决定放弃
- 意外收获: InsPLAD_Dataset.zip 内含 supervised_fault_classification(图像级 5 部件状态分类, 含 bird-nest/rust/missing-cap, ~5,800 张), 已解压至 external/insplad/fault/
- `src/build_pretrain_det.py` 合并 5 源 → `dataset/yolo_pretrain_det/`: 27,022 张(train 21,188/val 5,834), 52,078 实例, 21 类(pole/insulator/污秽闪络/破损/缺帽 + InsPLAD 金具 16 类); 校验: 图像标签对齐, 0 越界
- 注意: ID-2024 实际只有 flashover/broken 两类实例(yaml 里 insulator/missing-cap 无实例); 缺帽类仅图像级料
- 16 类覆盖: 绝缘子污秽✅/破损✅(检测框), 缺帽/鸟巢/横担锈蚀⚠️(仅图像级), 其余 11 类❌

## 2026-08-12 外部数据路线重大收缩(用户决策)

- 用户裁定: **输电侧数据不可用, 只保留配网数据(缺陷或部件标注皆可)**
- 归属核实: InsPLAD=巴西输电 69-230kV, Tomaszewski=实验架摆拍, CPLID=中国输电(且缺陷图为合成), ID-2024=中国输电(铁塔长串), 全部删除; ICARUS=中压配电 ✓ 保留; EPRI=配电 ✓ 但图像需表单(540GB)
- 硬结论: **公开世界不存在"配网+缺陷标注"数据集**(Energies 2026 综述结论的实测验证); 16 类缺陷外部覆盖归零
- 已删: external/{insplad,upid,id2024} + yolo_insplad + yolo_pretrain_det (~27G); insplad_to_yolo.py / build_pretrain_det.py 归档; r8 权重保留(不可复现的历史参照)
- 剩余外部资产: ICARUS 12,334 张(杆顶 T 结构, 单类) + EPRI 标签 CSV(29,620 条, 无图)
- OPDL(15kV 配电绝缘子, 本符合标准)官网已死无镜像, 不可得

## 2026-08-13 EPRI 图像到位 + 部件检测数据集 v2

- EPRI 表单申请成功, 22 zip 链接清单存入路线文档; Azure 下载需 --cacert /etc/ssl/certs/ca-certificates.crt(默认 CA 包缺 Microsoft TLS G2)
- 用户定: 只下 3 个 zip → Circuit5(428)+Circuit7(717)+Circuit10(575) = 1,720 张 5184×3888, 标签命中 99.4%
- `src/build_component_dataset.py` → `dataset/yolo_component/`: 14,091 张(train 11,146/val 2,945), 23,346 实例, 6 类(insulator/pole/crossarm/cutout/transformer/pole_top); EPRI 折线类不转框, 按线路地理隔离划分(circuit10 整线做 val); 校验通过(0 越界)

## 2026-08-13 ICARUS 删除(用户决策)

- 用户裁定: ICARUS 杆顶(俯视 T 结构)也用不了 → 删除 external/icarus(21G); 至此外部料只剩 EPRI 3 线路
- `dataset/yolo_component/` 重建为 EPRI-only: 1,709 张(train 1,134 / val 575, circuit10 整线 val), 9,999 实例, 5 类(insulator 4,573 / pole 1,737 / crossarm 1,608 / cutout 1,144 / transformer 937)

## 2026-08-13 EPRI 扩量至 8 线路(用户追加 5 zip)

- 追加下载 Circuit3/4/6/8/13B(~13.3G), 共 8 线路 5,678 张图像
- yolo_component 重建: **5,678 张(train 4,352 / val 1,326, val=circuit10+13B 两线地理隔离), 25,481 实例, 5 类**(insulator 11,281 / pole 5,758 / crossarm 3,443 / cutout 2,689 / transformer 2,310); 校验通过

## 2026-08-13 里程碑 3: EPRI→中国配网迁移验证通过 + DND 全量伪标注

- `runs/detect/r9_epri_component/`(yolo11n, imgsz=1280, 20 轮用户叫停): **val mAP50=0.8675**(circuit10+13B 整线地理隔离)
- 跨域实测(DND 斜拍特写): 绝缘子迁移极好(白/棕针式+盘式 0.5-0.8 置信), 电杆良好, 横担较弱(美木横担 vs 中角钢), 杆基特写/大俯视组件失败
- `src/pseudo_label_dnd.py` 全量伪标注 3,665 张: **2,341 张(64%)有 ≥0.5 检出** → `dataset/yolo_dnd_pseudo/`; 实例: insulator 5,895 / pole 692 / crossarm 642 / cutout 153 / transformer 34; 全部检出(≥0.25 含置信度)存 `experiments/pseudo_labels_raw/`, 零检出 1,324 张清单 `experiments/pseudo_zero_det.txt`

## 2026-08-13 VLM(Qwen2.5-VL-7B)部件级标注实验 — 否定结论

- **实验一 全图部件检测**(`src/vlm_label_experiment.py`, 16 张): 彻底失败。VLM 每图 0-3 框(EPRI 检测器 1-12 框), 与伪标注 IoU≥0.3 匹配 **0 对**; 框松散成群、建筑误检。1024px 降采样 + 7B grounding 能力对密集小目标无解。渲染: `experiments/vlm_exp/render/`
- **实验二 部件裁剪图状态分类**(`src/vlm_crop_status_experiment.py`, 10 个高置信裁剪): **严重缺陷偏置 — 9/10 判破损/锈蚀, 抽看 5 张全部为正常部件**(绝缘子完好无裂纹, 电杆组件无缺失); 即 VLM 在正常部件上幻觉裂纹。裁剪: `experiments/vlm_exp2/crops/`
- **结论: 7B VLM 零样本既不能替代检测器(Stage-1), 也不能替代状态分类器(Stage-2)**; 缺陷状态监督仍无外部来源, 只能靠合成/人工标注
- 环境备忘: transformers 4.49.0 + torch 2.0.1 eager 注意力, 图像须预缩 ≤1024px(fp32 softmax 峰值), HF_ENDPOINT=https://hf-mirror.com

## 2026-08-14 Claude 视觉标注试验(20 张, 子代理)

- 动机: VLM 实验证伪后, 测试 Claude 视觉能否替代人工做缺陷标注(验收协议方案 A 的低成本变体)
- 样本: 14 部件丰富 + 4 零检出 + 2 随机; 流程: 整图判场景型 → EPRI 伪标注框引导裁剪放大判部件状态型; 保守原则(宁漏勿错, 允许 0 标注)
- 结果: 13/20 张有标注共 16 条(high 4 / medium 6 / low 6): 树障 6 / 杆基杂物 3 / 横担锈蚀 3 / 鸟巢 2 / 其他 1 / 绑扎线 1; 7 张 0 标注
- 我抽查 high 标注 2 张(1387 鸟巢 / 689 树障)全部正确, 框位准确
- **结论: 场景异物型缺陷(鸟巢/树障/杆基杂物)Claude 标注可用; 部件状态型细微缺陷(裂纹/污秽/锈蚀分级)证据力不足** — 红丹防锈漆 vs 真锈无法区分, 污秽缩略后不可辨, 亚厘米裂纹本批 0 检出(不代表 0 漏检)
- 产物: experiments/claude_label_20/{annotations.jsonl, render/}; 分支 B(场景异物型 5 类)可用此路线低成本造训练料

## 2026-08-14 Claude 部件级检测标注试验(7 类 × 20 张, 子代理)

- 用户裁定: 砍发丝级 2 类(绑扎线/绝缘护套), 标 5 部件(绝缘子/横担/套管/避雷器/绝缘罩, 正常缺陷全标) + 2 场景(鸟巢/杆基杂物); 新抽 20 张(10 杆顶丰富 + 4 设备杆 + 4 零检出 + 2 随机)
- 结果: **286 实例**(high 218 / medium 63 / low 5): 绝缘子 185 / 横担 67 / 套管 14 / 绝缘罩 13 / 避雷器 4(全 medium) / 鸟巢 2 / 杆基杂物 1
- 我抽查 3 张(676 变压器台架 / 1729 村道杆 / 1659 林区杆顶): 绝缘子/横担/套管/绝缘罩框位准确, 鸟巢定位正确; 标注纪律执行好(远景不标记 notes, 耐张串一串一框, 驱鸟风车/路灯杆明确排除)
- 弱项: 避雷器 vs 红色复合针式绝缘子/电缆终端头难区分(4 例全 medium); 杆基杂物边界主观
- **结论: Claude 部件级检测标注可用(尤其绝缘子/横担), 可作 Stage-1 中国配网部件检测器的种子真标; 避雷器类需人工复核 medium 实例**
- 产物: experiments/claude_label_det20/{annotations.jsonl, render/}
