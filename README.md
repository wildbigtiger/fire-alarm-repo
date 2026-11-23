# 智能火灾报警系统 · Fire & Smoke Detection

> 合肥工业大学 计算机与信息学院 电子信息工程 · 课程设计（2026 秋）
>
> 一条完整、可复现的流水线：**数据采集与标注 → 大模型零样本预标注 → YOLO 训练 → 独立测试集评估 → 开放域模型对比 → 可视化报警原型**。

---

## 1. 结果速览

| 项目 | 数值 |
|---|---|
| 检测模型 | YOLO11n（2.60 M 参数 / 6.4 GFLOPs） |
| 测试集 | 203 张 / 354 个真值框（与训练、验证完全隔离） |
| **mAP@0.5** | **0.669** |
| mAP@0.5:0.95 | 0.350 |
| Precision / Recall / F1 | 0.728 / 0.611 / 0.664 |
| AP@0.5（fire / smoke） | 0.613 / 0.726 |
| 推理速度（RTX 5060 Laptop，640 输入，batch=1） | 单帧 7.7 ms，端到端约 **130 FPS** |

**开放域对比（同一测试集、同一 IoU 协议，零样本、未微调）：**

| 模型 | mAP@0.5 | FPS | 说明 |
|---|---|---|---|
| YOLO11n（本仓库，微调后） | **0.669** | ~130 | 精度更高、可部署 |
| GroundingDINO-tiny（零样本） | 0.286 | 3.1 | 免标注、开放类别，适合做预标注工具，不适合直接部署 |

> 结论：在火灾这类**类别少、场景固定、要求实时**的任务上，微调的小模型比通用开放域大模型更合适——精度约为后者 2.3 倍，速度快约 40 倍。

---

## 2. 数据集

| 项 | 数值 |
|---|---|
| 图像总数 | 3053 张 |
| 标注框总数 | 3652 个（fire / smoke） |
| 无目标负样本 | 1289 张（夕阳、红色物体、空监控画面） |
| 场景数 | 26 个（室内/室外、白天/夜间/逆光、后厨明火等） |
| 视频素材 | 1032 段 / 158.8 分钟 |
| 划分（train / val / test） | 2479 / 371 / 203 |

**关键设计：按“场景”而不是按“图片”划分**，保证同一段视频、同一机位的帧不会同时出现在训练集和测试集里，避免数据泄漏虚高指标。

标注流程：
1. **GroundingDINO 零样本预标注**（自然语言提示词：`fire. flame. burning. blaze. smoke. smoke plume. heavy smoke.`）
2. **人工复核与修正**（X-AnyLabeling），保留原始 json 可追溯
3. 格式统一为 YOLO（归一化矩形框）
4. 数据集体检：类别分布、框越界、空标注、跨集重复

数据与模型权重体积过大，**不入版本库**。数据集统计见 [`docs/数据集说明.md`](docs/数据集说明.md)。

---

## 3. 方法

```
原始视频/图像
   │  cut_videos / extract_frames（抽帧 + 去重）
   ▼
raw/images/<场景>/*.jpg
   │  autolabel（GroundingDINO 零样本）
   ▼
ann/raw/*.json ──► 人工复核 ──► ann/labels/<场景>/*.txt
   │  split_dataset（按场景划分）
   ▼
images/{train,val,test}
   │  train_yolo（YOLO11n + 3 组优化预设）
   ▼
runs/detect/<exp>/weights/best.pt
   │  eval_yolo / eval_openvocab / segment_area
   ▼
test_metrics.md + 掩码面积占比
   │
   ▼
gui/app.py（Streamlit：实时检测 / 三级报警 / 历史回放 / 摄像头角度与 GPS）
```

**优化策略（对应 9 组对照实验）**：输入分辨率（640 → 960）、Mosaic 与多尺度增强、小目标检测层、类别不均衡处理、自采数据比例。

**一个诚实的负面结果**：为验证“自采比例 ≥30%”这条指标，单独构造了 29.7% 自采比例的对照数据集并重训（v6），mAP@0.5 反而从 0.669 降到 0.611。说明在本任务中**数据规模与场景多样性**比“自采占比”这个数字更关键。详见 [`docs/模型版本对比.md`](docs/模型版本对比.md)。

---

## 4. 目录结构

```
fire-alarm-repo/
├── configs/fire.yaml            # 数据集配置（类别、路径）
├── scripts/
│   ├── common.py                # 路径与公共工具
│   ├── cut_videos.py            # 长视频切片段
│   ├── extract_frames.py        # 抽帧 + 去重
│   ├── autolabel.py             # GroundingDINO 零样本预标注
│   ├── split_dataset.py         # 按场景划分 train/val/test
│   ├── check_labels.py          # 数据集体检
│   ├── dataset_stats.py         # 统计表
│   ├── train_yolo.py            # 训练（3 个优化预设）
│   ├── eval_yolo.py             # 测试集评估（P/R/F1/mAP/FPS）
│   ├── eval_openvocab.py        # 开放域模型零样本评估
│   ├── segment_area.py          # SAM 掩码 + 燃烧面积占比
│   ├── demo_infer.py            # 单图/视频推理
│   └── check_delivery.py        # 交付自查
├── gui/app.py                   # Streamlit 可视化原型
├── modelarts/                   # 华为云 ModelArts 训练入口
├── docs/                        # 数据集说明、模型对比、复核记录
└── requirements*.txt
```

---

## 5. 快速开始

```bash
# 1. 环境（Python 3.10+，建议 CUDA 环境）
pip install -r requirements.txt

# 2. 启动可视化原型（需自备一个训练好的权重，见下方“复现”）
streamlit run gui/app.py
# Windows 也可直接双击 启动系统.bat

# 3. 单图推理
python scripts/demo_infer.py --weights path/to/best.pt --source path/to/image.jpg
```

---

## 6. 复现

| 项 | 值 |
|---|---|
| 硬件 | NVIDIA RTX 5060 Laptop GPU（8 GB） |
| 系统 / Python | Windows 11 / 3.13.5 |
| PyTorch / Ultralytics | 2.11.0+cu128 / 8.4.182 |
| 随机种子 | 42（全部实验固定） |
| 训练输入 / batch / epoch | 640 / 16 / 100 |

```bash
python scripts/split_dataset.py --test-scenes "..." --reviewed-scenes "..."
python scripts/check_labels.py
python scripts/train_yolo.py --preset baseline
python scripts/eval_yolo.py --weights runs/detect/baseline/weights/best.pt
```

> 权重与数据未入库；如需复现请按 `docs/数据集说明.md` 准备数据，或联系作者获取测试权重。

---

## 7. 已知不足与后续方向

1. **夜间与逆光场景明显掉点**——瓶颈在成像前端而非检测头，下一步接入低光照增强（Gamma / Retinex / 学习式增强）后重测。
2. **自采数据仅占 21.1%**（题目要求 ≥30%），场景多样性不足。
3. 目前是**单帧检测**，未利用视频时序；火情蔓延判断应引入跟踪与时间维度。
4. 标注一致性只做了单人复核，未计算双人 IoU。
5. GUI 中的摄像头角度与 GPS 为**模拟数据**（题目允许），接口已预留。

---

## 8. 团队分工

| 成员 | 负责 |
|---|---|
| 石磊（组长） | 项目框架搭建、数据流程设计、模型训练与迭代优化、评估实验、可视化界面实现、文档与报告 |
| 组员 A | 数据采集（图像）、部分人工标注 |
| 组员 B | 数据采集（视频）、部分人工标注 |

---

## 9. License

MIT License，详见 [LICENSE](LICENSE)。
