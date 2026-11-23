# -*- coding: utf-8 -*-
"""建目录 + 写类别表 + 生成 data.yaml 模板。

    python scripts/make_dirs.py

只需跑一次。重复跑不会覆盖已有的 classes.txt 和 data.yaml。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C

DATA_YAML = """\
# YOLO 数据集配置
# path 会在训练前被自动替换成绝对路径（见 scripts/train_yolo.py），这里写相对路径即可
path: datasets/fire
train: images/train
val: images/val
test: images/test

names:
  0: fire
  1: smoke
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="建立火灾数据集目录骨架")
    ap.add_argument("--classes", default="fire,smoke", help="类别名，逗号分隔，顺序即 id")
    ap.add_argument("--force", action="store_true", help="覆盖已有的 classes.txt / fire.yaml")
    args = ap.parse_args()

    C.banner("建立目录骨架")

    dirs = [
        C.RAW_IMAGES,
        C.RAW_VIDEOS,
        C.RAW_CLIPS,
        C.ANN_LABELS,
        C.ANN_RAW,
        C.ANN_MASKS,
        C.META_DIR,
        C.CONFIG_DIR,
        C.RUNS_DIR,
        *(C.IMG_DIR / s for s in C.SPLITS),
        *(C.LBL_DIR / s for s in C.SPLITS),
    ]
    for d in dirs:
        C.ensure_dir(d)
        C.info(f"  + {d.relative_to(C.PROJECT_ROOT)}")

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    cls_file = C.META_DIR / "classes.txt"
    if args.force or not cls_file.exists():
        cls_file.write_text("\n".join(classes) + "\n", encoding="utf-8")
        C.info(f"\n类别表已写入 {cls_file} -> {classes}")
        for i, name in enumerate(classes):
            C.info(f"  id {i} = {name}")
    else:
        C.info(f"\n类别表已存在，跳过：{cls_file}")

    yml = C.CONFIG_DIR / "fire.yaml"
    if args.force or not yml.exists():
        yml.write_text(DATA_YAML, encoding="utf-8")
        C.info(f"数据集配置已写入 {yml}")
    else:
        C.info(f"数据集配置已存在，跳过：{yml}")

    C.banner("下一步")
    C.info(
        "1) 采集素材，按场景建子目录（一个场景 = 一次拍摄/一个房间/一段视频）：\n"
        "     datasets/fire/raw/images/scene01_卧室白天/*.jpg\n"
        "     datasets/fire/raw/videos/scene01_卧室白天/*.mp4\n"
        "   没有火的场景单独建目录，例如 scene90_neg_夕阳，用来当负样本。\n"
        "   公开数据集来的场景用 pub_ 开头，例如 pub_dfire_sample。\n\n"
        "2) 长视频切短片段（凑题目'≥300 段 / ≥10 分钟'）：\n"
        "     python scripts/cut_videos.py --clip-len 4\n\n"
        "3) 视频抽帧： python scripts/extract_frames.py --fps 2\n\n"
        "4) 自动预标注： python scripts/autolabel.py --model IDEA-Research/grounding-dino-tiny\n"
        "   没有 GPU 想先试流程： python scripts/autolabel.py --mock\n\n"
        "5) 人工复核 ann/labels/ 下的 txt（用 X-AnyLabeling 打开，改完就地保存）\n\n"
        "6) 划分数据集： python scripts/split_dataset.py\n\n"
        "7) 体检 + 统计： python scripts/check_labels.py && python scripts/dataset_stats.py\n\n"
        "8) 训练： python scripts/train_yolo.py --epochs 100"
    )


if __name__ == "__main__":
    main()
