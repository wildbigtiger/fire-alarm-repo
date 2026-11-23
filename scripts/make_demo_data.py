# -*- coding: utf-8 -*-
"""生成一批假的"火灾场景"图片，用来在没数据的时候把整条流水线跑通。

它写到一个独立的目录 datasets/demo/，不会污染你的真实数据集。

    python scripts/make_demo_data.py
    # 之后所有命令加上数据根目录，例如（Windows PowerShell）：
    $env:FIRE_DATA_ROOT="datasets/demo"
    python scripts/autolabel.py --mock
    python scripts/split_dataset.py
    python scripts/check_labels.py
    python scripts/dataset_stats.py

    # （Linux / ModelArts）
    export FIRE_DATA_ROOT=datasets/demo

看清楚：这些图是画出来的色块，**只能验证流程，绝不能用来训练正式模型**。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_ROOT = PROJECT_ROOT / "datasets" / "demo"
os.environ.setdefault("FIRE_DATA_ROOT", str(DEMO_ROOT))

import common as C  # noqa: E402

SCENES = [
    ("demo_scene01_室内白天", False),
    ("demo_scene02_室外夜晚", False),
    ("demo_scene03_neg_夕阳", True),   # 负样本场景：没有火
    ("pub_demo04_公开集样例", False),  # pub_ 前缀 = 公开数据集来源
]


def draw(img_path: Path, has_fire: bool, idx: int) -> None:
    import random

    from PIL import Image, ImageDraw

    W, H = 480, 360
    img = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(img)

    # 背景渐变：室内偏灰、夜晚偏暗、夕阳偏橙
    for y in range(H):
        t = y / H
        if "neg" in img_path.parent.name:
            col = (int(120 + 80 * t), int(60 + 40 * t), int(30 + 20 * t))
        elif "夜晚" in img_path.parent.name:
            col = (int(12 + 18 * t), int(12 + 20 * t), int(28 + 26 * t))
        else:
            col = (int(70 + 60 * t), int(72 + 62 * t), int(78 + 66 * t))
        d.line([(0, y), (W, y)], fill=col)

    if has_fire:
        rnd = random.Random(idx * 977 + 13)
        cx = rnd.randint(90, W - 90)
        cy = rnd.randint(H // 2, H - 60)
        r = rnd.randint(35, 70)
        for k in range(12, 0, -1):
            rr = int(r * k / 12)
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                      fill=(255, int(120 + 10 * k), 30))
        # 烟
        sx = rnd.randint(60, W - 60)
        for k in range(6, 0, -1):
            rr = 30 * k
            d.ellipse([sx - rr, 40 - rr // 2, sx + rr, 40 + rr // 2],
                      fill=(150 + 10 * k, 150 + 10 * k, 150 + 10 * k))

    img = img.resize((W, H))
    C.ensure_dir(img_path.parent)
    img.save(img_path, quality=88)


def main() -> None:
    import random

    C.banner(f"生成演示数据 -> {C.DATASET_ROOT}")
    for d in (C.RAW_IMAGES, C.ANN_LABELS, C.ANN_RAW, C.META_DIR,
              *(C.IMG_DIR / s for s in C.SPLITS), *(C.LBL_DIR / s for s in C.SPLITS)):
        C.ensure_dir(d)
    (C.META_DIR / "classes.txt").write_text("fire\nsmoke\n", encoding="utf-8")

    rnd = random.Random(7)
    n = 0
    for scene, is_neg in SCENES:
        for i in range(6):
            p = C.RAW_IMAGES / scene / f"img_{i:03d}.jpg"
            draw(p, not is_neg and rnd.random() > 0.25, i)
            n += 1
    C.info(f"已生成 {n} 张图片，{len(SCENES)} 个场景")

    C.banner("接下来这样跑")
    env = "export FIRE_DATA_ROOT=datasets/demo" if os.name != "nt" else '$env:FIRE_DATA_ROOT="datasets/demo"'
    C.info(
        f"{env}\n"
        "python scripts/autolabel.py --mock\n"
        "python scripts/split_dataset.py "
        "--test-scenes \"demo_scene03_neg_夕阳,pub_demo04_公开集样例\" "
        "--reviewed-scenes \"demo_scene03_neg_夕阳,pub_demo04_公开集样例\"\n"
        "python scripts/check_labels.py\n"
        "python scripts/dataset_stats.py\n\n"
        "想回到真实数据集，清掉这个环境变量即可：\n"
        + ("unset FIRE_DATA_ROOT" if os.name != "nt" else 'Remove-Item Env:FIRE_DATA_ROOT')
    )


if __name__ == "__main__":
    main()
