# -*- coding: utf-8 -*-
"""把一个外来的 YOLO 格式公开数据集，导入成我们项目的结构。

    # 先小批量试（推荐：确认类别对调对不对）
    python scripts/import_public_dataset.py --src "D:\data\train\train" --limit 60
    python scripts/import_public_dataset.py --src "..." --limit 60 --no-swap

    # 正式导入（按自采数量决定，别贪多）
    python scripts/import_public_dataset.py --src "..." --limit 1200 --max-negatives 300

它会做四件事：
  1. 跳过 macOS 垃圾文件（._ 开头）—— 不跳过的话一半文件都是垃圾
  2. 类别对调：公开集常是 0=烟雾 1=火焰，我们是 0=火焰 1=烟雾（--swap 默认开）
  3. 按编号分桶成多个"伪场景"，避免公开集全落进同一个 split
  4. 图片进 raw/images/pub_xxx/，标签进 ann/labels/pub_xxx/

注意：题目要求自行采集不少于总量 30%。导入多少公开图，取决于你自采了多少：
  允许的公开图 ≈ 自采图 × 2.33
  例：自采 500 张 -> 公开最多约 1160 张。导入前先算一下。
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C


def valid_files(d: Path, exts: tuple[str, ...]) -> list[Path]:
    out = []
    for p in d.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith("._") or p.name.startswith("."):
            continue  # macOS 元数据/隐藏文件
        if p.suffix.lower() in exts:
            out.append(p)
    return sorted(out, key=lambda p: p.name)


def main() -> None:
    ap = argparse.ArgumentParser(description="导入公开 YOLO 数据集")
    ap.add_argument("--src", required=True, help="数据集根目录（里面有 images/ 和 labels/）")
    ap.add_argument("--scene-prefix", default="pub_firedetect", help="场景名前缀，必须以 pub_ 开头")
    ap.add_argument("--limit", type=int, default=0, help="最多导入多少张（0=全部，慎用）")
    ap.add_argument("--max-negatives", type=int, default=-1, help="最多导入多少张无框的（-1=不限）")
    ap.add_argument("--buckets", type=int, default=4, help="拆成几个伪场景（0=不拆）")
    ap.add_argument("--swap", dest="swap", action="store_true", default=True, help="类别 0/1 对调（默认开）")
    ap.add_argument("--no-swap", dest="swap", action="store_false")
    ap.add_argument("--mode", choices=["copy", "hardlink"], default="hardlink")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    img_dir, lbl_dir = src / "images", src / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        C.die(f"{src} 下没有 images/ 和 labels/")

    images = valid_files(img_dir, tuple(C.IMAGE_EXTS))
    if not images:
        C.die("没找到图片")

    # 读标签，区分有无目标
    with_box, no_box = [], []
    for ip in images:
        lp = lbl_dir / f"{ip.stem}.txt"
        if not lp.exists():
            continue
        rows = C.read_label(lp)
        (with_box if rows else no_box).append((ip, lp, rows))

    C.banner("源数据集")
    C.info(f"有效图片 {len(images)} 张（已跳过 ._ 垃圾文件）")
    C.info(f"有目标 {len(with_box)} 张，无目标（负样本） {len(no_box)} 张")
    note = "对调（0=烟雾->火焰，1=火焰->烟雾）" if args.swap else "不对调"
    C.info(f"类别处理：{note}")

    # 抽样
    rng = random.Random(args.seed)
    pos = list(with_box)
    neg = list(no_box)
    rng.shuffle(pos)
    rng.shuffle(neg)

    if args.max_negatives >= 0:
        neg = neg[: args.max_negatives]
    if args.limit and len(pos) + len(neg) > args.limit:
        # 负样本最多占一半，剩下的名额全给有目标的
        n_neg_take = min(len(neg), args.limit // 2)
        pos = pos[: args.limit - n_neg_take]
        neg = neg[:n_neg_take]
    sel = pos + neg
    sel.sort(key=lambda x: x[0].name)

    classes = C.load_classes()
    stats: dict[str, dict] = {}
    n_img = n_box = n_neg = 0
    n_skip = 0

    for ip, lp, rows in sel:
        bucket = ""
        if args.buckets > 0:
            import zlib

            bucket = f"_{zlib.crc32(ip.stem.encode()) % args.buckets}"
        scene = f"{args.scene_prefix}{bucket}"
        dst_img = C.RAW_IMAGES / scene / ip.name
        dst_lbl = C.ANN_LABELS / scene / f"{ip.stem}.txt"
        if dst_img.exists():
            n_skip += 1
            continue
        C.ensure_dir(dst_img.parent)
        C.ensure_dir(dst_lbl.parent)
        if args.mode == "hardlink":
            try:
                dst_img.hardlink_to(ip)
            except Exception:
                shutil.copy2(ip, dst_img)
        else:
            shutil.copy2(ip, dst_img)

        new_rows = []
        for cls, cx, cy, w, h in rows:
            if cls >= len(classes):
                continue
            new_cls = (1 - cls) if (args.swap and len(classes) == 2) else cls
            new_rows.append((new_cls, cx, cy, w, h))
        C.write_label(dst_lbl, new_rows)

        s = stats.setdefault(scene, {"images": 0, "boxes": 0, "neg": 0, "per_class": {c: 0 for c in classes}})
        s["images"] += 1
        s["boxes"] += len(new_rows)
        if not new_rows:
            s["neg"] += 1
        for r in new_rows:
            s["per_class"][classes[r[0]]] += 1
        n_img += 1
        n_box += len(new_rows)
        if not new_rows:
            n_neg += 1

    C.save_json(C.META_DIR / "import_log.json", {
        "time": C.now_str(), "src": str(src), "swap": args.swap, "buckets": args.buckets,
        "imported": n_img, "boxes": n_box, "negatives": n_neg, "scenes": stats,
    })

    C.banner("导入完成")
    for scene in sorted(stats):
        s = stats[scene]
        C.info(f"  {scene:24} {s['images']:>5} 张  {s['boxes']:>5} 框  负样本 {s['neg']:>4}  "
               + " ".join(f"{c}={s['per_class'][c]}" for c in classes))
    C.info(f"\n共导入 {n_img} 张（{n_box} 个框，{n_neg} 张负样本），跳过重复 {n_skip} 张")

    # 自采占比提醒
    all_imgs = C.iter_images(C.RAW_IMAGES)
    pub = sum(1 for p in all_imgs if C.scene_of(p).startswith("pub_"))
    if all_imgs:
        C.info(f"当前自采占比：{(len(all_imgs)-pub)/len(all_imgs):.1%}（{len(all_imgs)-pub}/{len(all_imgs)}），"
               f"题目要求 ≥30%")
        if (len(all_imgs) - pub) / len(all_imgs) < 0.30:
            C.warn("自采占比不足 30%！要么减少公开图，要么再多拍火。")
    C.info("\n下一步：人工抽检几张 ann/labels 里的 txt（确认类别没标反），再跑 autolabel")


if __name__ == "__main__":
    main()
