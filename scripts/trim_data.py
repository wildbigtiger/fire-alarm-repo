# -*- coding: utf-8 -*-
"""按目标比例裁剪数据集（只删公开数据，自采一张不动）。

    python scripts/trim_data.py --dry-run      # 预演
    python scripts/trim_data.py                # 执行

三个目标尽量同时满足：
  1. 自行采集占比 >= self-ratio（默认 0.30）
  2. 有目标图像 : 无目标图像 ~= pos-neg-ratio（默认 2.5:1，对应题目 500:200）
  3. 负样本总数仍 >= 200

删法优先级：全负样本的公开场景 > 公开场景里的无目标图像 > 框密度最低的公开场景里的有目标图像。

被删文件移到 datasets/fire/removed/ 保持原结构，随时可搬回。
测试集场景与 --protect 场景永不触碰。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C

REMOVED = C.DATASET_ROOT / "removed"


def scene_stats() -> dict:
    out = {}
    for d in sorted(p for p in C.RAW_IMAGES.iterdir() if p.is_dir()):
        imgs = C.iter_images(d)
        if not imgs:
            continue
        pos = neg = boxes = 0
        for p in imgs:
            rows = C.read_label(C.ANN_LABELS / d.name / (p.stem + ".txt"))
            if rows:
                pos += 1
            else:
                neg += 1
            boxes += len(rows)
        out[d.name] = {"img": len(imgs), "pos": pos, "neg": neg, "box": boxes,
                       "density": boxes / len(imgs) if imgs else 0}
    return out


def move_out(scene: str, stems: list) -> int:
    n = 0
    for st in stems:
        for p in (C.RAW_IMAGES / scene).glob(st + ".*"):
            dst = REMOVED / "images" / scene / p.name
            C.ensure_dir(dst.parent)
            shutil.move(str(p), str(dst))
            n += 1
        lp = C.ANN_LABELS / scene / (st + ".txt")
        if lp.exists():
            dst = REMOVED / "labels" / scene / lp.name
            C.ensure_dir(dst.parent)
            shutil.move(str(lp), str(dst))
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="按比例裁剪公开数据")
    ap.add_argument("--self-ratio", type=float, default=0.30)
    ap.add_argument("--pos-neg-ratio", type=float, default=2.5)
    ap.add_argument("--min-neg", type=int, default=200)
    ap.add_argument("--protect", default="pub_blueflame")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    protected = {s.strip() for s in args.protect.split(",") if s.strip()}
    plan = (C.load_json(C.META_DIR / "split.json", {}) or {}).get("scene_plan", {})
    test_scenes = {s for s, v in plan.items() if v == "test"}
    skip = test_scenes | protected

    stats = scene_stats()
    self_scenes = [s for s in stats if not s.startswith("pub_")]
    pub_scenes = [s for s in stats if s.startswith("pub_")]

    self_img = sum(stats[s]["img"] for s in self_scenes)
    self_pos = sum(stats[s]["pos"] for s in self_scenes)
    self_neg = sum(stats[s]["neg"] for s in self_scenes)
    pub_img = sum(stats[s]["img"] for s in pub_scenes)
    pub_pos = sum(stats[s]["pos"] for s in pub_scenes)
    pub_neg = sum(stats[s]["neg"] for s in pub_scenes)

    total = self_img + pub_img
    pos, neg = self_pos + pub_pos, self_neg + pub_neg

    C.banner("裁剪前")
    C.info("自采 %d（有目标 %d / 无目标 %d）" % (self_img, self_pos, self_neg))
    C.info("公开 %d（有目标 %d / 无目标 %d）" % (pub_img, pub_pos, pub_neg))
    C.info("合计 %d：有目标 %d，无目标 %d，比例 %.2f:1，自采占比 %.1f%%"
           % (total, pos, neg, pos / max(1, neg), 100.0 * self_img / total))
    C.info("保护场景：%s" % sorted(protected))

    total_target = int(self_img / args.self_ratio)
    removals = total - total_target
    if removals <= 0:
        C.info("已经达标，不需要删。")
        return

    x = (args.pos_neg_ratio * neg - pos + removals) / (1 + args.pos_neg_ratio)
    n_neg = int(round(max(0, min(x, pub_neg))))
    n_pos = removals - n_neg
    max_neg_del = max(0, neg - args.min_neg)
    if n_neg > max_neg_del:
        n_neg = max_neg_del
        n_pos = removals - n_neg
    if n_pos > pub_pos:
        n_pos = pub_pos

    C.info("")
    C.info("计划：删 %d 张公开数据（负样本 %d + 有目标 %d）" % (removals, n_neg, n_pos))
    C.info("预计结果：%d 张，有目标 %d，无目标 %d，比例 %.2f:1，自采占比 %.1f%%"
           % (total - removals, pos - n_pos, neg - n_neg,
              (pos - n_pos) / max(1, neg - n_neg), 100.0 * self_img / (total - removals)))

    neg_picks = {}
    left = n_neg
    for s in sorted(pub_scenes, key=lambda k: -stats[k]["neg"]):
        if left <= 0:
            break
        if s in skip:
            continue
        if stats[s]["pos"] == 0:
            take = min(stats[s]["neg"], left)
            stems = [p.stem for p in C.iter_images(C.RAW_IMAGES / s)][:take]
            neg_picks[s] = stems
            left -= len(stems)

    for s in sorted(pub_scenes, key=lambda k: -stats[k]["neg"]):
        if left <= 0:
            break
        if s in skip or s in neg_picks:
            continue
        stems = []
        for p in C.iter_images(C.RAW_IMAGES / s):
            if left <= 0:
                break
            if not C.read_label(C.ANN_LABELS / s / (p.stem + ".txt")):
                stems.append(p.stem)
                left -= 1
        if stems:
            neg_picks[s] = stems

    pos_picks = {}
    left = n_pos
    for s in sorted(pub_scenes, key=lambda k: stats[k]["density"]):
        if left <= 0:
            break
        if s in skip:
            continue
        stems = []
        for p in C.iter_images(C.RAW_IMAGES / s):
            if left <= 0:
                break
            if C.read_label(C.ANN_LABELS / s / (p.stem + ".txt")):
                stems.append(p.stem)
                left -= 1
        if stems:
            pos_picks[s] = stems

    C.info("")
    C.info("各场景删除明细：")
    for s in sorted(set(neg_picks) | set(pos_picks)):
        C.info("  %-26s 负样本 %4d  有目标 %4d"
               % (s, len(neg_picks.get(s, [])), len(pos_picks.get(s, []))))

    if args.dry_run:
        C.info("")
        C.info("--dry-run，未改动任何文件。")
        return

    moved = 0
    for s in neg_picks:
        moved += move_out(s, neg_picks[s])
    for s in pos_picks:
        moved += move_out(s, pos_picks[s])
    C.info("")
    C.info("已移出 %d 张图像到 %s" % (moved, REMOVED))

    stats2 = scene_stats()
    total2 = sum(v["img"] for v in stats2.values())
    pos2 = sum(v["pos"] for v in stats2.values())
    neg2 = sum(v["neg"] for v in stats2.values())
    self2 = sum(v["img"] for s, v in stats2.items() if not s.startswith("pub_"))
    C.banner("裁剪后")
    C.info("合计 %d：有目标 %d，无目标 %d，比例 %.2f:1"
           % (total2, pos2, neg2, pos2 / max(1, neg2)))
    C.info("自采 %d，占比 %.1f%%" % (self2, 100.0 * self2 / total2))
    C.info("")
    C.info("下一步：split_dataset.py -> check_labels.py -> train_yolo.py")


if __name__ == "__main__":
    main()
