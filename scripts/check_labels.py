# -*- coding: utf-8 -*-
"""数据集体检：查标签格式、查数据泄漏、查测试集是否"干净"。

这一步比训练重要。标签有问题，训练再久也白搭。

    python scripts/check_labels.py
    python scripts/check_labels.py --fix     # 顺手修掉越界坐标和非法行

检查项：
  1. 每个 split 的图片和标签是否一一对应
  2. 标签格式（5 列 / 类别 id / 坐标范围 / 框大小）
  3. 文件名是否跨 split 重复（同一张图混进两个 split = 泄漏）
  4. 同一个场景是否被拆到了不同 split（最隐蔽的泄漏）
  5. 测试集是否包含负样本（不然算不出误报率）
  6. 测试集场景是否标记为人工复核过（不然评估大模型会循环论证）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C


def check(args) -> int:
    classes = C.load_classes()
    split_info = C.load_json(C.META_DIR / "split.json", {}) or {}
    reviewed = set(split_info.get("reviewed_scenes", []))

    errors: list[str] = []
    warnings: list[str] = []
    lines: list[str] = []
    stats: dict[str, dict] = {}

    for split in C.SPLITS:
        img_dir, lbl_dir = C.IMG_DIR / split, C.LBL_DIR / split
        images = C.iter_images(img_dir)
        labels = sorted(p for p in lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        missing_lbl = img_stems - lbl_stems
        orphan_lbl = lbl_stems - img_stems
        if missing_lbl:
            errors.append(f"{split}: {len(missing_lbl)} 张图片缺标签（例：{sorted(missing_lbl)[0]}）")
        if orphan_lbl:
            errors.append(f"{split}: {len(orphan_lbl)} 个标签没有对应图片（例：{sorted(orphan_lbl)[0]}）")

        n_box = n_neg = n_fix = 0
        per_class = {c: 0 for c in classes}
        bad_lines = 0

        for lp in labels:
            rows = C.read_label(lp)
            if not rows:
                n_neg += 1
                continue
            fixed: list[tuple[int, float, float, float, float]] = []
            fixed_here = 0
            raw_lines = lp.read_text(encoding="utf-8", errors="ignore").splitlines()
            if len(rows) != len([l for l in raw_lines if l.strip()]):
                bad_lines += 1
            for cls, cx, cy, w, h in rows:
                if not (0 <= cls < len(classes)):
                    errors.append(f"{lp.name}: 类别 id={cls} 超出范围 0..{len(classes)-1}")
                    continue
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                    if args.fix:
                        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
                        n_fix += 1
                        fixed_here += 1
                    else:
                        errors.append(f"{lp.name}: 中心点越界 cx={cx} cy={cy}")
                        continue
                if w <= 0 or h <= 0:
                    errors.append(f"{lp.name}: 框宽高非正 w={w} h={h}")
                    continue
                if w > 1 or h > 1:
                    if args.fix:
                        w, h = min(w, 1.0), min(h, 1.0)
                        n_fix += 1
                        fixed_here += 1
                    else:
                        errors.append(f"{lp.name}: 框宽高超过 1 w={w} h={h}")
                        continue
                if w * h < 1e-5:
                    warnings.append(f"{lp.name}: 框极小（面积 {w*h:.2e}），可能是误标")
                fixed.append((cls, cx, cy, w, h))
            n_box += len(fixed)
            for cls, *_ in fixed:
                per_class[classes[cls]] += 1
            if args.fix and fixed_here:
                C.write_label(lp, fixed)

        if bad_lines:
            warnings.append(f"{split}: {bad_lines} 个标签文件含格式不合法的行（已自动忽略）")
        if n_fix:
            warnings.append(f"{split}: 自动修正了 {n_fix} 个越界框")

        stats[split] = {"images": len(images), "boxes": n_box, "negatives": n_neg, "per_class": per_class}

        if len(images) == 0:
            warnings.append(f"{split}: 没有任何图片")
        elif n_neg == 0:
            warnings.append(f"{split}: 没有负样本（无火图片）")
        if len(images) and n_neg / len(images) > 0.5:
            warnings.append(f"{split}: 负样本占 {n_neg/len(images):.0%}，比例偏高，正样本可能不足")
        for c in classes:
            if len(images) and per_class[c] == 0:
                warnings.append(f"{split}: 类别 {c} 一个目标都没有")

    # ---- 泄漏检查 1：文件名跨 split 重复 ----
    all_names: dict[str, list[str]] = {}
    for split in C.SPLITS:
        for p in C.iter_images(C.IMG_DIR / split):
            all_names.setdefault(p.name.lower(), []).append(split)
    dup = {k: v for k, v in all_names.items() if len(set(v)) > 1}
    if dup:
        errors.append(f"有 {len(dup)} 个文件名同时出现在多个 split，疑似泄漏（例：{list(dup)[0]}）")

    # ---- 泄漏检查 2：场景跨 split ----
    if split_info.get("scene_plan"):
        scene_splits: dict[str, set[str]] = {}
        for scene, sp in split_info["scene_plan"].items():
            scene_splits.setdefault(scene, set()).add(sp)
        bad = {k: v for k, v in scene_splits.items() if len(v) > 1}
        if bad:
            errors.append(f"有 {len(bad)} 个场景被拆到多个 split，必须修（例：{list(bad)[0]}）")

    # ---- 测试集是否干净 ----
    test_scenes = {s for s, sp in (split_info.get("scene_plan") or {}).items() if sp == "test"}
    if test_scenes:
        unreviewed = test_scenes - reviewed
        if unreviewed:
            warnings.append(
                "测试集里有场景没标记为人工复核过：" + ", ".join(sorted(unreviewed)[:5])
                + "。如果这些图是大模型预标注的，用它评估 GroundingDINO 就是循环论证。")
    if stats.get("test", {}).get("negatives", 1) == 0 and stats.get("test", {}).get("images", 0):
        warnings.append("测试集没有负样本，无法报告误报率（题目明确要求）")

    # ---- 输出 ----
    C.banner("数据集体检")
    md = ["# 数据集体检报告", "", f"生成时间：{C.now_str()}", "",
          "| split | 图片 | 目标框 | 负样本 | " + " | ".join(classes) + " |",
          "|---|---|---|---|" + "---|" * len(classes)]
    for s in C.SPLITS:
        d = stats[s]
        md.append(f"| {s} | {d['images']} | {d['boxes']} | {d['negatives']} | "
                  + " | ".join(str(d['per_class'][c]) for c in classes) + " |")

    if errors:
        md += ["", "## 错误（必须修）", ""] + [f"- ❌ {e}" for e in errors]
        for e in errors:
            C.info(f"[错误] {e}")
    if warnings:
        md += ["", "## 警告（建议看一眼）", ""] + [f"- ⚠️ {w}" for w in warnings]
        for w in warnings:
            C.info(f"[警告] {w}")
    if not errors and not warnings:
        md += ["", "一切正常。"]
    if not errors:
        C.info("\n[OK] 没有致命错误")
    else:
        C.info(f"\n[FAIL] {len(errors)} 个致命错误，修好再训练")

    C.save_json(C.META_DIR / "label_check.json", {"errors": errors, "warnings": warnings, "stats": stats})
    (C.META_DIR / "label_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    C.info(f"报告：{C.META_DIR / 'label_check.md'}")
    return 1 if errors else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="数据集体检")
    ap.add_argument("--fix", action="store_true", help="自动修正越界坐标（会写回标签文件）")
    args = ap.parse_args()
    sys.exit(check(args))


if __name__ == "__main__":
    main()
