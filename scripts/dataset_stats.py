# -*- coding: utf-8 -*-
"""生成数据集统计表，直接可以贴进设计报告。

    python scripts/dataset_stats.py

输出：
    datasets/fire/meta/dataset_stats.md    给人看（报告用）
    datasets/fire/meta/dataset_stats.csv   给机器看

约定：场景目录名以 pub_ 开头的算"公开数据集"，其余算"自行采集"。
这样脚本能自动算出"自行采集占比 ≥30%"这条硬性要求，不用手数。
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C


def video_stats() -> dict:
    files = C.iter_videos(C.RAW_VIDEOS) + C.iter_videos(C.RAW_CLIPS)
    total_sec = 0.0
    ok = False
    try:
        import cv2

        ok = True
        for f in files:
            cap = cv2.VideoCapture(str(f))
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            if fps > 0:
                total_sec += n / fps
    except Exception:
        pass
    return {"files": len(files), "seconds": total_sec, "duration_ok": ok}


def main() -> None:
    classes = C.load_classes()
    split_info = C.load_json(C.META_DIR / "split.json", {}) or {}
    plan: dict[str, str] = split_info.get("scene_plan", {})

    if not any((C.IMG_DIR / s).exists() and any((C.IMG_DIR / s).iterdir()) for s in C.SPLITS):
        C.die("还没有划分好的数据集，先跑 python scripts/split_dataset.py")

    per_split = {}
    scene_rows: dict[str, dict] = defaultdict(lambda: {"images": 0, "boxes": 0, "neg": 0,
                                                       "per_class": Counter(), "split": "-"})
    for split in C.SPLITS:
        imgs = C.iter_images(C.IMG_DIR / split)
        n_box = n_neg = 0
        per_class = Counter()
        for p in imgs:
            scene = p.stem.split("__")[0]
            rows = C.read_label(C.LBL_DIR / split / f"{p.stem}.txt")
            n_box += len(rows)
            if not rows:
                n_neg += 1
            for r in rows:
                per_class[classes[r[0]]] += 1
            s = scene_rows[scene]
            s["images"] += 1
            s["boxes"] += len(rows)
            s["neg"] += 0 if rows else 1
            s["per_class"].update(classes[r[0]] for r in rows)
            s["split"] = split
        per_split[split] = {"images": len(imgs), "boxes": n_box, "negatives": n_neg,
                            "per_class": per_class}

    total_imgs = sum(v["images"] for v in per_split.values())
    total_boxes = sum(v["boxes"] for v in per_split.values())

    # 自采 / 公开
    self_scenes = [s for s in scene_rows if not s.startswith("pub_")]
    pub_scenes = [s for s in scene_rows if s.startswith("pub_")]
    self_imgs = sum(scene_rows[s]["images"] for s in self_scenes)
    pub_imgs = sum(scene_rows[s]["images"] for s in pub_scenes)
    self_ratio = self_imgs / total_imgs if total_imgs else 0.0

    vs = video_stats()

    # ---------------- markdown ----------------
    md: list[str] = []
    md.append("# 数据集统计表")
    md.append("")
    md.append(f"生成时间：{C.now_str()}")
    md.append(f"图片总数：**{total_imgs}** 张；目标框总数：**{total_boxes}** 个；"
              f"场景数：**{len(scene_rows)}** 个  ")
    md.append(f"视频片段：**{vs['files']}** 段，总时长约 **{vs['seconds']/60:.1f}** 分钟"
              + ("" if vs["duration_ok"] else "（未安装 opencv，时长未统计）"))
    md.append("")
    md.append("## 1. 划分比例")
    md.append("")
    md.append("| 子集 | 图片数 | 占比 | 目标框 | 负样本 | " + " | ".join(classes) + " |")
    md.append("|---|---|---|---|---|" + "---|" * len(classes))
    for s in C.SPLITS:
        d = per_split[s]
        pct = d["images"] / total_imgs if total_imgs else 0
        md.append(f"| {s} | {d['images']} | {pct:.1%} | {d['boxes']} | {d['negatives']} | "
                  + " | ".join(str(d["per_class"].get(c, 0)) for c in classes) + " |")
    md.append(f"| **合计** | {total_imgs} | 100% | {total_boxes} | "
              f"{sum(v['negatives'] for v in per_split.values())} | "
              + " | ".join(str(sum(v["per_class"].get(c, 0) for v in per_split.values())) for c in classes)
              + " |")
    md.append("")
    md.append("## 2. 数据来源")
    md.append("")
    md.append("| 来源 | 场景数 | 图片数 | 占比 |")
    md.append("|---|---|---|---|")
    md.append(f"| 自行采集 | {len(self_scenes)} | {self_imgs} | {self_ratio:.1%} |")
    md.append(f"| 公开数据集 | {len(pub_scenes)} | {pub_imgs} | {(1-self_ratio):.1%} |")
    md.append("")
    if self_ratio < 0.30:
        md.append(f"> ⚠️ 自行采集占比 {self_ratio:.1%} 低于题目要求的 30%，需要再补自采数据。")
    else:
        md.append(f"> ✅ 自行采集占比 {self_ratio:.1%}，满足题目 ≥30% 的要求。")
    md.append("")
    md.append("## 3. 场景分布")
    md.append("")
    md.append("| 场景 | split | 图片 | 目标框 | 负样本 | " + " | ".join(classes) + " |")
    md.append("|---|---|---|---|---|" + "---|" * len(classes))
    for scene in sorted(scene_rows):
        d = scene_rows[scene]
        md.append(f"| {scene} | {d['split']} | {d['images']} | {d['boxes']} | {d['neg']} | "
                  + " | ".join(str(d["per_class"].get(c, 0)) for c in classes) + " |")
    md.append("")
    if per_split["test"]["images"]:
        md.append("## 4. 测试集说明")
        md.append("")
        test_scenes = [s for s, sp in plan.items() if sp == "test"]
        md.append(f"- 测试集场景：{', '.join(sorted(test_scenes)) or '（未记录）'}")
        md.append(f"- 是否人工全量复核："
                  f"{'是' if set(test_scenes) <= set(split_info.get('reviewed_scenes', [])) else '否（需确认）'}")
        md.append(f"- 测试集负样本：{per_split['test']['negatives']} 张，用于统计误报率")
        md.append("")

    out_md = C.META_DIR / "dataset_stats.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---------------- csv ----------------
    out_csv = C.META_DIR / "dataset_stats.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["scene", "split", "images", "boxes", "negatives", *classes])
        for scene in sorted(scene_rows):
            d = scene_rows[scene]
            w.writerow([scene, d["split"], d["images"], d["boxes"], d["neg"],
                        *(d["per_class"].get(c, 0) for c in classes)])

    C.banner("数据集统计")
    C.info(f"{'子集':8}{'图片':>8}{'目标框':>8}{'负样本':>8}")
    for s in C.SPLITS:
        d = per_split[s]
        C.info(f"{s:8}{d['images']:>8}{d['boxes']:>8}{d['negatives']:>8}")
    C.info(f"{'合计':8}{total_imgs:>8}{total_boxes:>8}")
    C.info(f"\n自采 {self_imgs} 张（{self_ratio:.1%}），公开 {pub_imgs} 张；"
           f"视频 {vs['files']} 段 / {vs['seconds']/60:.1f} 分钟")
    if self_ratio < 0.30:
        C.warn(f"自采占比 {self_ratio:.1%} < 30%，题目要求自采不少于 30%，需要补数据")
    C.info(f"\n报告用表格：{out_md}\nCSV：{out_csv}")


if __name__ == "__main__":
    main()
