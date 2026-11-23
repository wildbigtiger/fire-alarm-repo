# -*- coding: utf-8 -*-
"""交付自查：对照题目要求扫一遍项目，告诉你现在缺什么。

    python scripts/check_delivery.py

只读不写（除了生成 docs/交付自查.md），不需要 GPU。
任何时候都可以跑，建议每完成一步跑一次。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C

DOCS = C.PROJECT_ROOT / "docs"

# 题目硬指标（PDF 原文数字）
NEED_IMAGES = 500
NEED_VIDEOS = 300
NEED_VIDEO_MIN = 10.0
NEED_NEG = 200
NEED_SELF_RATIO = 0.30
NEED_REVIEW_IMAGES = 200


def count_raw() -> dict:
    images = C.iter_images(C.RAW_IMAGES)
    scenes = sorted({C.scene_of(p) for p in images})
    neg_scenes = [s for s in scenes if "neg" in s.lower()]
    pub_scenes = [s for s in scenes if s.startswith("pub_")]
    n_neg = sum(1 for p in images if C.scene_of(p) in neg_scenes)
    n_pub = sum(1 for p in images if C.scene_of(p) in pub_scenes)
    n_self = len(images) - n_pub
    videos = C.iter_videos(C.RAW_VIDEOS) + C.iter_videos(C.RAW_CLIPS)
    return {
        "images": len(images),
        "scenes": len(scenes),
        "neg": n_neg,
        "pub": n_pub,
        "self": n_self,
        "self_ratio": (n_self / len(images)) if images else 0.0,
        "videos": len(videos),
        "video_files": videos,
    }


def video_seconds(files: list[Path]) -> float:
    try:
        import cv2
    except Exception:
        return -1.0
    total = 0.0
    for f in files:
        cap = cv2.VideoCapture(str(f))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps > 0:
            total += n / fps
    return total


def count_ann() -> dict:
    imgs = C.iter_images(C.RAW_IMAGES)
    lbl = C.iter_ann_labels() if hasattr(C, "iter_ann_labels") else []
    if not lbl:
        lbl = list(C.ANN_LABELS.rglob("*.txt")) if C.ANN_LABELS.exists() else []
    raw = list(C.ANN_RAW.rglob("*.json")) if C.ANN_RAW.exists() else []
    return {"images": len(imgs), "labels": len(lbl), "raw_json": len(raw)}


def count_split() -> dict:
    out = {}
    for s in C.SPLITS:
        imgs = C.iter_images(C.IMG_DIR / s)
        boxes = 0
        for p in imgs:
            boxes += len(C.read_label(C.LBL_DIR / s / f"{p.stem}.txt"))
        out[s] = {"images": len(imgs), "boxes": boxes}
    return out


def find_train_runs() -> list[dict]:
    runs = []
    base = C.RUNS_DIR / "detect"
    if base.exists():
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            runs.append({
                "name": d.name,
                "best": (d / "weights" / "best.pt").exists(),
                "curve": (d / "results.png").exists(),
                "csv": (d / "results.csv").exists(),
                "repro": (d / "repro.json").exists(),
            })
    return runs


def find_evals() -> list[str]:
    base = C.RUNS_DIR / "eval"
    if not base.exists():
        return []
    return [d.name for d in sorted(base.iterdir()) if (d / "test_metrics.md").exists()]


def main() -> None:
    rows: list[tuple[str, str, str, bool]] = []  # 要求, 现状, 证据, 是否达标

    # ---------- 数据 ----------
    raw = count_raw()
    secs = video_seconds(raw["video_files"])
    rows.append(("火灾场景图像 ≥500 幅", f"{raw['images']} 张", "raw/images/", raw["images"] >= NEED_IMAGES))
    rows.append(("火灾视频 ≥300 段", f"{raw['videos']} 段", "raw/videos + raw/clips", raw["videos"] >= NEED_VIDEOS))
    rows.append(("视频总时长 ≥10 分钟",
                 f"{secs/60:.1f} 分钟" if secs >= 0 else "未统计（无 opencv）",
                 "raw/videos + raw/clips", secs >= NEED_VIDEO_MIN * 60))
    rows.append(("负样本 ≥200 幅", f"{raw['neg']} 张", "带 neg 的场景", raw["neg"] >= NEED_NEG))
    rows.append(("自行采集占比 ≥30%", f"{raw['self_ratio']:.1%}", "非 pub_ 开头的场景",
                 raw["self_ratio"] >= NEED_SELF_RATIO))
    rows.append(("场景数 ≥3（够划分三集）", f"{raw['scenes']} 个", "raw/images/<场景>", raw["scenes"] >= 3))

    # ---------- 标注 ----------
    ann = count_ann()
    rows.append(("预标注覆盖率",
                 f"{ann['labels']}/{ann['images']} 张" if ann["images"] else "0",
                 "ann/labels/", ann["images"] > 0 and ann["labels"] >= ann["images"]))
    rows.append(("标注可追溯（原始 json）", f"{ann['raw_json']} 个", "ann/raw/", ann["raw_json"] > 0))
    rows.append(("双人独立标注 ≥200 幅 + 一致性 IoU",
                 "需人工统计", "报告章节 / 自制统计表", False))

    # ---------- 划分与体检 ----------
    split = count_split()
    total_split = sum(v["images"] for v in split.values())
    rows.append(("数据集已划分 train/val/test",
                 f"train {split['train']['images']} / val {split['val']['images']} / test {split['test']['images']}",
                 "meta/split.json", total_split > 0 and all(v["images"] for v in split.values())))
    rows.append(("数据集体检报告", "已生成" if (C.META_DIR / "label_check.md").exists() else "无",
                 "meta/label_check.md", (C.META_DIR / "label_check.md").exists()))
    rows.append(("数据集统计表", "已生成" if (C.META_DIR / "dataset_stats.md").exists() else "无",
                 "meta/dataset_stats.md", (C.META_DIR / "dataset_stats.md").exists()))

    # ---------- 模型 ----------
    runs = find_train_runs()
    good = [r for r in runs if r["best"] and r["curve"] and r["repro"]]
    rows.append(("训练完成（权重+曲线+复现记录）",
                 f"{len(good)} 个合格实验 / 共 {len(runs)} 个",
                 "runs/detect/", len(good) >= 1))

    evals = find_evals()
    rows.append(("测试集评估（P/R/F1/mAP/FPS）", f"{len(evals)} 份", "runs/eval/", len(evals) >= 1))
    rows.append(("优化策略前后对比", f"{len(good)} 个实验可对比", "runs/detect/", len(good) >= 2))

    ov = [e for e in evals if "openvocab" in e]
    rows.append(("开放域模型对比（子任务4）",
                 f"{len(ov)} 份零样本评估" if ov else "未完成",
                 "runs/eval/openvocab_*/", len(ov) >= 1))

    seg_dir = C.RUNS_DIR / "segment"
    seg_ok = list(seg_dir.glob("*/summary.md")) if seg_dir.exists() else []
    rows.append(("像素级掩码与面积占比",
                 f"{len(seg_ok)} 份掩码统计" if seg_ok else "未完成",
                 "runs/segment/*/summary.md", len(seg_ok) >= 1))

    # ---------- 系统与提交 ----------
    ui = (C.PROJECT_ROOT / "gui" / "app.py")
    rows.append(("系统界面原型", "已实现（gui/app.py）" if ui.exists() else "未找到",
                 "gui/app.py", ui.exists()))
    rows.append(("系统演示视频", "未找到", "3_演示视频/", False))
    rows.append(("设计报告（按模板）", "未找到", "2_设计报告/", False))
    rows.append(("数据说明文档", "未找到（可据 dataset_stats.md 写）", "7_说明文档/", False))
    rows.append(("算力记录（型号+时长）", "RTX 5060 Laptop 8GB（时长见 repro.json）", "repro.json", bool(good)))
    rows.append(("源代码版本库", "已存在 .git" if (C.PROJECT_ROOT / ".git").exists() else "未初始化 git",
                 ".git", (C.PROJECT_ROOT / ".git").exists()))

    # ---------- 输出 ----------
    ok = sum(1 for r in rows if r[3])
    C.banner(f"交付自查：{ok}/{len(rows)} 项达标")
    C.info(f"{'状态':4} {'要求':38} {'现状':34} 证据")
    for req, cur, ev, done in rows:
        mark = "[OK]" if done else "[--]"
        C.info(f"{mark:4} {req:38} {cur:34} {ev}")

    # 写文件
    C.ensure_dir(DOCS)
    md = ["# 交付自查报告", "", f"生成时间：{C.now_str()}", "",
          f"**当前达标 {ok}/{len(rows)} 项**", "",
          "| 状态 | 要求 | 现状 | 证据 |", "|---|---|---|---|"]
    for req, cur, ev, done in rows:
        md.append(f"| {'✅' if done else '❌'} | {req} | {cur} | `{ev}` |")
    md += ["", "## 下一步建议", ""]
    todo = [r[0] for r in rows if not r[3]]
    for i, t in enumerate(todo[:5], 1):
        md.append(f"{i}. {t}")
    (DOCS / "交付自查.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    C.info(f"\n报告：{DOCS / '交付自查.md'}")
    if todo:
        C.info("最该先做的几件：")
        for i, t in enumerate(todo[:5], 1):
            C.info(f"  {i}. {t}")


if __name__ == "__main__":
    main()
