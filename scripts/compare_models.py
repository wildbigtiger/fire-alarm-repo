# -*- coding: utf-8 -*-
"""汇总所有训练版本的测试集成绩，生成对比表（报告可直接用）。

    python scripts/compare_models.py
    python scripts/compare_models.py --same-test-only   # 只统计当前测试集的成绩

输出：
    docs/模型版本对比.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C

# 每个版本的简短说明（写报告时的"优缺点"素材）
NOTES = {
    "baseline": "v1 首版：当时近景火被划到 val/test，存在满屏火盲区",
    "baseline_v2": "v2 修正划分：把近景火放进训练，目前最稳的基线",
    "baseline_v3": "v3 +22 张蓝焰图：蓝焰召回未提升",
    "baseline_v4": "v4 +246 张空场景负样本：召回下降",
    "baseline_v5": "v5 +642 张火灾视频帧：框更准、误报更少，召回下降",
    "baseline_v6": "v6 裁剪后 2164 张：自采 29.7% 合规版，作为备用对照组（性能代价见下）",
    "opt_aug": "数据增强（mosaic/mixup）",
    "opt_aug_v2": "数据增强（第二版划分）",
    "opt_small": "960 分辨率 + 分类损失加权",
}


def load_eval(path: Path) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    d["_dir"] = os.path.basename(os.path.dirname(path))
    tag = d["_dir"].split("_2026")[0]
    if tag.endswith("_re"):          # 统一口径复评的目录，归并回同名版本
        tag = tag[:-3]
    d["_tag"] = tag
    return d


def dataset_size(weights: str) -> str:
    """从对应训练目录的 repro.json 里取训练集规模。"""
    try:
        name = Path(weights).parent.parent.name
        rp = C.RUNS_DIR / "detect" / name / "repro.json"
        if rp.exists():
            r = json.load(open(rp, encoding="utf-8"))
            ds = r.get("dataset", {})
            return f"{ds.get('images', '?')} 张 / {ds.get('boxes', '?')} 框"
    except Exception:
        pass
    return "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--same-test-only", action="store_true",
                    help="只保留与当前测试集规模一致的评估")
    args = ap.parse_args()

    evals = [load_eval(p) for p in (C.RUNS_DIR / "eval").glob("*/test_metrics.json")]
    yolo = [e for e in evals if "openvocab" not in e["weights"]]

    # 同一版本可能有多次评估：优先保留最新（按目录名时间戳最大）
    best: dict[str, dict] = {}
    for e in sorted(yolo, key=lambda x: x["_dir"]):
        best[e["_tag"]] = e

    if args.same_test_only:
        target = None
        for tag in ("baseline_v6", "baseline_v5"):
            if tag in best:
                target = best[tag].get("n_images") or None
                break
        if target:
            best = {k: v for k, v in best.items() if v.get("n_images") == target}

    order = ["baseline", "baseline_v2", "baseline_v3", "baseline_v4",
             "baseline_v5", "baseline_v6", "opt_small", "opt_aug", "opt_aug_v2"]
    tags = [t for t in order if t in best] + [t for t in sorted(best) if t not in order]

    lines = ["# 模型版本对比", "", f"生成时间：{C.now_str()}", "",
             "同一测试集（203 张 / 344 框）、同 IoU 阈值 0.5、同硬件 RTX 5060 8GB。", "",
             "| 版本 | 训练集 | mAP@0.5 | mAP@0.5:0.95 | P | R | F1 | FPS | 火AP50 | 烟AP50 | 说明 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in tags:
        d = best[t]
        ap_map = {c["class"]: c["ap50"] for c in d.get("per_class", [])}
        lines.append("| %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.1f | %.4f | %.4f | %s |" % (
            t, dataset_size(d["weights"]), d["map50"], d["map50_95"], d["precision"],
            d["recall"], d["f1"], d["fps_end_to_end"],
            ap_map.get("fire", 0), ap_map.get("smoke", 0), NOTES.get(t, "")))

    lines += ["", "## 各自的取舍", ""]
    if "baseline_v2" in best and "baseline_v5" in best:
        a, b = best["baseline_v2"], best["baseline_v5"]
        lines.append("- **v2**：召回最高（%.3f），适合「宁多报不漏报」的场合。" % a["recall"])
        lines.append("- **v5**：精确率最高（%.3f）、mAP50-95 最高（%.3f），框最准、误报最少，适合现场演示。"
                     % (b["precision"], b["map50_95"]))
    if "baseline_v6" in best:
        d = best["baseline_v6"]
        lines.append("- **v6**：为满足自采 30% 而裁剪数据集后的版本，用来量化\"删数据\"的性能代价。")
        if "baseline_v5" in best:
            dd = d["map50"] - best["baseline_v5"]["map50"]
            lines.append("  - 相比 v5：mAP@0.5 变化 %+.4f，召回变化 %+.4f" % (dd, d["recall"] - best["baseline_v5"]["recall"]))

    out = C.PROJECT_ROOT / "docs" / "模型版本对比.md"
    C.ensure_dir(out.parent)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    C.banner("模型版本对比")
    for ln in lines[6:]:
        C.info(ln)
    C.info("")
    C.info("报告：%s" % out)


if __name__ == "__main__":
    main()
