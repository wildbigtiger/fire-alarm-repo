# -*- coding: utf-8 -*-
"""子任务 4：开放域模型（GroundingDINO）零样本评估，与自训 YOLO 逐项对比。

    python scripts/eval_openvocab.py --limit 20          # 先试 20 张
    python scripts/eval_openvocab.py                      # 全量测试集

关键点（对应题目要求）：
  * **未针对火灾数据微调**，纯零样本推理，提示词驱动
  * 与子任务 3 用**同一测试集**、**同一 IoU 阈值**、**同一输入分辨率**
  * 输出与 eval_yolo.py 完全相同的指标结构，方便直接并排比较
  * 明确记录：是否微调=否、提示词、模型版本、推理参数

输出：
    runs/eval/openvocab_<时间>/test_metrics.md
    runs/eval/openvocab_<时间>/test_metrics.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C
from autolabel import PROMPT_GROUPS, build_prompt, phrase_to_class

MODEL = "IDEA-Research/grounding-dino-tiny"


# ---------------------------------------------------------------- AP 计算
def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def compute_ap(preds: list[dict], gts: list[dict], cls_id: int, tp_iou: float):
    """单类别单 IoU 阈值的 AP（全点插值法，VOC2010+ 口径）。"""
    n_gt = sum(1 for g in gts if g["cls"] == cls_id)
    if n_gt == 0:
        return None, 0, 0
    ps = sorted([p for p in preds if p["cls"] == cls_id], key=lambda x: -x["score"])
    matched: set[tuple[str, int]] = set()
    tp, fp = [], []
    for p in ps:
        cands = [(i, g) for i, g in enumerate(gts) if g["cls"] == cls_id and g["image"] == p["image"]]
        best_iou, best_i = 0.0, -1
        for i, g in cands:
            if (p["image"], i) in matched:
                continue
            v = iou_xyxy(p["box"], g["box"])
            if v > best_iou:
                best_iou, best_i = v, i
        if best_iou >= tp_iou and best_i >= 0:
            matched.add((p["image"], best_i))
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    if not tp:
        return 0.0, 0, n_gt
    import numpy as np

    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    rec = tp_c / n_gt
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, int(tp_c[-1]), n_gt


def main() -> None:
    ap = argparse.ArgumentParser(description="开放域模型零样本评估")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--box-thresh", type=float, default=0.05, help="低阈值跑全量预测，mAP 需要")
    ap.add_argument("--text-thresh", type=float, default=0.15)
    ap.add_argument("--conf", type=float, default=0.25, help="算 P/R/F1 的工作阈值")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    classes = C.load_classes()
    imgs = C.iter_images(C.IMG_DIR / args.split)
    if not imgs:
        C.die(f"{args.split} 集没有图片，先跑 split_dataset.py")
    if args.limit:
        imgs = imgs[: args.limit]

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu"
    prompt = build_prompt(PROMPT_GROUPS)
    C.banner(f"开放域零样本评估：{args.model} on {args.split}（{len(imgs)} 张）")
    C.info(f"提示词：{prompt}")
    C.info("是否微调：**否**（纯零样本，权重为官方预训练）")

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()

    preds: list[dict] = []
    gts: list[dict] = []
    lat_ms: list[float] = []

    for k, p in enumerate(imgs, 1):
        image = Image.open(p).convert("RGB")
        W, H = image.size
        # 真值
        for i, (c, cx, cy, w, h) in enumerate(C.read_label(C.LBL_DIR / args.split / f"{p.stem}.txt")):
            gts.append({"image": p.name, "cls": c,
                        "box": [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]})
        # 推理
        t0 = time.time()
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        try:
            res = processor.post_process_grounded_object_detection(
                out, inputs.input_ids, threshold=args.box_thresh,
                text_threshold=args.text_thresh, target_sizes=[image.size[::-1]])[0]
        except TypeError:
            res = processor.post_process_grounded_object_detection(
                out, inputs.input_ids, box_threshold=args.box_thresh,
                text_threshold=args.text_thresh, target_sizes=[image.size[::-1]])[0]
        lat_ms.append((time.time() - t0) * 1000)

        phrases = res.get("text_labels", res.get("labels")) or []
        for box, score, ph in zip(res["boxes"].tolist(), res["scores"].tolist(), phrases):
            cname = phrase_to_class(str(ph), PROMPT_GROUPS)
            if cname not in classes:
                continue
            preds.append({"image": p.name, "cls": classes.index(cname),
                          "score": float(score), "box": [float(v) for v in box]})
        if k % 20 == 0:
            C.info(f"  {k}/{len(imgs)}")

    # ---- 指标 ----
    def eval_at(th: float) -> tuple[float, list]:
        aps = []
        for ci in range(len(classes)):
            a, _, _ = compute_ap(preds, gts, ci, th)
            aps.append(a)
        aps = [a for a in aps if a is not None]
        return (float(np.mean(aps)) if aps else 0.0), aps

    map50, aps50 = eval_at(0.5)
    ious = [0.5 + 0.05 * i for i in range(10)]
    map5095 = float(np.mean([eval_at(t)[0] for t in ious]))

    # 工作阈值下的 P/R/F1（直接按 conf 过滤后统计，与 YOLO 侧口径一致）
    kept = [p for p in preds if p["score"] >= args.conf]
    n_gt = len(gts)
    tp = 0
    used: set[tuple[str, int]] = set()
    for cname_i in range(len(classes)):
        cp = sorted([p for p in kept if p["cls"] == cname_i], key=lambda x: -x["score"])
        for p in cp:
            best, bi = 0.0, -1
            for i, g in enumerate(gts):
                if g["cls"] != cname_i or g["image"] != p["image"] or (p["image"], i) in used:
                    continue
                v = iou_xyxy(p["box"], g["box"])
                if v > best:
                    best, bi = v, i
            if best >= 0.5 and bi >= 0:
                used.add((p["image"], bi)); tp += 1
    prec = tp / len(kept) if kept else 0.0
    rec = tp / n_gt if n_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    mean_lat = float(np.mean(lat_ms))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
    env = C.env_report()
    gpu = ", ".join(env.get("gpu", [])) if env.get("gpu") else "CPU"

    out_dir = C.RUNS_DIR / "eval" / f"openvocab_{C.stamp()}"
    C.ensure_dir(out_dir)

    md = [
        "# 开放域模型零样本评估（子任务 4）", "",
        f"- 模型：`{args.model}`（**未针对火灾数据微调**）",
        f"- 提示词：`{prompt}`",
        f"- 测试集：`{args.split}`，{len(imgs)} 张，{n_gt} 个真值框",
        f"- 硬件：{gpu}；输入：原图不缩放（GroundingDINO 内部 resize）；batch=1",
        f"- 阈值：box_thresh={args.box_thresh}，text_thresh={args.text_thresh}，"
        f"P/R/F1 工作阈值 conf={args.conf}",
        "",
        "## 总体指标", "",
        "| 指标 | 数值 | 说明 |", "|---|---|---|",
        f"| mAP@0.5 | **{map50:.4f}** | 与 YOLO 同一测试集、同一 IoU 口径 |",
        f"| mAP@0.5:0.95 | **{map5095:.4f}** | 10 个 IoU 阈值平均 |",
        f"| Precision | {prec:.4f} | conf={args.conf} |",
        f"| Recall | {rec:.4f} | conf={args.conf} |",
        f"| F1 | {f1:.4f} | |",
        f"| 单帧耗时 | {mean_lat:.1f} ms | 纯推理，不含后处理细分 |",
        f"| FPS | {fps:.1f} | 端到端（含预处理与后处理） |",
        "",
        "## 分类别 AP@0.5", "",
        "| 类别 | AP@0.5 |", "|---|---|",
    ]
    for ci, a in enumerate(aps50):
        if ci < len(classes):
            md.append(f"| {classes[ci]} | {a:.4f} |")

    md += [
        "", "## 说明（写报告用）", "",
        "1. 本模型**未用火灾数据训练**，仅靠自然语言提示词驱动，属于零样本泛化。",
        f"2. 所有指标与子任务 3 的 YOLO 在**同一测试集（{len(imgs)} 张）**、"
        f"**同一 IoU 阈值（0.5）**下取得。",
        "3. 推理速度对比时注意：两者输入预处理不同，表中已注明各自设置。",
    ]
    (out_dir / "test_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    C.save_json(out_dir / "test_metrics.json", {
        "weights": f"openvocab/{args.model}", "model_id": args.model, "finetuned": False,
        "prompt": prompt, "split": args.split, "n_images": len(imgs),
        "imgsz": 640, "batch": 1, "conf": args.conf, "iou": 0.5,
        "map50": map50, "map50_95": map5095, "precision": prec, "recall": rec, "f1": f1,
        "fps_end_to_end": fps, "fps_inference_only": fps,
        "latency_ms": mean_lat,
        "per_class": [{"class": classes[i], "ap50": a} for i, a in enumerate(aps50) if i < len(classes)],
        "hardware": gpu, "env": env, "time": C.now_str(),
    })

    C.banner("结果")
    C.info(f"mAP@0.5 = {map50:.4f}   mAP@0.5:0.95 = {map5095:.4f}")
    C.info(f"P = {prec:.4f}  R = {rec:.4f}  F1 = {f1:.4f}  (conf={args.conf})")
    C.info(f"单帧 {mean_lat:.1f} ms  ->  {fps:.1f} FPS")
    C.info(f"\n报告：{out_dir / 'test_metrics.md'}")


if __name__ == "__main__":
    main()
