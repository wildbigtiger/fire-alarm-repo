# -*- coding: utf-8 -*-
"""在**测试集**上评估，输出报告直接能用的表格。

    python scripts/eval_yolo.py --weights runs/detect/opt_small_xxx/weights/best.pt

为什么跑两遍？
    算 mAP 要把置信度阈值压到很低（0.001），否则曲线不完整；
    但报告里的 Precision/Recall/F1 需要一个"实际部署时"的阈值（默认 0.25）。
    两者混在一起会被评委问住，所以分开算、分开写。

输出：
    runs/eval/<name>/test_metrics.md
    runs/eval/<name>/test_metrics.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C


def fmt(x) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def main() -> None:
    ap = argparse.ArgumentParser(description="测试集评估")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(C.CONFIG_DIR / "fire.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1, help="测速用的 batch，报告里要写明")
    ap.add_argument("--conf", type=float, default=0.25, help="算 P/R/F1 用的工作阈值")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS 的 IoU 阈值（全流程固定）")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        C.die(f"权重不存在：{weights}")
    if not C.iter_images(C.IMG_DIR / args.split):
        C.die(f"{args.split} 集没有图片，先跑 split_dataset.py")

    name = args.name or weights.parent.parent.name
    out_dir = C.RUNS_DIR / "eval" / f"{name}_{C.stamp()}"
    C.ensure_dir(out_dir)

    import torch
    from ultralytics import YOLO

    runtime_yaml = C.prepare_data_yaml(Path(args.data), out_dir / "data.yaml")
    model = YOLO(str(weights))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    C.banner(f"评估 {weights.name} on {args.split}")
    C.info(f"图片尺寸 {args.imgsz}, batch {args.batch}, NMS IoU {args.iou}")

    # ---- 第一遍：低阈值，算 mAP ----
    res_map = model.val(data=str(runtime_yaml), split=args.split, imgsz=args.imgsz,
                        batch=args.batch, conf=0.001, iou=args.iou, device=args.device,
                        project=str(out_dir), name="map", plots=True, verbose=False)
    b = res_map.box
    map50, map5095 = float(b.map50), float(b.map)

    # ---- 第二遍：工作阈值，算 P/R/F1 ----
    res_pr = model.val(data=str(runtime_yaml), split=args.split, imgsz=args.imgsz,
                       batch=args.batch, conf=args.conf, iou=args.iou, device=args.device,
                       project=str(out_dir), name="pr", plots=False, verbose=False)
    b2 = res_pr.box
    p, r = float(b2.mp), float(b2.mr)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    speed = getattr(res_pr, "speed", {}) or {}
    pre, inf, post = (float(speed.get(k, 0)) for k in ("preprocess", "inference", "postprocess"))
    total_ms = pre + inf + post
    fps_infer = 1000.0 / inf if inf > 0 else 0.0
    fps_e2e = 1000.0 / total_ms if total_ms > 0 else 0.0

    hw = C.env_report()
    gpu = ", ".join(hw.get("gpu", [])) if hw.get("gpu") else "CPU"

    # ---- 分类别 ----
    per_class = []
    try:
        idx = list(b.ap_class_index)
        for i, ci in enumerate(idx):
            cp, cr = float(b.p[i]), float(b.r[i])
            per_class.append({
                "class": names.get(int(ci), str(ci)),
                "precision": cp, "recall": cr,
                "f1": 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0.0,
                "ap50": float(b.ap50[i]), "ap": float(b.ap[i]),
            })
    except Exception as e:
        C.warn(f"分类别指标解析失败：{e}")

    # ---- 输出 ----
    md = [
        "# 测试集评估结果", "",
        f"- 时间：{C.now_str()}",
        f"- 权重：`{weights}`",
        f"- 测试集：`{C.IMG_DIR / args.split}`（{len(C.iter_images(C.IMG_DIR / args.split))} 张）",
        f"- 硬件：{gpu}；batch={args.batch}；输入分辨率={args.imgsz}；NMS IoU={args.iou}",
        f"- 加速：{'CUDA ' + str(hw.get('cuda')) if hw.get('cuda_available') else 'CPU'}",
        "",
        "## 总体指标", "",
        "| 指标 | 数值 | 说明 |", "|---|---|---|",
        f"| mAP@0.5 | **{fmt(map50)}** | 置信度阈值 0.001，衡量整体检测质量 |",
        f"| mAP@0.5:0.95 | **{fmt(map5095)}** | 更严格，看框贴合程度 |",
        f"| Precision | **{fmt(p)}** | 置信度扫描中 F1 最大处的取值 |",
        f"| Recall | **{fmt(r)}** | 同上 |",
        f"| F1 | **{fmt(f1)}** | 2PR/(P+R) |",
        f"| FPS（仅推理） | {fps_infer:.1f} | 不含前后处理 |",
        f"| FPS（端到端） | {fps_e2e:.1f} | 含预处理+NMS，实际部署看这个 |",
        f"| 单帧耗时 | {total_ms:.1f} ms | 预处理 {pre:.1f} + 推理 {inf:.1f} + NMS {post:.1f} |",
        "",
    ]
    if per_class:
        md += ["## 分类别指标（阈值 0.001 下的 AP）", "",
               "| 类别 | Precision | Recall | F1 | AP@0.5 | AP@0.5:0.95 |",
               "|---|---|---|---|---|---|"]
        for d in per_class:
            md.append(f"| {d['class']} | {fmt(d['precision'])} | {fmt(d['recall'])} | "
                      f"{fmt(d['f1'])} | {fmt(d['ap50'])} | {fmt(d['ap'])} |")
        md.append("")
    md += [
        "## 口径说明（评委一定会问）", "",
        f"1. 所有指标在**同一测试集**、**同一 IoU 阈值({args.iou})**、**同一输入分辨率({args.imgsz})**下取得。",
        "2. 测试集与训练/验证集**按场景隔离**，没有同一段视频的帧跨集。",
        f"3. 速度在 {gpu} 上以 batch={args.batch} 测得，含/不含后处理已分别列出。",
        f"4. P/R/F1 对应的工作阈值 conf={args.conf}，mAP 用 0.001，二者不可混用。",
    ]
    md_path = out_dir / "test_metrics.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    C.save_json(out_dir / "test_metrics.json", {
        "weights": str(weights), "split": args.split, "imgsz": args.imgsz,
        "batch": args.batch, "conf": args.conf, "iou": args.iou,
        "map50": map50, "map50_95": map5095, "precision": p, "recall": r, "f1": f1,
        "speed_ms": {"preprocess": pre, "inference": inf, "postprocess": post, "total": total_ms},
        "fps_inference_only": fps_infer, "fps_end_to_end": fps_e2e,
        "per_class": per_class, "hardware": gpu, "env": hw, "time": C.now_str(),
    })

    C.banner("结果")
    C.info(f"mAP@0.5 = {map50:.4f}   mAP@0.5:0.95 = {map5095:.4f}")
    C.info(f"P = {p:.4f}   R = {r:.4f}   F1 = {f1:.4f}   (conf={args.conf})")
    C.info(f"FPS: 推理 {fps_infer:.1f} / 端到端 {fps_e2e:.1f}   单帧 {total_ms:.1f} ms")
    C.info(f"\n报告：{md_path}")
    C.info("把 test_metrics.md 里的表格直接贴进设计报告。")


if __name__ == "__main__":
    main()
