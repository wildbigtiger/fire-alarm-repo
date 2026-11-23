# -*- coding: utf-8 -*-
"""训练 YOLO 火灾检测模型。

    # 基线（第一次就跑这个）
    python scripts/train_yolo.py --epochs 100

    # 优化策略之一：小目标增强（题目要求"至少一项优化策略 + 前后对比"）
    python scripts/train_yolo.py --preset small_obj --epochs 100 --name opt_small

    # 先看配置对不对，不真的训练
    python scripts/train_yolo.py --dry-run

训练产物在 runs/detect/<name>/：
    weights/best.pt        最好的权重
    weights/last.pt        最后一轮
    results.png            损失曲线 + mAP 曲线（报告直接用这个）
    results.csv            每个 epoch 的数值
    args.yaml              本次全部超参（可复现）
另外会写 runs/detect/<name>/repro.json，记录显卡、torch 版本、数据指纹。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C

# 优化策略预设。每个预设都对应报告里要解释的一句话。
PRESETS: dict[str, dict] = {
    "baseline": {},
    # 小目标 + 明火/烟雾混淆：提高分辨率、加强分类损失、延长关闭 mosaic 的轮数
    "small_obj": {
        "imgsz": 960,
        "mosaic": 1.0,
        "scale": 0.9,
        "cls": 0.7,
        "close_mosaic": 15,
        "copy_paste": 0.1,
    },
    # 数据增强型：应对光照、视角、遮挡差异
    "augment": {
        "mosaic": 1.0,
        "mixup": 0.15,
        "copy_paste": 0.3,
        "degrees": 5.0,
        "translate": 0.15,
        "scale": 0.6,
        "hsv_h": 0.02,
        "hsv_s": 0.8,
        "hsv_v": 0.5,
        "flipud": 0.1,
        "close_mosaic": 15,
    },
}


def dataset_fingerprint() -> dict:
    """给数据集打个指纹，确保报告里的实验结果能对上哪一版数据。"""
    import hashlib

    h = hashlib.sha256()
    n_img = n_box = 0
    for split in C.SPLITS:
        imgs = C.iter_images(C.IMG_DIR / split)
        n_img += len(imgs)
        for p in imgs:
            h.update(p.name.encode())
            rows = C.read_label(C.LBL_DIR / split / f"{p.stem}.txt")
            n_box += len(rows)
            for r in sorted(rows):
                h.update(f"{r[0]}:{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}".encode())
    split_json = C.META_DIR / "split.json"
    if split_json.exists():
        h.update(split_json.read_bytes())
    return {"images": n_img, "boxes": n_box, "sha256": h.hexdigest()[:16]}


def build_config(args) -> dict:
    """把命令行参数组装成 ultralytics 的 train 参数。"""
    cfg: dict = {
        "data": None,  # 下面填
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(C.RUNS_DIR / "detect"),
        "name": args.name,
        "exist_ok": args.exist_ok,
        "patience": args.patience,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "cos_lr": args.cos_lr,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "amp": args.amp,
        "cache": args.cache,
        "plots": True,
        "val": True,
        "save_period": args.save_period,
        "pretrained": True,
    }
    # 预设里的项，命令行没显式指定的才用
    for k, v in PRESETS[args.preset].items():
        cfg[k] = v
    return cfg


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="训练 YOLO 火灾检测模型")
    ap.add_argument("--data", default=str(C.CONFIG_DIR / "fire.yaml"))
    ap.add_argument("--model", default="yolo11n.pt",
                    help="预训练权重：yolo11n/s/m/l.pt 或 yolov8n.pt。先用 n 跑通再换大的")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="0 / 0,1 / cpu；ModelArts 单卡填 0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--optimizer", default="auto", help="auto / SGD / AdamW")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.937)
    ap.add_argument("--weight-decay", type=float, default=0.0005)
    ap.add_argument("--warmup-epochs", type=float, default=3.0)
    ap.add_argument("--cos-lr", action="store_true")
    ap.add_argument("--patience", type=int, default=50, help="多少轮没提升就早停")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true", help="更严格复现（略慢）")
    ap.add_argument("--amp", dest="amp", action="store_true", default=True, help="混合精度（默认开）")
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="关掉混合精度，排查 NaN 时用")
    ap.add_argument("--cache", default="False", help="False / ram / disk")
    ap.add_argument("--save-period", type=int, default=-1)
    ap.add_argument("--preset", choices=list(PRESETS), default="baseline")
    ap.add_argument("--name", default="", help="实验名，默认用 preset_时间戳")
    ap.add_argument("--exist-ok", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只打印配置，不训练")
    args = ap.parse_args(argv)

    C.seed_everything(args.seed)

    # 数据检查
    missing = [s for s in C.SPLITS if not C.iter_images(C.IMG_DIR / s)]
    if missing:
        C.die(f"这些子集没有图片：{missing}。先跑 split_dataset.py")

    if not args.name:
        args.name = f"{args.preset}_{args.model.split('.')[0]}_{C.stamp()}"
    cache_val = args.cache if args.cache in ("ram", "disk") else False

    runtime_yaml = C.prepare_data_yaml(Path(args.data), C.RUNS_DIR / f"_data_{args.name}.yaml")

    cfg = build_config(args)
    cfg["data"] = str(runtime_yaml)
    cfg["cache"] = cache_val

    fp = dataset_fingerprint()
    C.banner(f"训练配置（{args.preset}）")
    for k, v in cfg.items():
        C.info(f"  {k:16} = {v}")
    C.info(f"  数据集指纹       = {fp}")
    C.info(f"  环境             = {json.dumps(C.env_report(), ensure_ascii=False)}")

    if args.dry_run:
        C.info("\n--dry-run，不实际训练。去掉这个参数就开始。")
        return

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(**cfg)

    save_dir = Path(getattr(results, "save_dir", "") or (C.RUNS_DIR / "detect" / args.name))
    best = save_dir / "weights" / "best.pt"
    C.save_json(
        save_dir / "repro.json",
        {
            "args": vars(args),
            "resolved_config": cfg,
            "env": C.env_report(),
            "dataset": fp,
            "time": C.now_str(),
        },
    )

    C.banner("训练结束")
    C.info(f"权重：{best}")
    C.info(f"曲线图：{save_dir / 'results.png'}  <- 报告里贴这张")
    C.info(f"原始指标：{save_dir / 'results.csv'}")
    C.info(f"可复现记录：{save_dir / 'repro.json'}")
    C.info(f"\n下一步评估测试集：\n  python scripts/eval_yolo.py --weights \"{best}\" --name {args.name}")


if __name__ == "__main__":
    main()
