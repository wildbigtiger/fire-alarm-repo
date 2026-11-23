# -*- coding: utf-8 -*-
"""ModelArts 训练作业入口。

本地直接跑 scripts/train_yolo.py 就行；这个文件是给 ModelArts 的"训练作业"
（不是 Notebook）用的，它会：
    1. 从 OBS 把数据同步到本地缓存盘
    2. 调用 scripts/train_yolo.py 训练
    3. 把 runs/ 结果再传回 OBS

ModelArts 训练作业的启动命令填：
    python train_entry.py --obs-data obs://你的桶/fire-alarm/data/fire \
        --obs-output obs://你的桶/fire-alarm/output --epochs 100

如果是在 Notebook 里跑，不需要这个文件，直接用 scripts/ 下的脚本即可。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sync(src: str, dst: str, what: str) -> None:
    """用 moxing 同步 OBS 与本地。模型训练作业里才装了 moxing。"""
    if not src:
        return
    try:
        import moxing as mox
    except ImportError:
        print(f"[WARN] 没有 moxing，跳过 {what} 同步（{src} -> {dst}）")
        return
    print(f"[INFO] {what}：{src} -> {dst}", flush=True)
    mox.file.make_dirs(dst)
    mox.file.copy_parallel(src, dst)
    print(f"[INFO] {what}完成", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="ModelArts 训练作业入口")
    ap.add_argument("--data-root", default="/cache/data/fire",
                    help="本地数据根目录（对应 FIRE_DATA_ROOT）；/cache 是 ModelArts 的本地盘，最快")
    ap.add_argument("--obs-data", default="", help="数据在 OBS 上的位置，给了就先同步下来")
    ap.add_argument("--obs-output", default="", help="训练结果传回 OBS 的位置")
    ap.add_argument("--data", default=str(ROOT / "configs" / "fire.yaml"))
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--preset", default="baseline",
                    choices=["baseline", "small_obj", "augment"])
    ap.add_argument("--name", default="", help="实验名，留空自动生成")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 数据同步
    if args.obs_data:
        sync(args.obs_data, args.data_root, "数据同步")

    if not Path(args.data_root, "images", "train").exists():
        print(f"[ERROR] {args.data_root}/images/train 不存在。"
              f"请用 --obs-data 指向 OBS 数据集，或把数据放到这个目录。", file=sys.stderr)
        sys.exit(1)

    os.environ["FIRE_DATA_ROOT"] = args.data_root
    os.chdir(ROOT)  # 让相对路径（runs/、configs/）都按项目根解析

    sys.path.insert(0, str(ROOT / "scripts"))
    import train_yolo  # noqa: E402  （必须在设置环境变量之后导入）

    argv = [
        "--data", args.data,
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--device", args.device,
        "--workers", str(args.workers),
        "--preset", args.preset,
        "--seed", str(args.seed),
        "--exist-ok",
    ]
    if args.name:
        argv += ["--name", args.name]

    print(f"[INFO] 开始训练，参数：{' '.join(argv)}", flush=True)
    train_yolo.main(argv)

    # 结果回传
    if args.obs_output:
        sync(str(ROOT / "runs"), args.obs_output, "结果上传")
        print(f"[INFO] 训练结果已上传到 {args.obs_output}", flush=True)


if __name__ == "__main__":
    main()
