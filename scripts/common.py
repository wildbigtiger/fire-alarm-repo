# -*- coding: utf-8 -*-
"""公共工具。

所有脚本的路径都从这里取，保证目录约定只有一份定义。
在 ModelArts 上数据不在项目目录里时，用环境变量 FIRE_DATA_ROOT 覆盖数据集根目录。

    export FIRE_DATA_ROOT=/cache/data/fire
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------------
# 路径约定
# ----------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent

# 数据集根目录：默认 <项目>/datasets/fire，可用环境变量覆盖
DATASET_ROOT = Path(os.environ.get("FIRE_DATA_ROOT") or (PROJECT_ROOT / "datasets" / "fire")).resolve()

RAW_DIR = DATASET_ROOT / "raw"
RAW_IMAGES = RAW_DIR / "images"          # raw/images/<场景名>/*.jpg   原始素材，按场景分目录
RAW_VIDEOS = RAW_DIR / "videos"          # raw/videos/<场景名>/*.mp4
RAW_CLIPS = RAW_DIR / "clips"            # raw/clips/<场景名>/*.mp4    切好的短片段（用于凑"300 段"）

ANN_DIR = DATASET_ROOT / "ann"           # 标注与追溯信息
ANN_LABELS = ANN_DIR / "labels"          # ann/labels/<场景名>/<文件名>.txt  YOLO 格式
ANN_RAW = ANN_DIR / "raw"                # ann/raw/<场景名>/<文件名>.json    大模型原始输出
ANN_MASKS = ANN_DIR / "masks"            # 分割掩码（子任务 4 用）

IMG_DIR = DATASET_ROOT / "images"        # images/<split>/  划分后的图（train/val/test）
LBL_DIR = DATASET_ROOT / "labels"        # labels/<split>/  与 images 一一对应
META_DIR = DATASET_ROOT / "meta"         # classes.txt / split.json / 统计表

CONFIG_DIR = PROJECT_ROOT / "configs"
RUNS_DIR = PROJECT_ROOT / "runs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv"}

SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------
# 控制台
# ----------------------------------------------------------------------------
def _fix_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


_fix_stdout()


def info(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def die(msg: str) -> "None":
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def banner(msg: str) -> None:
    line = "=" * 66
    info(f"\n{line}\n{msg}\n{line}")


# ----------------------------------------------------------------------------
# 文件
# ----------------------------------------------------------------------------
def iter_images(root: Path) -> list[Path]:
    """递归列出所有图片，排序后返回（保证可复现）。"""
    if not Path(root).exists():
        return []
    files = [p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    return sorted(files, key=lambda p: str(p).lower())


def iter_videos(root: Path) -> list[Path]:
    if not Path(root).exists():
        return []
    files = [p for p in Path(root).rglob("*") if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    return sorted(files, key=lambda p: str(p).lower())


def scene_of(img_path: Path, base: Path = RAW_IMAGES) -> str:
    """取图片所属场景名 = raw/images/<场景名>/ 里那一层目录。

    约定：一个场景 = 一次拍摄（同一房间/同一段视频/同一台设备同一时段）。
    划分数据集时整场景一起走，绝不拆开，这是防止数据泄漏的关键。
    """
    try:
        rel = Path(img_path).resolve().relative_to(Path(base).resolve())
    except ValueError:
        return "default"
    return rel.parts[0] if len(rel.parts) > 1 else "default"


def ensure_dir(path: Path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


# ----------------------------------------------------------------------------
# YOLO 标签读写
# ----------------------------------------------------------------------------
def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    """读 YOLO txt，返回 [(cls, cx, cy, w, h), ...]。空文件 = 负样本（无目标）。"""
    rows: list[tuple[int, float, float, float, float]] = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            rows.append((int(float(parts[0])), *(float(v) for v in parts[1:])))
        except ValueError:
            continue
    return rows


def write_label(path: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    text = "".join(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rows)
    p.write_text(text, encoding="utf-8")


def image_size(path: Path) -> tuple[int, int]:
    """(width, height)，只用 Pillow，不依赖 opencv。"""
    from PIL import Image

    with Image.open(path) as im:
        return im.size


# ----------------------------------------------------------------------------
# JSON / 杂项
# ----------------------------------------------------------------------------
def save_json(path: Path, obj) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def env_report() -> dict:
    """记录训练环境，写进报告用的"可复现"章节。"""
    rep = {
        "time": now_str(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "machine": platform.machine(),
    }
    try:
        import torch

        rep["torch"] = torch.__version__
        rep["cuda"] = torch.version.cuda
        rep["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            rep["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        rep["torch"] = "not installed"
    try:
        import ultralytics

        rep["ultralytics"] = ultralytics.__version__
    except Exception:
        rep["ultralytics"] = "not installed"
    try:
        import transformers

        rep["transformers"] = transformers.__version__
    except Exception:
        pass
    return rep


def load_classes() -> list[str]:
    """从 meta/classes.txt 读类别名，顺序即类别 id。"""
    p = META_DIR / "classes.txt"
    if not p.exists():
        die(f"找不到 {p}，请先运行 python scripts/make_dirs.py")
    names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not names:
        die(f"{p} 是空的")
    return names


def prepare_data_yaml(src: Path, dst: Path, dataset_root: Path = DATASET_ROOT) -> Path:
    """把 configs/fire.yaml 里的 path 换成绝对路径，避免 ultralytics 按 cwd 解析出错。

    没装 pyyaml 时退化为逐行替换，这样 --dry-run 这类操作在裸环境也能跑。
    """
    text = Path(src).read_text(encoding="utf-8")
    abs_path = str(Path(dataset_root).resolve())
    try:
        import yaml

        cfg = yaml.safe_load(text) or {}
        cfg["path"] = abs_path
        out = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    except ImportError:
        import re

        out, done = [], False
        for line in text.splitlines():
            if re.match(r"^\s*path\s*:", line):
                out.append(f"path: {abs_path}")
                done = True
            else:
                out.append(line)
        if not done:
            out.insert(0, f"path: {abs_path}")
        out = "\n".join(out) + "\n"
    ensure_dir(Path(dst).parent)
    Path(dst).write_text(out, encoding="utf-8")
    return Path(dst)
