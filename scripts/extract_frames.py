# -*- coding: utf-8 -*-
"""从视频抽帧，自动去重（相邻帧几乎一样，全留下只会让数据虚胖）。

    python scripts/extract_frames.py --fps 2
    python scripts/extract_frames.py --fps 2 --max-per-video 300 --dedup 6

视频放在 datasets/fire/raw/videos/<场景名>/ 下，抽出来的图会自动进
datasets/fire/raw/images/<场景名>/，场景名保持不变——这是后面按场景划分的基础。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C


def ahash(gray) -> int:
    """8x8 平均哈希，用来判断两帧是否几乎相同。"""
    import cv2

    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    bits = 0
    for v in small.flatten():
        bits = (bits << 1) | (1 if v > mean else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> None:
    ap = argparse.ArgumentParser(description="从视频抽帧")
    ap.add_argument("--video", default="", help="只处理单个视频；默认处理 raw/videos 下全部")
    ap.add_argument("--fps", type=float, default=2.0, help="每秒抽几帧")
    ap.add_argument("--max-per-video", type=int, default=0, help="每个视频最多抽多少张")
    ap.add_argument("--dedup", type=int, default=5,
                    help="哈希差异小于该值视为重复并跳过（0 表示不去重，建议 4~8）")
    ap.add_argument("--quality", type=int, default=90)
    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        C.die("需要 opencv：pip install opencv-python")

    videos = [Path(args.video)] if args.video else C.iter_videos(C.RAW_VIDEOS)
    if not videos:
        C.die(f"{C.RAW_VIDEOS} 下没有视频。先放视频进去。")

    C.banner(f"抽帧：{len(videos)} 个视频，目标 {args.fps} fps，去重阈值 {args.dedup}")
    total_saved = total_frames = 0

    for v in videos:
        if not v.exists():
            C.warn(f"跳过不存在的视频：{v}")
            continue
        if v.parent == C.RAW_VIDEOS:
            scene = "scene_未分类"
        else:
            scene = v.parent.name
        out_dir = C.ensure_dir(C.RAW_IMAGES / scene)

        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            C.warn(f"打不开：{v.name}")
            continue
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(src_fps / args.fps)))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        idx = saved = 0
        last_hash = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h = ahash(gray)
                dup = last_hash is not None and hamming(h, last_hash) < args.dedup
                if not dup:
                    name = f"{v.stem}_f{idx:06d}.jpg"
                    cv2.imwrite(str(out_dir / name), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                    saved += 1
                    last_hash = h
                    if args.max_per_video and saved >= args.max_per_video:
                        break
            idx += 1
        cap.release()

        total_saved += saved
        total_frames += n_frames
        C.info(f"  {scene}/{v.name}: {n_frames} 帧 @ {src_fps:.1f}fps -> 抽存 {saved} 张")

    C.banner("完成")
    C.info(f"共保存 {total_saved} 张到 {C.RAW_IMAGES}")
    C.info("下一步：python scripts/autolabel.py  或在没 GPU 时先 --mock 试流程")


if __name__ == "__main__":
    main()
