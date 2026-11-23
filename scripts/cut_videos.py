# -*- coding: utf-8 -*-
"""把长视频切成短片段，用来满足题目"不少于 300 段视频"的数量要求。

    python scripts/cut_videos.py --clip-len 4

原始素材 -> datasets/fire/raw/videos/<场景>/*.mp4
短片段 -> datasets/fire/raw/clips/<场景>/<原名>_part000.mp4

为什么要切：题目要 "≥300 段、总时长≥10 分钟"。你录 12 分钟素材，
切成 300 多个 3~4 秒的小片段就同时满足了两个条件，不用真去拍 300 次。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C


def main() -> None:
    ap = argparse.ArgumentParser(description="切分视频片段")
    ap.add_argument("--clip-len", type=float, default=4.0, help="每段几秒")
    ap.add_argument("--min-sec", type=float, default=1.5, help="太短的尾巴丢掉")
    ap.add_argument("--source", default="", help="只处理单个视频")
    args = ap.parse_args()

    try:
        import cv2
    except ImportError:
        C.die("需要 opencv：pip install opencv-python")

    videos = [Path(args.source)] if args.source else C.iter_videos(C.RAW_VIDEOS)
    if not videos:
        C.die(f"{C.RAW_VIDEOS} 下没有视频")

    C.banner(f"切片段：{len(videos)} 个视频，每段 {args.clip_len}s")
    total_clips = 0
    total_sec = 0.0

    for v in videos:
        scene = v.parent.name if v.parent != C.RAW_VIDEOS else "scene_未分类"
        out_dir = C.ensure_dir(C.RAW_CLIPS / scene)

        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            C.warn(f"打不开：{v.name}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        per_clip = max(1, int(round(fps * args.clip_len)))

        idx = part = 0
        writer = None
        in_clip = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if in_clip == 0:
                if writer is not None:
                    writer.release()
                path = out_dir / f"{v.stem}_part{part:03d}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                part += 1
            writer.write(frame)
            in_clip += 1
            idx += 1
            if in_clip >= per_clip:
                in_clip = 0
        if writer is not None:
            writer.release()
        cap.release()

        kept = part
        if in_clip > 0 and in_clip / fps < args.min_sec:
            kept = max(0, part - 1)  # 最后一段太短，不计
        total_clips += kept
        total_sec += idx / fps
        C.info(f"  {scene}/{v.name}: {idx/fps:.1f}s -> {kept} 段")

    C.banner("完成")
    C.info(f"共 {total_clips} 段，原始总时长 {total_sec/60:.1f} 分钟")
    if total_clips < 300:
        C.warn(f"距离题目要求的 300 段还差 {300 - total_clips} 段，可以调小 --clip-len 再切一遍")
    if total_sec < 600:
        C.warn(f"总时长 {total_sec/60:.1f} 分钟，题目要求不少于 10 分钟")
    C.info(f"片段目录：{C.RAW_CLIPS}")


if __name__ == "__main__":
    main()
