# -*- coding: utf-8 -*-
"""推理演示：给视频画框、算火灾面积占比、判报警等级。

两个用途：
1) 命令行看效果：
       python scripts/demo_infer.py --weights runs/detect/xxx/weights/best.pt --source test.mp4
2) 给做界面的同学当轮子用（子任务 5 直接 import）：
       from demo_infer import FireDetector
       det = FireDetector("best.pt")
       out = det.infer(frame)          # frame 是 cv2 的 BGR 图
       out["alarm"]["level"]           # 0 正常 / 1 关注 / 2 预警 / 3 报警

注意：这里画的是英文标签（fire 0.87），因为 cv2 写中文要额外挂字体文件。
界面里显示中文没问题，那是 Qt/Gradio 的字体，不是 cv2 的。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C

# 报警等级阈值（示例值，报告里要说明依据，现场演示前按实际情况微调）
RATIO_WARN = 0.05      # 火焰面积占比 >=5% 算预警
RATIO_ALARM = 0.15     # >=15% 算报警
RATIO_ALARM_HARD = 0.30  # >=30% 直接最高级，不等持续时间
SUSTAIN_SEC = 2.0      # 需要持续多少秒才升级

COLORS = {0: (60, 60, 220), 1: (200, 200, 200)}  # fire=红, smoke=灰
LEVEL_TEXT = {0: "NORMAL", 1: "ATTENTION", 2: "WARNING", 3: "ALARM"}
LEVEL_COLOR = {0: (80, 200, 80), 1: (0, 200, 255), 2: (0, 140, 255), 3: (0, 0, 255)}


class FireDetector:
    def __init__(self, weights: str, imgsz: int = 640, device: str = "0", alarm_conf: float = 0.5):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.device = device
        self.alarm_conf = alarm_conf  # 只有这个置信度以上的框才计入报警面积
        self.names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        self.high_fire_since: float | None = None  # 连续高风险的起点时间

    def infer(self, frame, conf: float = 0.25, iou: float = 0.5, now: float | None = None) -> dict:
        """返回检测框、面积占比、报警等级。frame 为 BGR ndarray。

        报警用的面积只统计置信度 >= alarm_conf 的框：低置信度的框照样画出来给人看，
        但不应把报警等级抬高（实测低分大框会把 5% 虚报到 16%）。
        """
        import time

        now = time.time() if now is None else now
        h, w = frame.shape[:2]
        res = self.model.predict(frame, imgsz=self.imgsz, conf=conf, iou=iou,
                                 device=self.device, verbose=False)[0]

        dets = []
        fire_area = smoke_area = 0.0
        for bx in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in bx.xyxy[0].tolist())
            cls_id = int(bx.cls.item())
            score = float(bx.conf.item())
            name = self.names.get(cls_id, str(cls_id))
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (w * h)
            if score >= self.alarm_conf:
                if name == "fire":
                    fire_area += area
                elif name == "smoke":
                    smoke_area += area
            dets.append({"box": [x1, y1, x2, y2], "cls": name, "cls_id": cls_id,
                         "conf": score, "area_ratio": area})

        fire_area = min(fire_area, 1.0)
        smoke_area = min(smoke_area, 1.0)
        level = self._alarm(fire_area, dets, now)
        return {
            "detections": dets,
            "counts": {"fire": sum(1 for d in dets if d["cls"] == "fire"),
                       "smoke": sum(1 for d in dets if d["cls"] == "smoke")},
            "fire_area_ratio": fire_area,
            "smoke_area_ratio": smoke_area,
            "alarm": {"level": level, "text": LEVEL_TEXT[level]},
        }

    def _alarm(self, fire_ratio: float, dets: list[dict], now: float) -> int:
        if fire_ratio >= RATIO_ALARM_HARD:
            return 3
        hot = fire_ratio >= RATIO_WARN
        if hot:
            if self.high_fire_since is None:
                self.high_fire_since = now
            dur = now - self.high_fire_since
        else:
            self.high_fire_since = None
            dur = 0.0
        if fire_ratio >= RATIO_ALARM and dur >= SUSTAIN_SEC:
            return 3
        if hot and dur >= SUSTAIN_SEC:
            return 2
        if any(d["cls"] == "fire" and d["conf"] >= 0.5 for d in dets):
            return 1
        return 0


def draw(frame, result: dict):
    import cv2

    for d in result["detections"]:
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        color = COLORS.get(d["cls_id"], (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{d['cls']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    lv = result["alarm"]["level"]
    txt = f"{result['alarm']['text']}  fire={result['fire_area_ratio']*100:.1f}%"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (30, 30, 30), -1)
    cv2.putText(frame, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, LEVEL_COLOR[lv], 2)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser(description="视频推理演示")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="视频文件 / 图片 / 目录")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=str(C.RUNS_DIR / "demo"))
    ap.add_argument("--max-frames", type=int, default=0, help="只处理前 N 帧，调试用")
    args = ap.parse_args()

    import cv2

    out_dir = C.ensure_dir(Path(args.out))
    det = FireDetector(args.weights, args.imgsz, args.device)

    src = Path(args.source)
    if not src.exists():
        C.die(f"源文件不存在：{src}")

    if src.suffix.lower() in C.IMAGE_EXTS:
        frame = cv2.imread(str(src))
        r = det.infer(frame, args.conf, args.iou)
        drawn = draw(frame, r)
        outp = out_dir / f"{src.stem}_pred.jpg"
        cv2.imwrite(str(outp), drawn)
        C.info(f"检测到 {r['counts']}，火焰占比 {r['fire_area_ratio']*100:.1f}%，"
               f"报警等级 {r['alarm']['text']}")
        C.info(f"输出：{outp}")
        return

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        C.die(f"打不开视频：{src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    outp = out_dir / f"{src.stem}_pred.mp4"
    vw = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    C.banner(f"推理 {src.name}（{w}x{h}, {fps:.1f}fps, {total} 帧）")
    rows = []
    i = 0
    t_start = cv2.getTickCount()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        r = det.infer(frame, args.conf, args.iou)
        vw.write(draw(frame, r))
        rows.append([i, round(i / fps, 3), r["counts"]["fire"], r["counts"]["smoke"],
                     round(r["fire_area_ratio"], 4), round(r["smoke_area_ratio"], 4),
                     r["alarm"]["level"]])
        if i % 50 == 0:
            C.info(f"  {i}/{total if total else '?'}")
        if args.max_frames and i >= args.max_frames:
            break
    cap.release()
    vw.release()
    elapsed = (cv2.getTickCount() - t_start) / cv2.getTickFrequency()
    proc_fps = i / elapsed if elapsed > 0 else 0.0

    csv_path = out_dir / f"{src.stem}_frames.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        wtr = csv.writer(f)
        wtr.writerow(["frame", "time_s", "n_fire", "n_smoke", "fire_ratio", "smoke_ratio", "alarm_level"])
        wtr.writerows(rows)

    max_ratio = max((r[4] for r in rows), default=0.0)
    n_alarm = sum(1 for r in rows if r[6] == 3)
    C.banner("完成")
    C.info(f"处理 {i} 帧，耗时 {elapsed:.1f}s，吞吐 {proc_fps:.1f} FPS")
    C.info(f"最大火焰占比 {max_ratio*100:.1f}%，报警帧 {n_alarm} 帧")
    C.info(f"标注视频：{outp}")
    C.info(f"逐帧数据：{csv_path}   <- 做趋势曲线可以直接读这个")


if __name__ == "__main__":
    main()
