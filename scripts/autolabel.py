# -*- coding: utf-8 -*-
"""用开放域大模型（GroundingDINO）给图片自动预标注，产出 YOLO 格式标签。

产出两份东西：
  ann/labels/<场景>/<文件名>.txt    YOLO 标签，人工就地修改的就是这一份
  ann/raw/<场景>/<文件名>.json      大模型原始输出 + 提示词 + 阈值，用于"可追溯"

用法：
    # 真实标注（需要 GPU + 首次自动下载权重约 700MB）
    python scripts/autolabel.py --model IDEA-Research/grounding-dino-tiny

    # 没有 GPU 先验证流程（生成假框，不需要任何模型）
    python scripts/autolabel.py --mock

改提示词：直接改下面的 PROMPT_GROUPS，并在报告里说明理由。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C

# ---------------------------------------------------------------------------
# 提示词：类别 -> 一组同义说法
# GroundingDINO 的文本输入是"用句号分隔的名词短语"，它会把每个短语当一类。
# 多给几种说法能提高召回（比如 blaze / burning 对应不同形态的火）。
# ---------------------------------------------------------------------------
PROMPT_GROUPS: dict[str, list[str]] = {
    "fire": ["fire", "flame", "burning", "blaze"],
    "smoke": ["smoke", "smoke plume", "heavy smoke"],
}

DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"


def build_prompt(groups: dict[str, list[str]]) -> str:
    phrases: list[str] = []
    for words in groups.values():
        phrases.extend(words)
    return ". ".join(phrases) + "."


def phrase_to_class(phrase: str, groups: dict[str, list[str]]) -> str | None:
    """把大模型吐出来的短语归到 fire / smoke 两类里。

    先长后短匹配，避免 "smoke plume" 被更短的 "smoke" 抢先（结果一样，但便于调试）。
    """
    p = phrase.lower().strip()
    for cls, words in groups.items():
        for w in sorted(words, key=len, reverse=True):
            if w in p:
                return cls
    return None


# ---------------------------------------------------------------------------
# 假标注：没有 GPU 时用来把上下游流程跑通
# ---------------------------------------------------------------------------
def mock_detect(img_path: Path, scene: str, classes: list[str]) -> list[dict]:
    """根据文件名哈希生成稳定的假框；场景名里带 neg 的当作负样本（无目标）。"""
    if "neg" in scene.lower():
        return []
    h = int(hashlib.md5(img_path.stem.encode()).hexdigest(), 16)
    n = 1 + h % 3
    dets = []
    for i in range(n):
        seed = (h >> (i * 8)) & 0xFFFF
        bw = 0.10 + (seed % 300) / 1000.0
        bh = 0.10 + ((seed >> 3) % 300) / 1000.0
        cx = 0.15 + ((seed >> 6) % 700) / 1000.0
        cy = 0.15 + ((seed >> 9) % 700) / 1000.0
        cx = min(max(cx, bw / 2), 1 - bw / 2)
        cy = min(max(cy, bh / 2), 1 - bh / 2)
        cls = classes[0] if i == 0 else classes[min(1, len(classes) - 1)]
        dets.append(
            {
                "cls": cls,
                "cx": cx, "cy": cy, "w": bw, "h": bh,
                "score": round(0.55 + ((seed >> 12) % 40) / 100.0, 4),
                "phrase": cls,
            }
        )
    return dets


# ---------------------------------------------------------------------------
# 真实检测
# ---------------------------------------------------------------------------
class GroundingDinoAnnotator:
    def __init__(self, model_id: str, device: str, box_thresh: float, text_thresh: float):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device
        C.info(f"加载模型 {model_id} -> {device}（首次运行会下载权重，请耐心等）")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        self.box_thresh = box_thresh
        self.text_thresh = text_thresh

    def __call__(self, img_path: Path, prompt: str) -> list[dict]:
        from PIL import Image

        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        # transformers 不同版本参数名有差异，做一次兼容
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_thresh,
                text_threshold=self.text_thresh,
                target_sizes=[image.size[::-1]],
            )[0]
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_thresh,
                text_threshold=self.text_thresh,
                target_sizes=[image.size[::-1]],
            )[0]

        phrases = results.get("text_labels", results.get("labels")) or []
        boxes = results["boxes"].tolist()
        scores = results["scores"].tolist()
        W, H = image.size

        dets = []
        for box, score, phrase in zip(boxes, scores, phrases):
            x1, y1, x2, y2 = box
            x1, x2 = max(0.0, x1), min(float(W), x2)
            y1, y2 = max(0.0, y1), min(float(H), y2)
            bw, bh = x2 - x1, y2 - y1
            if bw <= 1 or bh <= 1:
                continue
            dets.append(
                {
                    "cls": None,
                    "phrase": str(phrase),
                    "score": round(float(score), 4),
                    "box_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "cx": (x1 + x2) / 2 / W,
                    "cy": (y1 + y2) / 2 / H,
                    "w": bw / W,
                    "h": bh / H,
                }
            )
        return dets


def nms(dets: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """按类别做 NMS，压掉重复框。用 numpy 手写，不额外依赖 torchvision。"""
    import numpy as np

    out: list[dict] = []
    for cls in {d["cls"] for d in dets if d["cls"]}:
        group = sorted([d for d in dets if d["cls"] == cls], key=lambda d: -d["score"])
        while group:
            best = group.pop(0)
            out.append(best)
            keep = []
            for d in group:
                b1, b2 = best["box_xyxy"], d["box_xyxy"]
                ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih
                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou = inter / (a1 + a2 - inter + 1e-9)
                if iou < iou_thresh:
                    keep.append(d)
            group = keep
    return out


def to_yolo_rows(dets: list[dict], classes: list[str]) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for d in dets:
        if d.get("cls") not in classes:
            continue
        rows.append((classes.index(d["cls"]), d["cx"], d["cy"], d["w"], d["h"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="GroundingDINO 自动预标注")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace 模型名")
    ap.add_argument("--mock", action="store_true", help="假标注，不需要模型和 GPU")
    ap.add_argument("--box-thresh", type=float, default=0.30)
    ap.add_argument("--text-thresh", type=float, default=0.25)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--device", default=None, help="cuda / cpu，默认自动")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张，调试用")
    ap.add_argument("--force", action="store_true", help="已标注过的也重标")
    args = ap.parse_args()

    classes = C.load_classes()
    images = C.iter_images(C.RAW_IMAGES)
    if not images:
        C.die(f"{C.RAW_IMAGES} 下没有图片。请先采集/抽帧，或跑 python scripts/make_demo_data.py")

    prompt = build_prompt(PROMPT_GROUPS)
    C.banner(f"自动预标注：{len(images)} 张图片")
    if args.mock:
        C.warn("当前是 --mock 模式，产出的是假框！只能用来验证流程，不能用来训练最终模型。")
        detect = lambda p: mock_detect(p, C.scene_of(p), classes)  # noqa: E731
        run_cfg = {
            "mode": "mock",
            "note": "假标注，仅用于流程验证",
            "prompt": prompt,
            "prompt_groups": PROMPT_GROUPS,
            "classes": classes,
            "time": C.now_str(),
        }
    else:
        device = args.device or ("cuda" if _cuda_ok() else "cpu")
        if device == "cpu":
            C.warn("没检测到可用 GPU，用 CPU 跑会非常慢（约几秒一张），建议先 --limit 100 试水。")
        eng = GroundingDinoAnnotator(args.model, device, args.box_thresh, args.text_thresh)

        def detect(p: Path) -> list[dict]:
            dets = eng(p, prompt)
            for d in dets:
                d["cls"] = phrase_to_class(d["phrase"], PROMPT_GROUPS)
            dets = [d for d in dets if d["cls"]]
            return nms(dets, args.nms_iou)

        run_cfg = {
            "mode": "groundingdino",
            "model_id": args.model,
            "prompt": prompt,
            "prompt_groups": PROMPT_GROUPS,
            "box_threshold": args.box_thresh,
            "text_threshold": args.text_thresh,
            "nms_iou": args.nms_iou,
            "device": device,
            "classes": classes,
            "time": C.now_str(),
            "transformers": C.env_report().get("transformers"),
        }

    # 只处理"还没有标签"的图（公开集自带的标签不重标），--limit 也作用在这批上
    pending: list[Path] = []
    for img in images:
        out_txt = C.ANN_LABELS / C.scene_of(img) / f"{img.stem}.txt"
        if args.force or not out_txt.exists():
            pending.append(img)
    C.info(f"共 {len(images)} 张图：已有标签 {len(images) - len(pending)} 张（自动跳过），"
           f"待标注 {len(pending)} 张")
    if not pending:
        C.die("没有需要标注的图片了。要全部重标请加 --force")
    todo = pending[: args.limit] if args.limit else pending
    n_done = n_skip = n_fail = 0
    n_boxes = 0

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None  # type: ignore

    it = tqdm(todo, desc="标注") if tqdm else todo
    for i, img in enumerate(it, 1):
        scene = C.scene_of(img)
        out_txt = C.ANN_LABELS / scene / f"{img.stem}.txt"
        out_json = C.ANN_RAW / scene / f"{img.stem}.json"

        if out_txt.exists() and not args.force:
            n_skip += 1
            continue
        try:
            dets = detect(img)
        except Exception as e:  # 单张失败不中断整批
            n_fail += 1
            C.warn(f"{img.name} 失败：{e}")
            continue

        rows = to_yolo_rows(dets, classes)
        C.write_label(out_txt, rows)
        C.save_json(
            out_json,
            {
                "image": str(img.relative_to(C.DATASET_ROOT)),
                "config": run_cfg,
                "detections": dets,
            },
        )
        n_done += 1
        n_boxes += len(rows)
        if not tqdm and i % 50 == 0:
            C.info(f"  {i}/{len(todo)}")

    C.save_json(C.ANN_DIR / "last_run.json", run_cfg)

    C.banner("完成")
    C.info(
        f"新标注 {n_done} 张，跳过 {n_skip} 张（已存在），失败 {n_fail} 张，共写入 {n_boxes} 个框\n"
        f"标签目录：{C.ANN_LABELS}\n"
        f"原始输出（可追溯）：{C.ANN_RAW}\n\n"
        "下一步：人工复核。用 X-AnyLabeling 打开图片和同名 txt 修改，"
        "改完覆盖保存即可；这一步的认真程度直接决定模型上限。"
    )


def _cuda_ok() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
