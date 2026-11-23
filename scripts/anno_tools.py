# -*- coding: utf-8 -*-
"""标注辅助工具：一键把某个场景准备好，标注完再一键写回。

用法（一般通过根目录的 标注开始.bat / 标注完成.bat 双击调用）：

    python scripts/anno_tools.py start      # 选场景 -> 准备好工作目录并打开软件
    python scripts/anno_tools.py start --scene scene04_后厨明火
    python scripts/anno_tools.py finish     # 把改好的 json 写回 YOLO txt
    python scripts/anno_tools.py finish --scene scene04_后厨明火
    python scripts/anno_tools.py list       # 看各场景标注进度

为什么要转 json：
    X-AnyLabeling 只认自己那套 json（LabelMe 格式），不认 YOLO txt。
    每次在软件里"导入 YOLO 标注"很麻烦，所以这里预先把它生成好，
    放在图片旁边，软件打开目录就直接显示框，改完 Ctrl+S 就存到 json。

写回：
    finish 会把 json 转回 YOLO txt，覆盖 ann/labels/<场景>/，
    覆盖前自动备份到 ann/labels_backup/<场景>/<时间戳>/。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C

APP = Path(r"C:\Users\wildbigtiger\AppData\Local\Programs\X-AnyLabeling\X-AnyLabeling.exe")
WORK = C.PROJECT_ROOT / "anno_work"
BACKUP = C.ANN_DIR / "labels_backup"
JSON_VERSION = "2.4.0"


def list_scenes() -> list[str]:
    scenes = []
    for d in sorted(p for p in C.RAW_IMAGES.iterdir() if p.is_dir()):
        if any(d.iterdir()):
            scenes.append(d.name)
    return scenes


def scene_progress(scene: str) -> tuple[int, int]:
    imgs = C.iter_images(C.RAW_IMAGES / scene)
    n_box = 0
    for p in imgs:
        if C.read_label(C.ANN_LABELS / scene / (p.stem + ".txt")):
            n_box += 1
    return len(imgs), n_box


def to_json(image_path: Path, rows: list, classes: list[str]) -> dict:
    W, H = C.image_size(image_path)
    shapes = []
    for cls, cx, cy, w, h in rows:
        x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
        x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
        shapes.append({
            "label": classes[cls] if 0 <= cls < len(classes) else str(cls),
            "points": [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)]],
            "group_id": None, "description": "", "difficult": False,
            "shape_type": "rectangle", "flags": {}, "attributes": {},
        })
    return {
        "version": JSON_VERSION, "flags": {}, "shapes": shapes,
        "imagePath": image_path.name, "imageData": None,
        "imageHeight": H, "imageWidth": W,
    }


ALIASES = {
    "火": "fire", "火焰": "fire", "明火": "fire", "flame": "fire", "fire": "fire",
    "烟": "smoke", "烟雾": "smoke", "浓烟": "smoke", "smoke": "smoke",
}


def normalize_label(raw: str) -> str | None:
    """把各种写法规整成 fire / smoke，认不出的返回 None。"""
    s = str(raw).strip().strip("，,。.")
    if not s:
        return None
    key = s.lower()
    if key in ALIASES:
        return ALIASES[key]
    return None


def from_json(data: dict, classes: list[str], unknown: set | None = None) -> list:
    out = []
    W, H = float(data.get("imageWidth") or 1), float(data.get("imageHeight") or 1)
    for s in data.get("shapes", []):
        label = normalize_label(s.get("label", ""))
        if label is None or label not in classes:
            if unknown is not None:
                unknown.add(str(s.get("label", "")))
            continue
        pts = s.get("points") or []
        if len(pts) < 2:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        x1, x2 = max(0.0, min(xs)), min(W, max(xs))
        y1, y2 = max(0.0, min(ys)), min(H, max(ys))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append((classes.index(label),
                    (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                    (x2 - x1) / W, (y2 - y1) / H))
    return out


def pick_scene(arg: str | None) -> str:
    scenes = list_scenes()
    if arg:
        if arg not in scenes:
            C.die(f"场景不存在：{arg}\n可选：{', '.join(scenes)}")
        return arg
    C.info("可用场景（图片数 / 已标注数）：")
    for i, s in enumerate(scenes, 1):
        n, nb = scene_progress(s)
        C.info(f"  {i:2}. {s:24} {n:4} 张   {nb:4} 张有框")
    raw = input("\n输入序号（直接回车默认 1）：").strip() or "1"
    try:
        return scenes[int(raw) - 1]
    except Exception:
        C.die("序号不合法")


def cmd_start(scene: str, launch: bool = True) -> None:
    classes = C.load_classes()
    imgs = C.iter_images(C.RAW_IMAGES / scene)
    if not imgs:
        C.die(f"{scene} 没有图片")

    work = C.ensure_dir(WORK / scene)
    n_new = n_skip = 0
    for p in imgs:
        dst_img = work / p.name
        if not dst_img.exists():
            try:
                dst_img.hardlink_to(p)
            except Exception:
                shutil.copy2(p, dst_img)
        js = work / (p.stem + ".json")
        if not js.exists():  # 已有 json 说明改过，别覆盖
            rows = C.read_label(C.ANN_LABELS / scene / (p.stem + ".txt"))
            js.write_text(json.dumps(to_json(p, rows, classes), ensure_ascii=False, indent=2),
                          encoding="utf-8")
            n_new += 1
        else:
            n_skip += 1

    C.banner(f"已准备：{scene}")
    C.info(f"工作目录：{work}")
    C.info(f"生成标注 {n_new} 份（已有 {n_skip} 份，保留你的修改）")
    C.info("")
    C.info("接下来：软件里点【文件 -> 打开目录】，选上面这个 work 目录")
    C.info("会直接看到预标注的框，改完 Ctrl+S 自动存 json。")
    C.info("全部改完，双击【标注完成.bat】把结果写回 YOLO 标签。")

    if not launch:
        return
    # 用系统外壳打开文件夹：直接在打包程序里起 explorer.exe 会报 0xc0000142
    opened = False
    try:
        os.startfile(str(work))  # noqa: S606 - 打开资源管理器
        opened = True
    except Exception:
        pass
    if not opened:
        try:
            subprocess.Popen(["cmd", "/c", "start", "", str(work)], shell=False)
            opened = True
        except Exception:
            pass
    if not opened:
        C.warn("自动打开文件夹失败，请手动打开上面的路径")
    if APP.exists():
        C.info("\n正在启动 X-AnyLabeling ...")
        subprocess.Popen([str(APP)])
    else:
        C.warn(f"没找到软件：{APP}")


def cmd_finish(scene: str) -> None:
    classes = C.load_classes()
    work = WORK / scene
    if not work.exists():
        C.die(f"没有工作目录 {work}，先运行 start")

    jsons = sorted(work.glob("*.json"))
    if not jsons:
        C.die("工作目录里没有 json，先运行 start 并在软件里标注")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = C.ensure_dir(BACKUP / scene / stamp)
    n_changed = n_total = 0
    unknown: set = set()
    for js in jsons:
        stem = js.stem
        tgt = C.ANN_LABELS / scene / (stem + ".txt")
        old = C.read_label(tgt)
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except Exception as e:
            C.warn(f"{js.name} 解析失败：{e}")
            continue
        new = from_json(data, classes, unknown)
        n_total += 1
        if (len(old) != len(new)) or any(
                abs(a[1] - b[1]) + abs(a[2] - b[2]) + abs(a[3] - b[3]) + abs(a[4] - b[4]) > 1e-4
                or a[0] != b[0] for a, b in zip(sorted(old), sorted(new))):
            if tgt.exists():
                C.ensure_dir(bak)
                shutil.copy2(tgt, bak / tgt.name)
            C.write_label(tgt, new)
            n_changed += 1

    C.banner("已写回 YOLO 标签")
    C.info(f"场景：{scene}    检查 {n_total} 份，实际改动 {n_changed} 份")
    if unknown:
        C.warn(f"有 {len(unknown)} 种标签名认不出来，对应框被跳过：{sorted(unknown)}")
        C.warn("只认 fire / smoke（也接受 火/烟/火焰/烟雾/flame 这些别名）")
    if n_changed:
        C.info(f"原标签已备份到：{bak}")
    C.info("")
    C.info("提醒：工作目录 anno_work 可以留着（下次 start 会保留你的修改），也可以手动删。")


def cmd_list() -> None:
    C.banner("各场景标注进度")
    tot = tot_box = 0
    for s in list_scenes():
        n, nb = scene_progress(s)
        tot += n
        tot_box += nb
        C.info(f"  {s:24} {n:5} 张   有框 {nb:5} 张")
    C.info(f"  {'合计':24} {tot:5} 张   有框 {tot_box:5} 张")


def main() -> None:
    ap = argparse.ArgumentParser(description="标注辅助工具")
    ap.add_argument("action", choices=["start", "finish", "list"])
    ap.add_argument("--scene", default=None)
    ap.add_argument("--no-launch", action="store_true", help="只准备目录，不打开软件")
    args = ap.parse_args()

    if args.action == "list":
        cmd_list()
        return

    scene = pick_scene(args.scene)
    if args.action == "start":
        cmd_start(scene, launch=not args.no_launch)
    else:
        cmd_finish(scene)


if __name__ == "__main__":
    main()
