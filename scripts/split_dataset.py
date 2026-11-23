# -*- coding: utf-8 -*-
"""按"场景"划分 train/val/test，生成 YOLO 训练用的目录。

为什么按场景而不是按图片随机划分？
    同一段视频抽出来的相邻帧几乎一样。如果按图片随机分，测试集里会出现
    "和训练集长得几乎相同"的图，指标虚高，现场换个场地就崩。评委最常抓这个。
    所以：一个场景目录（一次拍摄）整体落在一个 split 里，绝不拆开。

用法：
    python scripts/split_dataset.py
    python scripts/split_dataset.py --test-scenes scene20_会议室 --reviewed-scenes scene20_会议室
    python scripts/split_dataset.py --ratio 0.7 0.2 0.1 --seed 42

每次运行会清空 images/{train,val,test} 和 labels/{train,val,test} 后重建，
所以别把人工改过的标签放在这两个目录里（要改就改 ann/labels/）。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C


def label_for(img: Path, scene: str) -> Path:
    return C.ANN_LABELS / scene / f"{img.stem}.txt"


def collect() -> tuple[dict[str, list[Path]], list[Path]]:
    """按场景归集图片，同时返回被跳过的（没有标签的）图片。"""
    groups: dict[str, list[Path]] = {}
    skipped: list[Path] = []
    for img in C.iter_images(C.RAW_IMAGES):
        scene = C.scene_of(img)
        groups.setdefault(scene, [])
        if label_for(img, scene).exists():
            groups[scene].append(img)
        else:
            skipped.append(img)
    groups = {k: v for k, v in groups.items() if v}
    return groups, skipped


def assign(groups: dict[str, list[Path]], ratio: list[float], seed: int,
           forced_test: set[str], forced_train: set[str] = frozenset()) -> dict[str, str]:
    """把整场景分配到 split，返回 {场景: split}。"""
    import random

    total = sum(len(v) for v in groups.values())
    targets = {s: r * total for s, r in zip(C.SPLITS, ratio)}
    counts = {s: 0 for s in C.SPLITS}
    plan: dict[str, str] = {}

    # 强制进训练集的场景（比如唯一的近景火，绝不能被分到测试集）
    for scene in sorted(forced_train):
        if scene not in groups:
            C.warn(f"--train-scenes 里的 {scene} 不存在或没有标签，忽略")
            continue
        plan[scene] = "train"
        counts["train"] += len(groups[scene])

    for scene in forced_test:
        if scene not in groups or scene in plan:
            C.warn(f"--test-scenes 里的 {scene} 不存在/已由 --train-scenes 占用，忽略")
            continue
        plan[scene] = "test"
        counts["test"] += len(groups[scene])

    rest = [s for s in groups if s not in plan]
    rng = random.Random(seed)
    rng.shuffle(rest)
    # 大场景先放，能让比例更接近目标
    rest.sort(key=lambda s: -len(groups[s]))

    for scene in rest:
        # 放进"离目标差距最大"的那个 split
        split = max(C.SPLITS, key=lambda s: targets[s] - counts[s])
        plan[scene] = split
        counts[split] += len(groups[scene])

    # 保底：val / test 不能是空的
    for need in ("val", "test"):
        if counts[need] == 0:
            donors = [s for s, sp in plan.items() if sp == "train" and len(plan) >= 3]
            donors.sort(key=lambda s: len(groups[s]))
            if donors:
                scene = donors[0]
                plan[scene] = need
                counts["train"] -= len(groups[scene])
                counts[need] += len(groups[scene])
                C.warn(f"{need} 集原本为空，已从 train 挪一个场景过去：{scene}")
    return plan


def reset_split_dirs() -> None:
    for s in C.SPLITS:
        for base in (C.IMG_DIR / s, C.LBL_DIR / s):
            if base.exists():
                shutil.rmtree(base)
            C.ensure_dir(base)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    """优先硬链接（不占额外空间），跨盘失败时退回复制。"""
    if dst.exists():
        dst.unlink()
    if mode in ("auto", "hardlink"):
        try:
            dst.hardlink_to(src)
            return
        except Exception:
            if mode == "hardlink":
                C.warn("硬链接失败，改为复制")
    if mode == "symlink":
        try:
            dst.symlink_to(src)
            return
        except Exception:
            C.warn("符号链接失败，改为复制")
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="按场景划分数据集")
    ap.add_argument("--ratio", type=float, nargs=3, default=[0.7, 0.2, 0.1], help="train val test 比例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-scenes", default="", help="强制放进测试集的场景，逗号分隔")
    ap.add_argument("--train-scenes", default="",
                    help="强制放进训练集的场景，逗号分隔（用于保证关键样本一定被学到）")
    ap.add_argument("--reviewed-scenes", default="",
                    help="已经人工全量复核过的场景，逗号分隔；这些场景才适合当测试集")
    ap.add_argument("--mode", choices=["auto", "hardlink", "symlink", "copy"], default="auto")
    ap.add_argument("--include-unlabeled", action="store_true",
                    help="把没有标签文件的图片也纳入（当作负样本），默认跳过")
    args = ap.parse_args()

    ratio = list(args.ratio)
    if abs(sum(ratio) - 1.0) > 1e-6:
        C.die(f"比例之和必须为 1，当前 {sum(ratio)}")

    C.banner("划分数据集")
    groups, skipped = collect()

    if args.include_unlabeled and skipped:
        for img in skipped:
            groups.setdefault(C.scene_of(img), []).append(img)
        skipped = []

    if not groups:
        C.die(f"{C.ANN_LABELS} 下没有找到任何标签。先跑 python scripts/autolabel.py")
    if skipped:
        C.warn(f"{len(skipped)} 张图片没有标签，已跳过（例：{skipped[0].name}）")

    forced = {s.strip() for s in args.test_scenes.split(",") if s.strip()}
    forced_train = {s.strip() for s in args.train_scenes.split(",") if s.strip()}
    reviewed = {s.strip() for s in args.reviewed_scenes.split(",") if s.strip()}

    plan = assign(groups, ratio, args.seed, forced, frozenset(forced_train))
    reset_split_dirs()

    classes = C.load_classes()
    manifest: dict[str, list[str]] = {s: [] for s in C.SPLITS}
    n_img = {s: 0 for s in C.SPLITS}
    n_neg = {s: 0 for s in C.SPLITS}
    n_box = {s: 0 for s in C.SPLITS}
    class_box = {s: {c: 0 for c in classes} for s in C.SPLITS}

    for scene, split in sorted(plan.items()):
        for img in groups[scene]:
            new_stem = f"{scene}__{img.stem}"
            dst_img = C.IMG_DIR / split / f"{new_stem}{img.suffix.lower()}"
            link_or_copy(img, dst_img, args.mode)

            rows = C.read_label(label_for(img, scene))
            C.write_label(C.LBL_DIR / split / f"{new_stem}.txt", rows)

            n_img[split] += 1
            n_box[split] += len(rows)
            if not rows:
                n_neg[split] += 1
            for r in rows:
                if 0 <= r[0] < len(classes):
                    class_box[split][classes[r[0]]] += 1
            manifest[split].append(f"{new_stem}{img.suffix.lower()}")

    C.save_json(
        C.META_DIR / "split.json",
        {
            "time": C.now_str(),
            "seed": args.seed,
            "ratio": ratio,
            "classes": classes,
            "test_scenes_forced": sorted(forced),
            "reviewed_scenes": sorted(reviewed),
            "scene_plan": plan,
            "counts": {"images": n_img, "negatives": n_neg, "boxes": n_box, "per_class": class_box},
            "total_images": sum(n_img.values()),
            "files": manifest,
        },
    )

    C.banner("划分结果")
    C.info(f"{'split':6} {'图片':>6} {'负样本':>8} {'目标框':>8}")
    for s in C.SPLITS:
        C.info(f"{s:6} {n_img[s]:>6} {n_neg[s]:>8} {n_box[s]:>8}")
    C.info(f"{'合计':6} {sum(n_img.values()):>6} {sum(n_neg.values()):>8} {sum(n_box.values()):>8}")
    C.info(f"\n场景数：{len(plan)} 个")
    for s in C.SPLITS:
        scenes = [k for k, v in plan.items() if v == s]
        C.info(f"  {s}: {len(scenes)} 个场景 -> {', '.join(sorted(scenes)[:6])}{' ...' if len(scenes) > 6 else ''}")

    if not plan or "test" not in plan.values():
        C.warn("测试集为空，无法评估")
    if n_neg["test"] == 0:
        C.warn("测试集里没有负样本，算不出误报率。建议把某个纯'无火'场景用 --test-scenes 放进测试集。")
    if forced and not (forced <= set(reviewed)):
        C.warn("测试集里有场景没标记为 --reviewed-scenes，"
               "意味着它可能是大模型预标注的。用来评估 GroundingDINO 会形成循环论证，请确认。")

    C.info(f"\n明细：{C.META_DIR / 'split.json'}\n下一步：python scripts/check_labels.py")


if __name__ == "__main__":
    main()
