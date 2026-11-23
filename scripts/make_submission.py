# -*- coding: utf-8 -*-
"""一键打包提交物。

    # 只打包代码 + 文档 + 权重（体积小，先看看结构）
    python scripts/make_submission.py

    # 连图片、标注一起打（视频仍不含）
    python scripts/make_submission.py --with-data

    # 连视频也打进去（体积可能几个 G，慎用）
    python scripts/make_submission.py --with-data --with-videos

产物：
    submission/          整理好的目录结构
    submission_<时间>.zip 最终压缩包

打包完会生成 8_清单.md，列出每类文件数量和缺失项。
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C

# 打包源码时要排除的东西
SRC_IGNORE = {
    ".venv", ".git", "datasets", "runs", "submission", "__pycache__",
    ".ultra", ".tmp", ".pkgs", ".pkgs2", "weights", "Ultralytics",
    ".hf", "anno_work",
}
SRC_IGNORE_GLOB = ("torchinductor_*", "*.pyc", "*.zip", "yolo*.pt", "submission_*.zip",
                   "PortableGit*")


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def ignore_factory():
    def _ignore(directory, names):
        skipped = set()
        for n in names:
            if n in SRC_IGNORE:
                skipped.add(n)
            elif any(Path(n).match(g) for g in SRC_IGNORE_GLOB):
                skipped.add(n)
        return skipped

    return _ignore


class Collector:
    def __init__(self):
        self.groups: dict[str, dict] = {}

    def take(self, label: str, src: Path, dst: Path, mode: str = "tree") -> None:
        info = self.groups.setdefault(label, {"files": 0, "size": 0, "src": [], "missing": []})
        if not src.exists():
            info["missing"].append(str(src))
            return
        if mode == "tree":
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=ignore_factory())
        else:
            for f in src.rglob("*") if src.is_dir() else [src]:
                if not f.is_file():
                    continue
                if any(f.match(g) for g in SRC_IGNORE_GLOB):
                    continue
                target = dst / f.name
                C.ensure_dir(target.parent)
                shutil.copy2(f, target)
        files = [p for p in dst.rglob("*") if p.is_file()]
        info["files"] += len(files)
        info["size"] += sum(p.stat().st_size for p in files)
        info["src"].append(str(src.relative_to(C.PROJECT_ROOT)))


def main() -> None:
    ap = argparse.ArgumentParser(description="打包提交物")
    ap.add_argument("--out", default=str(C.PROJECT_ROOT / "submission"))
    ap.add_argument("--with-data", action="store_true", help="包含原始图片与标注")
    ap.add_argument("--with-videos", action="store_true", help="包含视频（很大）")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    C.ensure_dir(out)

    col = Collector()
    C.banner(f"整理提交物 -> {out}")

    # 1 源代码
    col.take("1_源代码", C.PROJECT_ROOT, out / "1_源代码")

    # 2 设计报告（找得到的都放进去）
    rep = C.ensure_dir(out / "2_设计报告")
    found_rep = False
    for cand in list(C.PROJECT_ROOT.glob("docs/*报告*")) + list(C.PROJECT_ROOT.glob("*报告*")):
        if cand.is_file():
            shutil.copy2(cand, rep / cand.name)
            found_rep = True
    if not found_rep:
        (rep / "_放置报告.txt").write_text(
            "把按官方模板写好的设计报告（.docx 可编辑版）放这里，文件名建议：\n"
            "第4题-智能火灾报警系统-设计报告.docx\n", encoding="utf-8")

    # 3 演示视频
    vid = C.ensure_dir(out / "3_演示视频")
    demo = C.PROJECT_ROOT / "runs" / "demo"
    n_demo = 0
    if demo.exists():
        for f in demo.glob("*.mp4"):
            shutil.copy2(f, vid / f.name)
            n_demo += 1
    if n_demo == 0:
        (vid / "_放置演示视频.txt").write_text(
            "把系统演示录屏放这里（建议 3~5 分钟，覆盖：实时检测、趋势曲线、历史回放、\n"
            "摄像头角度、GPS 地图、三级报警）。\n", encoding="utf-8")

    # 4 数据
    data = C.ensure_dir(out / "4_数据")
    if args.with_data:
        col.take("4_数据/原始图片", C.RAW_IMAGES, data / "原始图片")
        col.take("4_数据/标注", C.ANN_LABELS, data / "标注")
        col.take("4_数据/标注原始输出", C.ANN_RAW, data / "标注原始输出")
        col.take("4_数据/划分与统计", C.META_DIR, data / "划分与统计")
        if args.with_videos:
            col.take("4_数据/视频素材", C.RAW_VIDEOS, data / "视频素材")
            col.take("4_数据/视频片段", C.RAW_CLIPS, data / "视频片段")
    else:
        (data / "_未包含数据.txt").write_text(
            "本次打包未包含数据。要打数据请加参数：--with-data（图片+标注），\n"
            "再加 --with-videos 连视频一起打。\n", encoding="utf-8")

    # 5 模型权重
    w = C.ensure_dir(out / "5_模型权重")
    n_pt = 0
    for run in sorted((C.RUNS_DIR / "detect").glob("*")) if (C.RUNS_DIR / "detect").exists() else []:
        best = run / "weights" / "best.pt"
        if best.exists():
            d = C.ensure_dir(w / run.name)
            shutil.copy2(best, d / "best.pt")
            for extra in ("repro.json", "args.yaml"):
                if (run / extra).exists():
                    shutil.copy2(run / extra, d / extra)
            n_pt += 1
    if n_pt == 0:
        (w / "_无权重.txt").write_text("还没有训练好的权重（runs/detect/*/weights/best.pt）\n", encoding="utf-8")

    # 6 评估结果
    ev = C.ensure_dir(out / "6_评估结果")
    n_ev = 0
    if (C.RUNS_DIR / "eval").exists():
        for d in (C.RUNS_DIR / "eval").iterdir():
            if (d / "test_metrics.md").exists():
                dst = C.ensure_dir(ev / d.name)
                for f in d.rglob("*"):
                    if f.is_file() and f.suffix in (".md", ".json", ".png", ".csv", ".jpg"):
                        target = dst / f.relative_to(d)
                        C.ensure_dir(target.parent)
                        shutil.copy2(f, target)
                n_ev += 1
    if n_ev == 0:
        (ev / "_无评估.txt").write_text("还没有测试集评估结果（runs/eval/）\n", encoding="utf-8")

    # 7 说明文档
    doc = C.ensure_dir(out / "7_说明文档")
    for f in [C.PROJECT_ROOT / "README.md", C.PROJECT_ROOT / "运行说明.md",
              C.PROJECT_ROOT / "docs" / "题目要求与交付清单.md",
              C.PROJECT_ROOT / "docs" / "交付自查.md"]:
        if f.exists():
            shutil.copy2(f, doc / f.name)
    for f in [C.META_DIR / "dataset_stats.md", C.META_DIR / "label_check.md"]:
        if f.exists():
            shutil.copy2(f, doc / f.name)

    # 8 清单
    lines = ["# 提交物清单", "", f"打包时间：{C.now_str()}", "",
             f"项目根目录：`{C.PROJECT_ROOT}`", "",
             "| 分组 | 文件数 | 体积 | 来源 |", "|---|---|---|---|"]
    for label, info in col.groups.items():
        lines.append(f"| {label} | {info['files']} | {human(info['size'])} | {', '.join(info['src'])} |")
    lines.append(f"| 5_模型权重 | {n_pt} | - | runs/detect/*/weights/best.pt |")
    lines.append(f"| 6_评估结果 | {n_ev} | - | runs/eval/* |")
    total_size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    lines += ["", f"合计：{human(total_size)}", ""]

    missing = [m for info in col.groups.values() for m in info["missing"]]
    if missing:
        lines += ["## 缺失（源目录不存在）", ""] + [f"- `{m}`" for m in missing]

    lines += ["", "## 人工确认项", "",
              "- [ ] 设计报告已按官方模板填写并放入 `2_设计报告/`",
              "- [ ] 演示视频已放入 `3_演示视频/`",
              "- [ ] 报告中已写明：类别体系、提示词与调参过程、标注可追溯说明",
              "- [ ] 报告中已写明：算力型号（RTX 5060 Laptop 8GB）与训练时长",
              "- [ ] 报告中已写明：所有指标的测试集/IoU阈值/输入分辨率口径一致",
              "- [ ] 报告中已写明：是否微调开放域模型、微调样本量",
              ""]
    (out / "8_清单.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    C.info(f"整理完成，合计 {human(total_size)}")
    for label, info in col.groups.items():
        C.info(f"  {label:16} {info['files']:>5} 个文件  {human(info['size']):>9}")

    if not args.no_zip:
        zip_path = C.PROJECT_ROOT / f"submission_{C.stamp()}.zip"
        C.info(f"\n压缩中 -> {zip_path.name}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for f in out.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(out.parent))
        C.info(f"完成：{zip_path}（{human(zip_path.stat().st_size)}）")

    C.info("\n别忘了：README 里的「还未做的」三块（SAM掩码、双人一致性、开放域对比）"
           "要补完，报告才有发挥分。")


if __name__ == "__main__":
    main()
