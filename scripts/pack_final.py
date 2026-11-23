# -*- coding: utf-8 -*-
"""把 submission 目录打成最终提交包。

对已压缩格式（jpg/png/mp4/pt/zip）用 STORED 提速，其余用 DEFLATED。
"""
import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "submission"
OUT = ROOT / "submission_final.zip"

STORED_EXT = {".jpg", ".jpeg", ".png", ".mp4", ".mov", ".pt", ".pth", ".webp", ".gz"}

if not SRC.exists():
    sys.exit("submission 目录不存在")

files = [p for p in SRC.rglob("*") if p.is_file()]
total = sum(p.stat().st_size for p in files)
print("待打包 %d 个文件，共 %.1f MB" % (len(files), total / 1048576))

t0 = time.time()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as z:
    for i, p in enumerate(files, 1):
        arc = p.relative_to(SRC.parent)          # 保留 submission/ 顶层目录
        if p.suffix.lower() in STORED_EXT:
            z.write(p, arc, compress_type=zipfile.ZIP_STORED)
        else:
            z.write(p, arc, compress_type=zipfile.ZIP_DEFLATED)
        if i % 500 == 0:
            print("  已打包 %d/%d" % (i, len(files)))

print("完成：%s" % OUT)
print("体积：%.1f MB，耗时 %.1f 秒" % (OUT.stat().st_size / 1048576, time.time() - t0))
