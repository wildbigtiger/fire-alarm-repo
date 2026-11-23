#!/bin/bash
# ModelArts Notebook / 训练作业里一次性装依赖。
# 用法：bash bootstrap.sh
set -e

echo "=== 1/3 检查 GPU ==="
nvidia-smi || echo "[WARN] 没看到 GPU，确认创建 Notebook/训练作业时选了 GPU 规格"

echo "=== 2/3 检查 PyTorch ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'available', torch.cuda.is_available())" || {
    echo "[ERROR] 镜像里没有 torch，请换成 PyTorch 镜像重新创建环境"
    exit 1
}

echo "=== 3/3 安装依赖 ==="
# 先装 ultralytics 的依赖，但不让它乱改已有的 torch
pip install --no-cache-dir -U pip
pip install --no-cache-dir ultralytics "opencv-python-headless>=4.8" PyYAML tqdm

# 自动标注才需要的（子任务 2 / 4 用得上）
if [ "${WITH_AUTOLABEL:-0}" = "1" ]; then
    pip install --no-cache-dir "transformers>=4.41" accelerate
fi

python -c "import ultralytics, cv2; print('ultralytics', ultralytics.__version__, '| cv2', cv2.__version__)"
echo "=== 环境就绪 ==="
