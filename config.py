"""集中配置 — 所有路径/参数默认值都在这里，可被 CLI 参数覆盖。

约定：默认路径尽量相对项目根目录。模型目录可用环境变量 MODELS_DIR
或 `--models-dir` 覆盖。
"""

import os

# ---------------------------------------------------------------------------
# 项目根目录（以此推导其他默认路径）
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 服务器
# ---------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 5001            # 避开 Image-Tool 默认使用的 5000 端口

# ---------------------------------------------------------------------------
# 摄像头（红外，分辨率自适应：256×192 或 640×512，启动时按实际设备读取）
# ---------------------------------------------------------------------------
DEFAULT_CAM_INDEX = 0
CAM_FPS_TARGET = 30          # 请求的采集帧率
CAM_API = "MSMF"             # 该 IR 相机必须用 MSMF（DSHOW 无法打开），且 CONVERT_RGB=0 取 YUY2 原始帧

# ---------------------------------------------------------------------------
# 模型（目录内所有 .pth 均可选，每个面板独立选择显示哪个模型的效果）
# ---------------------------------------------------------------------------
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))  # 可用 --models-dir 覆盖
RAW_LABEL = "原图"                # 面板可选项之一：显示原始图像（最近邻放大到面板）

# 模型期望输入尺寸：按权重文件名中包含的关键字匹配（按列表顺序，先长后短）。
# 推理前会把相机帧缩放为该模型的输入尺寸：名字含 640 ← 640×512，含 256 ← 256×192。
MODEL_INPUT_SIZES = [
    ("640", (640, 512)),
    ("256", (256, 192)),
]
DEFAULT_INPUT_SIZE = (640, 512)   # 名字不含已知关键字的模型使用此默认输入

DEFAULT_DEVICE = "cpu"            # "cpu" 或 "cuda"

# ---------------------------------------------------------------------------
# HAT（重模型：直播不推理；拍照即时；录像停止后离线超分）
# ---------------------------------------------------------------------------
HAT_LABEL = "HAT×4"
HAT_SCALE = 4
# 默认指向同级目录下 Image-Tool 仓库（https://github.com/wuzhongqi57/Image-Tool）的
# model_zoo/hat/ 权重；可用 --hat-weights / --fusion-root / 环境变量覆盖
FUSION_TOOL_ROOT = os.environ.get(
    "FUSION_TOOL_ROOT",
    os.path.join(os.path.dirname(PROJECT_ROOT), "image_fusion_tool"),
)
HAT_WEIGHTS = os.environ.get(
    "HAT_WEIGHTS",
    os.path.join(FUSION_TOOL_ROOT, "model_zoo", "hat", "Real_HAT_GAN_SRx4_sharper.pth"),
)
HAT_DEVICE = os.environ.get("HAT_DEVICE", "")  # 空则跟随 --device / DEFAULT_DEVICE

# ---------------------------------------------------------------------------
# 推流
# ---------------------------------------------------------------------------
JPEG_QUALITY = 80

# Stream downscaling: max dimensions for live stream (aspect preserved).
# Full resolution still used for snapshots/recordings.
# Default fits within 1280x1024; set to 0 to disable.
STREAM_MAX_WIDTH = 1280
STREAM_MAX_HEIGHT = 1024

# ---------------------------------------------------------------------------
# FP16 Inference
# ---------------------------------------------------------------------------
# Use FP16 (half precision) on CUDA for faster inference. Negligible quality loss.
# Default: True for CUDA (auto-fallback to FP32 on CPU or if unavailable)
USE_FP16 = True

# ---------------------------------------------------------------------------
# Stats Overlay
# ---------------------------------------------------------------------------
# Show per-panel stats (FPS, render ms, frame age) - toggled via toolbar or ?stats=1
STATS_OVERLAY = False

# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[realtime_infer] {msg}")
