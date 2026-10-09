"""Real-HAT-GAN ×4 后端 — 拍照即时推理；录像仅在停止后离线调用。直播路径禁止调用。

复用 Image-Tool 仓库（https://github.com/wuzhongqi57/Image-Tool）的 network_hat.py 与本地权重，不把大权重打进本仓库。
"""

from __future__ import annotations

import os
import sys
import threading

import cv2
import numpy as np

import config

_lock = threading.Lock()
_model = None
_device = None

_HAT_CFG = dict(
    upscale=4,
    in_chans=3,
    img_size=64,
    window_size=16,
    compress_ratio=3,
    squeeze_factor=30,
    conv_scale=0.01,
    overlap_ratio=0.5,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler="pixelshuffle",
    resi_connection="1conv",
)


def _fusion_root() -> str:
    return os.environ.get("FUSION_TOOL_ROOT", config.FUSION_TOOL_ROOT)


def weight_path() -> str:
    return os.path.abspath(config.HAT_WEIGHTS)


def available() -> tuple[bool, str]:
    """权重与网络模块是否可用（不强制立刻加载到 GPU）。"""
    wp = weight_path()
    if not os.path.isfile(wp):
        return False, f"找不到 HAT 权重: {wp}"
    if os.path.getsize(wp) < 1024:
        return False, f"HAT 权重文件异常（过小）: {wp}"
    root = _fusion_root()
    net = os.path.join(root, "network_hat.py")
    if not os.path.isfile(net):
        return False, f"找不到 network_hat.py: {net}"
    return True, wp


def is_heavy(name: str) -> bool:
    return name == config.HAT_LABEL


def scale() -> int:
    return int(config.HAT_SCALE)


def output_size_for_frame(frame: np.ndarray) -> tuple[int, int]:
    """返回 (W, H) = 输入灰度帧 × HAT_SCALE。"""
    h, w = frame.shape[:2]
    s = scale()
    return w * s, h * s


def _ensure_hat_importable() -> None:
    root = os.path.abspath(_fusion_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_model():
    global _model, _device
    with _lock:
        if _model is not None:
            return _model
        ok, detail = available()
        if not ok:
            raise RuntimeError(detail)

        import torch
        _ensure_hat_importable()
        from network_hat import HAT

        config.log(f"加载 HAT 权重: {weight_path()}")
        net = HAT(**_HAT_CFG)
        sd = torch.load(weight_path(), map_location="cpu", weights_only=False)
        if not isinstance(sd, dict) or "params_ema" not in sd:
            raise RuntimeError("HAT 权重缺少 params_ema")
        sd_ema = {
            k: v for k, v in sd["params_ema"].items()
            if "relative_position_index" not in k
        }
        net.load_state_dict(sd_ema, strict=False)
        net.eval()
        for p in net.parameters():
            p.requires_grad = False

        prefer = (config.HAT_DEVICE or config.DEFAULT_DEVICE or "cuda").lower()
        if prefer == "cuda" and torch.cuda.is_available():
            net = net.cuda()
            _device = "cuda"
        else:
            _device = "cpu"
            config.log("HAT 使用 CPU（可能非常慢）")
        _model = net
        config.log(f"HAT 已就绪 (device={_device}, ×{scale()})")
        return _model


def infer(gray: np.ndarray) -> np.ndarray:
    """全帧 HAT ×4。gray: uint8 (H,W) → uint8 (H*4, W*4)。

    仅应在拍照/录像路径调用。
    """
    import torch

    if gray is None or gray.size == 0:
        raise ValueError("空帧")
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.ndim != 2:
        raise ValueError(f"期望灰度图，got shape={gray.shape}")

    model = _load_model()
    x = np.stack([gray] * 3, axis=-1)
    t = torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    dev = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    t = t.to(device=dev, dtype=dtype)
    with torch.no_grad():
        out = model(t)
    out_np = out.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    del t, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
    return out_np[:, :, 0]


def make_placeholder(frame: np.ndarray, panel_wh: tuple[int, int]) -> np.ndarray:
    """直播占位：原图放大到面板尺寸并叠加提示文字。"""
    pw, ph = panel_wh
    if frame is None:
        canvas = np.zeros((max(ph, 1), max(pw, 1)), dtype=np.uint8)
    else:
        img = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ph and pw and (img.shape[0], img.shape[1]) != (ph, pw):
            canvas = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_NEAREST)
        else:
            canvas = img.copy()
    # 半透明底条 + 提示
    h, w = canvas.shape[:2]
    bar_h = max(28, h // 16)
    y0 = max(0, h - bar_h - 8)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), 0, -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    tip = "HAT x4: snapshot now; after record stop"
    cv2.putText(
        canvas, tip, (12, min(h - 12, y0 + bar_h - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, max(0.45, w / 900.0), 255, 1, cv2.LINE_AA,
    )
    tip2 = config.HAT_LABEL + " offline"
    cv2.putText(
        canvas, tip2, (12, max(24, bar_h)),
        cv2.FONT_HERSHEY_SIMPLEX, max(0.5, w / 800.0), 255, 1, cv2.LINE_AA,
    )
    return canvas
