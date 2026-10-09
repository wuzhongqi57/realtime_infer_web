"""YUY2 编解码与 UV 缩放 — 与板端对照的统一契约。

像素格式：YUY2 / YUYV 4:2:2，W 必须为偶数。
字节序（每宏像素 4 字节，覆盖 2 个亮度采样）：
  Y0 U0 Y1 V0 | Y2 U1 Y3 V1 | …

颜色转换：BT.601 full-range（Y/U/V ∈ [0,255]，U/V 零点 = 128）。
  Y = 0.299 R + 0.587 G + 0.114 B
  U = -0.168736 R - 0.331264 G + 0.5 B + 128
  V = 0.5 R - 0.418688 G - 0.081312 B + 128
反向：
  R = Y + 1.402 (V-128)
  G = Y - 0.344136 (U-128) - 0.714136 (V-128)
  B = Y + 1.772 (U-128)
"""

from __future__ import annotations

import numpy as np
import cv2


def split_yuy2(buf: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """YUY2 原始缓冲 → Y (H,W), U (H,W), V (H,W)。

    U/V 在偶数列写入宏像素色度，奇数列复制同宏像素值（便于按像素索引）。
    """
    if width % 2 != 0:
        raise ValueError(f"YUY2 要求宽度为偶数，收到 W={width}")
    flat = np.asarray(buf, dtype=np.uint8).reshape(-1)
    expect = width * height * 2
    if flat.size != expect:
        raise ValueError(f"YUY2 缓冲长度不符: got {flat.size}, expect {expect}")
    packed = flat.reshape(height, width // 2, 4)
    y0 = packed[:, :, 0]
    u = packed[:, :, 1]
    y1 = packed[:, :, 2]
    v = packed[:, :, 3]
    y = np.empty((height, width), dtype=np.uint8)
    y[:, 0::2] = y0
    y[:, 1::2] = y1
    u_full = np.repeat(u, 2, axis=1)
    v_full = np.repeat(v, 2, axis=1)
    return y, u_full, v_full


def uv_pair_from_yuy2(buf: np.ndarray, width: int, height: int) -> np.ndarray:
    """返回 UV (H, W, 2)，与 split_yuy2 的 U/V 全分辨率平面一致。"""
    _y, u, v = split_yuy2(buf, width, height)
    return np.stack([u, v], axis=-1)


def pack_yuy2(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Y/U/V (H,W) → YUY2 一维 uint8 缓冲（长度 H*W*2）。

    每个宏像素取偶数列的 U/V（标准 4:2:2 抽样）。
    """
    if y.ndim != 2 or u.shape != y.shape or v.shape != y.shape:
        raise ValueError(f"Y/U/V 形状必须同为 (H,W)，收到 Y={y.shape} U={u.shape} V={v.shape}")
    h, w = y.shape
    if w % 2 != 0:
        raise ValueError(f"YUY2 要求宽度为偶数，收到 W={w}")
    y = np.asarray(y, dtype=np.uint8)
    u = np.asarray(u, dtype=np.uint8)
    v = np.asarray(v, dtype=np.uint8)
    out = np.empty((h, w // 2, 4), dtype=np.uint8)
    out[:, :, 0] = y[:, 0::2]
    out[:, :, 1] = u[:, 0::2]
    out[:, :, 2] = y[:, 1::2]
    out[:, :, 3] = v[:, 0::2]
    return out.reshape(-1)


def pack_yuy2_neutral(y: np.ndarray) -> np.ndarray:
    """灰度 Y + 中性 UV=128 → YUY2。"""
    h, w = y.shape
    u = np.full((h, w), 128, dtype=np.uint8)
    v = np.full((h, w), 128, dtype=np.uint8)
    return pack_yuy2(y, u, v)


def upsample_uv_bilinear(uv: np.ndarray, scale: int) -> np.ndarray:
    """UV (H,W,2) 或 U/V (H,W) 双线性放大 scale 倍。"""
    if scale < 1:
        raise ValueError(f"scale 必须 >= 1，收到 {scale}")
    if scale == 1:
        return uv.copy()
    if uv.ndim == 2:
        h, w = uv.shape
        return cv2.resize(uv, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    if uv.ndim == 3 and uv.shape[2] == 2:
        h, w = uv.shape[:2]
        return cv2.resize(uv, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"uv 形状应为 (H,W) 或 (H,W,2)，收到 {uv.shape}")


def bgr_to_yuv_planes(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BGR uint8 (H,W,3) → Y,U,V uint8 (H,W)，BT.601 full-range。"""
    bgr = np.asarray(bgr, dtype=np.float32)
    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    y = np.clip(y, 0, 255).astype(np.uint8)
    u = np.clip(u, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    return y, u, v


def yuv_planes_to_bgr(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Y,U,V uint8 → BGR uint8，BT.601 full-range。"""
    yf = y.astype(np.float32)
    uf = u.astype(np.float32) - 128.0
    vf = v.astype(np.float32) - 128.0
    r = yf + 1.402 * vf
    g = yf - 0.344136 * uf - 0.714136 * vf
    b = yf + 1.772 * uf
    bgr = np.stack([
        np.clip(b, 0, 255),
        np.clip(g, 0, 255),
        np.clip(r, 0, 255),
    ], axis=-1).astype(np.uint8)
    return bgr


def bgr_to_yuy2(bgr: np.ndarray) -> np.ndarray:
    """BGR → YUY2 缓冲。"""
    y, u, v = bgr_to_yuv_planes(bgr)
    return pack_yuy2(y, u, v)


def yuy2_to_bgr(buf: np.ndarray, width: int, height: int) -> np.ndarray:
    """YUY2 缓冲 → BGR。"""
    y, u, v = split_yuy2(buf, width, height)
    return yuv_planes_to_bgr(y, u, v)


def gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.asarray(gray, dtype=np.uint8), cv2.COLOR_GRAY2BGR)


def roundtrip_ok(y: np.ndarray) -> bool:
    """自测：neutral pack → split 应还原 Y，U/V≈128。"""
    packed = pack_yuy2_neutral(y)
    h, w = y.shape
    y2, u2, v2 = split_yuy2(packed, w, h)
    return bool(np.array_equal(y, y2) and np.all(u2 == 128) and np.all(v2 == 128))
