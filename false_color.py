"""红外伪彩 LUT — 与板端共享的 256 档调色板。

档位（id）：
  off / white_hot  白热：Y 直出，UV=128（关伪彩）
  black_hot        黑热：Y'=255-Y，UV=128
  ironbow          铁虹：COLORMAP_INFERNO 烘焙（主推）
  lava             熔岩：COLORMAP_HOT
  rainbow          彩虹：COLORMAP_TURBO

彩色档 LUT 文件布局（luts/<id>.bin）：
  256 × 3 字节，顺序 index=0..255，每项 B,G,R（与 OpenCV BGR 一致）
同内容另存 luts/<id>.csv（每行 B,G,R）。
"""

from __future__ import annotations

import csv
import os
import threading

import cv2
import numpy as np

import config
import yuv_codec

PALETTE_IDS = ("off", "white_hot", "black_hot", "ironbow", "lava", "rainbow")
PALETTE_LABELS = {
    "off": "关伪彩(白热)",
    "white_hot": "白热",
    "black_hot": "黑热",
    "ironbow": "铁虹",
    "lava": "熔岩",
    "rainbow": "彩虹",
}
# off 与 white_hot 行为相同
_ALIASES = {"off": "white_hot"}

_GRAY_IDS = frozenset({"white_hot", "black_hot"})
_COLOR_OPENCV = {
    "ironbow": cv2.COLORMAP_INFERNO,
    "lava": cv2.COLORMAP_HOT,
    "rainbow": cv2.COLORMAP_TURBO,
}

_lock = threading.Lock()
_luts: dict[str, np.ndarray] = {}  # id -> (256,3) BGR uint8
# lut: 对当前画面的 Y 直接查表（原图用采集 Y，模型用原始 Y 的超分结果）
# uv:  先 LUT 得到 Y'、U、V；Y' 走超分（原图则保持 Y'）；只有 U、V 双线性放大
_MODES = ("lut", "uv")
_MODE_ALIASES = {"recolor": "lut", "linear": "uv", "lut": "lut", "uv": "uv"}
_state = {
    "enabled": False,
    "palette": "ironbow",
    "mode": "lut",
    "keep_sensor_uv": False,  # 关伪彩时是否保留放大后的机芯 UV（默认中性 128）
}


def lut_dir() -> str:
    d = os.path.join(config.PROJECT_ROOT, "luts")
    os.makedirs(d, exist_ok=True)
    return d


def _bake_opencv_lut(cmap: int) -> np.ndarray:
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    colored = cv2.applyColorMap(ramp, cmap)  # (256,1,3) BGR
    return colored.reshape(256, 3).astype(np.uint8)


def _export_lut(palette_id: str, table: np.ndarray) -> None:
    base = os.path.join(lut_dir(), palette_id)
    table = np.asarray(table, dtype=np.uint8).reshape(256, 3)
    table.tofile(base + ".bin")
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "B", "G", "R"])
        for i in range(256):
            b, g, r = int(table[i, 0]), int(table[i, 1]), int(table[i, 2])
            w.writerow([i, b, g, r])


def ensure_luts() -> dict[str, np.ndarray]:
    """烘焙并落盘彩色 LUT；灰度档无 bin（逻辑生成）。返回彩色表字典。"""
    global _luts
    with _lock:
        if _luts:
            return _luts
        out = {}
        for pid, cmap in _COLOR_OPENCV.items():
            table = _bake_opencv_lut(cmap)
            _export_lut(pid, table)
            out[pid] = table
        # 灰度档也导出恒等/反相表，便于嵌入式统一加载接口
        identity = np.stack([np.arange(256, dtype=np.uint8)] * 3, axis=1)
        inverted = np.stack([255 - np.arange(256, dtype=np.uint8)] * 3, axis=1)
        _export_lut("white_hot", identity)
        _export_lut("black_hot", inverted)
        out["white_hot"] = identity
        out["black_hot"] = inverted
        _luts = out
        config.log(f"伪彩 LUT 已就绪: {lut_dir()}")
        return _luts


def normalize_palette(palette: str | None) -> str:
    p = (palette or "ironbow").strip().lower()
    p = _ALIASES.get(p, p)
    if p not in PALETTE_IDS and p not in _COLOR_OPENCV and p not in _GRAY_IDS:
        return "ironbow"
    if p == "off":
        return "white_hot"
    return p


def get_state() -> dict:
    with _lock:
        return {
            "enabled": bool(_state["enabled"]),
            "palette": _state["palette"],
            "mode": _state["mode"],
            "keep_sensor_uv": bool(_state["keep_sensor_uv"]),
            "palettes": [
                {"id": pid, "label": PALETTE_LABELS.get(pid, pid)}
                for pid in ("off", "white_hot", "black_hot", "ironbow", "lava", "rainbow")
            ],
            "lut_dir": lut_dir(),
        }


def set_state(enabled=None, palette=None, keep_sensor_uv=None, mode=None) -> dict:
    with _lock:
        if enabled is not None:
            _state["enabled"] = bool(enabled)
        if palette is not None:
            _state["palette"] = normalize_palette(palette)
        if keep_sensor_uv is not None:
            _state["keep_sensor_uv"] = bool(keep_sensor_uv)
        if mode is not None:
            m = _MODE_ALIASES.get(str(mode).strip().lower())
            if m:
                _state["mode"] = m
    return get_state()


def effective_palette() -> str:
    """当前生效档：关伪彩时强制 white_hot。"""
    st = get_state()
    if not st["enabled"]:
        return "white_hot"
    return normalize_palette(st["palette"])


def apply_to_bgr(y: np.ndarray, palette: str | None = None) -> np.ndarray:
    """Y (H,W) uint8 → 预览用 BGR。"""
    ensure_luts()
    pid = normalize_palette(palette if palette is not None else effective_palette())
    y = np.asarray(y, dtype=np.uint8)
    if pid == "black_hot":
        y = 255 - y
        return yuv_codec.gray_to_bgr(y)
    if pid == "white_hot":
        return yuv_codec.gray_to_bgr(y)
    table = _luts[pid]  # (256,3) BGR
    return table[y]


def compose_yuy2(
    y_sr: np.ndarray,
    uv_scaled: np.ndarray | None = None,
    palette: str | None = None,
    keep_sensor_uv: bool | None = None,
) -> np.ndarray:
    """板端同构输出：Y_SR + 伪彩分支 → YUY2 缓冲。

    - white_hot / black_hot：灰度 Y（可反相）+ UV=128，或 keep_sensor_uv 时用放大 UV
    - 彩色档：LUT(Y)→BGR→YUY2（丢弃机芯 UV）
    """
    ensure_luts()
    st = get_state()
    pid = normalize_palette(palette if palette is not None else effective_palette())
    keep = st["keep_sensor_uv"] if keep_sensor_uv is None else bool(keep_sensor_uv)
    y = np.asarray(y_sr, dtype=np.uint8)

    if pid in _GRAY_IDS:
        if pid == "black_hot":
            y = 255 - y
        h, w = y.shape
        if keep and uv_scaled is not None and uv_scaled.shape[:2] == (h, w):
            u = uv_scaled[:, :, 0]
            v = uv_scaled[:, :, 1]
            return yuv_codec.pack_yuy2(y, u, v)
        return yuv_codec.pack_yuy2_neutral(y)

    bgr = apply_to_bgr(y, pid)
    return yuv_codec.bgr_to_yuy2(bgr)


def lut_to_yuv(y_src: np.ndarray, palette: str | None = None):
    """先 LUT，再拆成伪彩自己的 Y'、U、V。不做缩放。"""
    ensure_luts()
    pid = normalize_palette(palette if palette is not None else effective_palette())
    y_src = np.asarray(y_src, dtype=np.uint8)
    if y_src.ndim != 2:
        raise ValueError("lut_to_yuv 需要单通道 Y")
    if pid == "black_hot":
        y_fc = 255 - y_src
        u = np.full_like(y_fc, 128)
        v = np.full_like(y_fc, 128)
        return y_fc, u, v
    if pid == "white_hot":
        y_fc = y_src.copy()
        u = np.full_like(y_fc, 128)
        v = np.full_like(y_fc, 128)
        return y_fc, u, v
    return yuv_codec.bgr_to_yuv_planes(apply_to_bgr(y_src, pid))


def attach_uv(y_hr: np.ndarray, u: np.ndarray, v: np.ndarray):
    """Y 保持超分结果，只把 U、V 双线性对齐到 Y 的尺寸。"""
    y_hr = np.asarray(y_hr, dtype=np.uint8)
    h, w = y_hr.shape[:2]
    if w % 2 != 0:
        raise ValueError(f"YUY2 要求宽度为偶数，收到 W={w}")
    if (u.shape[0], u.shape[1]) != (h, w):
        u = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)
    bgr = yuv_codec.yuv_planes_to_bgr(y_hr, u, v)
    return bgr, yuv_codec.pack_yuy2(y_hr, u, v)


def compose_uv(y_src: np.ndarray, out_hw: tuple[int, int], palette: str | None = None):
    """无超分时的 UV 通路：Y' 保持 LUT 分辨率，不双线性放大 Y。

    out_hw 仅用于校验；与 y_src 不一致时只放大 UV，Y' 尺寸必须已经是目标尺寸。
    原图面板走这条（Y' 与采集分辨率相同）。
    """
    y_fc, u, v = lut_to_yuv(y_src, palette)
    th, tw = int(out_hw[0]), int(out_hw[1])
    if (y_fc.shape[0], y_fc.shape[1]) != (th, tw):
        raise ValueError("UV插值不会双线性放大 Y；请先对 Y' 超分再调用 attach_uv")
    return attach_uv(y_fc, u, v)


def compose_linear(y_lq: np.ndarray, y_sr: np.ndarray, palette: str | None = None):
    """兼容旧调用：Y 用已经超分好的 y_sr，UV 从 y_lq 的 LUT 线性放大。"""
    _y_fc, u, v = lut_to_yuv(y_lq, palette)
    return attach_uv(y_sr, u, v)


def list_lut_files() -> list[dict]:
    ensure_luts()
    files = []
    for pid in ("white_hot", "black_hot", "ironbow", "lava", "rainbow"):
        bin_path = os.path.join(lut_dir(), f"{pid}.bin")
        csv_path = os.path.join(lut_dir(), f"{pid}.csv")
        files.append({
            "id": pid,
            "label": PALETTE_LABELS.get(pid, pid),
            "bin": os.path.join("luts", f"{pid}.bin") if os.path.isfile(bin_path) else None,
            "csv": os.path.join("luts", f"{pid}.csv") if os.path.isfile(csv_path) else None,
        })
    return files
