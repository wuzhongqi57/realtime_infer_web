"""推理引擎 — 多模型注册表 + 自动配置加载。

从 checkpoint 自动推导架构参数：
  - first_dim = conv_first.0.weight.shape[0]
  - scale     = round(sqrt(conv_last.weight.shape[0]))

跨分辨率支持：
  - 按文件名关键字识别模型期望的输入尺寸（见 config.MODEL_INPUT_SIZES）
  - 推理前自动将相机帧缩放到模型期望的输入尺寸（降采样用 INTER_AREA，升采样用 INTER_LINEAR）
  - 所有模型均可选择，无论相机分辨率如何

多面板共享：同一模型同一帧序号只推理一次，结果缓存在 _results。
"""

import glob
import os
import threading
import time

import cv2
import numpy as np
import torch

import config
from model import IE_Net


# ---------------------------------------------------------------------------
# 发现与加载
# ---------------------------------------------------------------------------
def discover_models(models_dir: str) -> list[dict]:
    """扫描目录下所有 .pth，返回 [{name, path}]（name = 文件名去扩展名）。"""
    if not os.path.isdir(models_dir):
        return []
    return [{"name": os.path.splitext(os.path.basename(p))[0], "path": p}
            for p in sorted(glob.glob(os.path.join(models_dir, "*.pth")))]


def resolve_input_size(name: str) -> tuple[int, int]:
    """按模型名关键字返回期望输入尺寸 (W,H)，未知系列用默认值。"""
    for key, size in config.MODEL_INPUT_SIZES:
        if key in name:
            return size
    return config.DEFAULT_INPUT_SIZE


def matched_output_size(models_dir: str, cam_size: tuple[int, int]) -> tuple[int, int]:
    """给定相机尺寸，返回面板固定尺寸 (W,H)。

    跨分辨率支持：面板尺寸 = 相机尺寸 × 所有模型中的最大 scale。
    这样原图（最近邻放大）和模型输出（平滑缩放到面板）可以在同一画布并排对比。
    cam_size=(0,0) 或无模型时返回 (0,0)。
    """
    w, h = cam_size
    if not w:
        return (0, 0)
    max_scale = 0
    for info in discover_models(models_dir):
        try:
            sd = _load_state(info["path"])
            scale = int(round(sd["conv_last.weight"].shape[0] ** 0.5))
            max_scale = max(max_scale, scale)
        except Exception:
            continue
    if not max_scale:
        return (0, 0)
    return w * max_scale, h * max_scale


def _load_state(model_path: str) -> dict:
    """加载 state_dict。pre_* 权重含 numpy 标量，weights_only=True 会失败，需回退。"""
    try:
        sd = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    return sd


def build_model(model_path: str) -> tuple[IE_Net, int, int]:
    """按 checkpoint 自动配置并加载 IE_Net，返回 (model, first_dim, scale)。"""
    sd = _load_state(model_path)
    first_dim = int(sd["conv_first.0.weight"].shape[0])
    scale = int(round(sd["conv_last.weight"].shape[0] ** 0.5))

    net = IE_Net(scale_factor=scale, first_dim=first_dim)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"架构不匹配 ({os.path.basename(model_path)}): missing={missing}")

    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net, first_dim, scale


# ---------------------------------------------------------------------------
# 模型注册表
# ---------------------------------------------------------------------------
class ModelHub:
    def __init__(self, models_dir: str, device: str = "cpu"):
        self.models_dir = models_dir
        self.device = torch.device(device)
        self.infos = []         # [{name, path}]，由 rescan() 填充
        self._engines = {}      # name -> (model, scale)
        self._results = {}      # name -> (seq, sr_uint8, ms)  按帧缓存，多面板共享
        self._mtimes = {}       # name -> mtime，用于判断权重是否被替换
        self._lock = threading.Lock()
        self.rescan()

    @property
    def names(self) -> list[str]:
        return [i["name"] for i in self.infos]

    def rescan(self) -> bool:
        """重新扫描模型目录；移除/替换的权重会丢掉对应引擎缓存。

        返回 True 表示列表或文件有变化（调用方可据此刷新面板尺寸等）。
        """
        new_infos = discover_models(self.models_dir)
        new_map = {i["name"]: i["path"] for i in new_infos}
        new_mtimes = {}
        for name, path in new_map.items():
            try:
                new_mtimes[name] = os.path.getmtime(path)
            except OSError:
                new_mtimes[name] = -1.0

        with self._lock:
            old_map = {i["name"]: i["path"] for i in self.infos}
            changed = (set(new_map) != set(old_map)
                       or any(new_map.get(n) != old_map.get(n) for n in new_map)
                       or any(new_mtimes.get(n) != self._mtimes.get(n) for n in new_map))

            for name in list(self._engines.keys()):
                if (name not in new_map
                        or new_map[name] != old_map.get(name)
                        or new_mtimes.get(name) != self._mtimes.get(name)):
                    self._engines.pop(name, None)
                    self._results.pop(name, None)

            self.infos = new_infos
            self._mtimes = new_mtimes

        if changed:
            config.log(f"模型目录已刷新: {self.names or '(空)'}")
        return changed

    def _get_engine(self, name: str) -> tuple[IE_Net, int]:
        with self._lock:
            if name not in self._engines:
                path = next(i["path"] for i in self.infos if i["name"] == name)
                model, first_dim, scale = build_model(path)
                model = model.to(self.device)   # 移到推理设备（GPU 必需）
                # 按该模型期望输入尺寸预热，消除首次推理开销
                input_w, input_h = resolve_input_size(name)
                dummy = torch.randn(1, 1, input_h, input_w, device=self.device)
                with torch.no_grad():
                    model(dummy)
                self._engines[name] = (model, scale)
                n = sum(p.numel() for p in model.parameters())
                config.log(f"模型已加载: {name} (dim={first_dim}, x{scale}, "
                           f"输入 {input_w}x{input_h}, {n/1e6:.3f}M, {self.device})")
            return self._engines[name]

    def infer(self, name: str, frame: np.ndarray, seq: int) -> tuple[np.ndarray, float]:
        """推理一帧。自动把帧缩放到模型期望输入尺寸。返回 (sr_uint8, ms)。"""
        with self._lock:
            hit = self._results.get(name)
            if hit is not None and hit[0] == seq:
                return hit[1], hit[2]

        model, scale = self._get_engine(name)
        input_w, input_h = resolve_input_size(name)
        if frame.shape != (input_h, input_w):
            # 降采样用 INTER_AREA 保质量，升采样用 INTER_LINEAR
            interp = cv2.INTER_AREA if frame.shape[0] >= input_h else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (input_w, input_h), interpolation=interp)

        tensor = torch.from_numpy(frame.astype(np.float32) / 255.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(tensor)
        ms = (time.perf_counter() - t0) * 1000.0
        sr = out.squeeze().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()

        with self._lock:
            self._results[name] = (seq, sr, ms)
        return sr, ms
