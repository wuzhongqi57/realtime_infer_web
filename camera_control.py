"""可选的本机相机控制插件加载器。

只加载 `plugins/local.py`（不扫描 example_stub 或其他文件）。
无插件时所有能力接口返回不可用，核心采集/推理不受影响。
"""

from __future__ import annotations

import importlib.util
import os

import config

_PLUGIN_PATH = os.path.join(config.PROJECT_ROOT, "plugins", "local.py")
_mod = None
_loaded = False
_unavailable_logged = False


class CameraControlError(RuntimeError):
    pass


def _load():
    global _mod, _loaded
    if _loaded:
        return _mod
    _loaded = True
    if not os.path.isfile(_PLUGIN_PATH):
        return None
    try:
        spec = importlib.util.spec_from_file_location("ir_camera_plugin", _PLUGIN_PATH)
        if spec is None or spec.loader is None:
            config.log("相机控制插件无法加载: spec 为空")
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not callable(getattr(mod, "available", None)) or not callable(getattr(mod, "ffc_update", None)):
            config.log("相机控制插件缺少 available() / ffc_update()")
            return None
        _mod = mod
        config.log("已加载相机控制插件: plugins/local.py")
        return _mod
    except Exception as exc:
        config.log(f"相机控制插件加载失败: {exc}")
        return None


def _log_unavailable(detail: str) -> None:
    global _unavailable_logged
    if _unavailable_logged:
        return
    _unavailable_logged = True
    config.log(f"相机控制插件不可用: {detail}")


def plugin_present() -> bool:
    return _load() is not None


def ffc_capable() -> bool:
    mod = _load()
    if mod is None:
        return False
    try:
        ok, detail = mod.available()
        if not ok:
            _log_unavailable(str(detail))
            return False
        return True
    except Exception as exc:
        _log_unavailable(str(exc))
        return False


def needs_device_release() -> bool:
    mod = _load()
    if mod is None:
        return True
    return bool(getattr(mod, "NEEDS_DEVICE_RELEASE", True))


def ffc_update() -> None:
    mod = _load()
    if mod is None:
        raise CameraControlError("未安装相机控制插件")
    try:
        ok, detail = mod.available()
    except Exception as exc:
        raise CameraControlError(str(exc)) from exc
    if not ok:
        raise CameraControlError(str(detail))
    try:
        mod.ffc_update()
    except CameraControlError:
        raise
    except Exception as exc:
        raise CameraControlError(str(exc)) from exc
