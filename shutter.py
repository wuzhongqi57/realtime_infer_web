"""机芯手动快门校正（FFC）— 调用 Thermal Cam AC020 SDK。

调用链与上位机一致：
  ir_control_handle_create → iruvc_usb_handle_create
  → ircmd_create_handle → adv_device_open/init
  → basic_ffc_update（合上挡板做一次 KB/FFC）

-205：命令已下发，机芯忙于挡板动作时状态回读超时。上位机只记 warning，按成功处理。
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from ctypes import POINTER, c_int, c_int32, c_uint16, c_uint32, c_void_p

import config

_lock = threading.Lock()
_dll_lock = threading.Lock()
_dlls = None  # (cam, uvc, cmd)


class UsbOpenParam(ctypes.Structure):
    """adv_device_open 的匹配结构（VID/PID/序号）。"""

    _fields_ = [
        ("vid", c_uint32),
        ("pid", c_uint32),
        ("same_idx", c_int32),
    ]


class ShutterError(RuntimeError):
    pass


def _err_text(rc: int) -> str:
    known = {
        0: "ok",
        -100: "参数为空",
        -101: "分配失败",
        -103: "打开设备失败",
        -104: "枚举/初始化失败",
        -107: "找不到匹配设备",
        -200: "ircmd 参数无效",
        -205: "命令已下发，状态回读忙/超时",
        -600: "iruvc 参数无效",
    }
    return known.get(rc, f"未知错误码 {rc}")


# 打快门时机芯正忙，standard_cmd 回读常返回 -205
_FFC_OK_CODES = {0, -205}


def _bind(fn, restype, *argtypes):
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


def _load_dlls():
    global _dlls
    with _dll_lock:
        if _dlls is not None:
            return _dlls
        sdk = config.IRS_SDK_DIR
        if not os.path.isdir(sdk):
            raise ShutterError(f"找不到机芯 SDK 目录: {sdk}（可用 --sdk-dir 指定）")
        needed = ["pthreadVC2.dll", "libircam.dll", "libiruvc.dll", "libircmd.dll"]
        missing = [n for n in needed if not os.path.isfile(os.path.join(sdk, n))]
        if missing:
            raise ShutterError(f"SDK 目录缺文件: {missing}")

        os.add_dll_directory(os.path.abspath(sdk))
        ctypes.WinDLL(os.path.join(sdk, "pthreadVC2.dll"))
        cam = ctypes.WinDLL(os.path.join(sdk, "libircam.dll"))
        uvc = ctypes.WinDLL(os.path.join(sdk, "libiruvc.dll"))
        cmd = ctypes.WinDLL(os.path.join(sdk, "libircmd.dll"))

        _bind(cam.ir_control_handle_create, c_int, POINTER(c_void_p))
        _bind(cam.ir_control_handle_delete, c_int, POINTER(c_void_p))
        _bind(uvc.iruvc_usb_handle_create, c_void_p, c_void_p)
        _bind(uvc.iruvc_usb_handle_delete, c_int, c_void_p)
        _bind(cmd.ircmd_create_handle, c_void_p, c_void_p)
        _bind(cmd.ircmd_delete_handle, c_int, c_void_p)
        _bind(cmd.adv_device_open, c_int, c_void_p, POINTER(UsbOpenParam))
        _bind(cmd.adv_device_init, c_int, c_void_p)
        _bind(cmd.adv_device_close, c_int, c_void_p)
        _bind(cmd.basic_ffc_update, c_int, c_void_p)
        _bind(cmd.vdcmd_set_polling_wait_time, c_int, c_void_p, c_uint16)

        _dlls = (cam, uvc, cmd)
        config.log(f"机芯 SDK 已加载: {sdk}")
        return _dlls


def _check(rc: int, step: str) -> None:
    if rc != 0:
        raise ShutterError(f"{step} 失败: {_err_text(rc)}")


def _check_ffc(rc: int, step: str) -> None:
    if rc in _FFC_OK_CODES:
        if rc != 0:
            config.log(f"{step} 返回 {rc}（{_err_text(rc)}），按成功处理")
        return
    raise ShutterError(f"{step} 失败: {_err_text(rc)}")


def sdk_available() -> tuple[bool, str]:
    try:
        _load_dlls()
        return True, config.IRS_SDK_DIR
    except Exception as exc:
        return False, str(exc)


def ffc_update(vid: int | None = None, pid: int | None = None) -> None:
    """打开命令通道，执行一次手动快门校正，然后关闭通道。"""
    with _lock:
        _ffc_update_locked(vid, pid)


def _safe_call(step: str, fn, *args) -> None:
    try:
        fn(*args)
    except Exception as exc:
        config.log(f"{step} 释放异常（已忽略）: {exc}")


def _ffc_update_locked(vid: int | None, pid: int | None) -> None:
    cam, uvc, cmd = _load_dlls()
    vid = int(vid if vid is not None else config.IR_USB_VID)
    pid = int(pid if pid is not None else config.IR_USB_PID)

    ctl = c_void_p()
    usb = None
    ircmd = None
    opened = False
    try:
        _check(cam.ir_control_handle_create(ctypes.byref(ctl)), "ir_control_handle_create")
        if not ctl.value:
            raise ShutterError("ir_control_handle_create 返回空句柄")

        usb = uvc.iruvc_usb_handle_create(ctl)
        if not usb:
            raise ShutterError("iruvc_usb_handle_create 失败")

        ircmd = cmd.ircmd_create_handle(ctl)
        if not ircmd:
            raise ShutterError("ircmd_create_handle 失败")

        param = UsbOpenParam(vid=vid, pid=pid, same_idx=0)
        rc = cmd.adv_device_open(ircmd, ctypes.byref(param))
        if rc not in _FFC_OK_CODES:
            param = UsbOpenParam(vid=pid, pid=vid, same_idx=0)
            rc = cmd.adv_device_open(ircmd, ctypes.byref(param))
        _check_ffc(rc, "adv_device_open")
        opened = True

        cmd.vdcmd_set_polling_wait_time(ircmd, 3000)
        _check_ffc(cmd.adv_device_init(ircmd), "adv_device_init")
        _check_ffc(cmd.basic_ffc_update(ircmd), "basic_ffc_update")
        time.sleep(max(0.0, float(config.FFC_SETTLE_S)))
        config.log("快门校正完成")
    finally:
        if ircmd:
            if opened:
                _safe_call("adv_device_close", cmd.adv_device_close, ircmd)
            _safe_call("ircmd_delete_handle", cmd.ircmd_delete_handle, ircmd)
        if usb:
            _safe_call("iruvc_usb_handle_delete", uvc.iruvc_usb_handle_delete, usb)
        if ctl.value:
            _safe_call("ir_control_handle_delete", cam.ir_control_handle_delete, ctypes.byref(ctl))
