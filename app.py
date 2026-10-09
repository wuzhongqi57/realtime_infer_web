"""本地实时推理模拟 Web 应用 — Flask + WebSocket 入口。

启动：conda run -n ir_realtime python app.py [--cam 0] [--port 5001]
打开：http://127.0.0.1:5001

架构：1～3 个可调面板，每面板独立选择显示"原图"或某个模型的效果。
  - /ws/panel：每个面板一条 WebSocket，客户端可发送 {"model": <名字>} 切换
  - /api/models：返回可选模型列表
  - 同一模型同一帧的推理结果跨面板共享（ModelHub 帧缓存）
  - 跨分辨率支持：相机帧自动缩放到模型输入尺寸再推理，SR 结果再缩放到面板尺寸
  - 面板尺寸 = 相机分辨率 × 最大放大倍数（展示框统一双线性缩放到面板尺寸）
  - 拍照/录像按当前各面板所选内容并行保存（最多 3 路，同模型去重）
  - HAT×4 为重模型：直播仅占位；拍照即时推理；录像中不推理，停止后对原图离线超分
  - YUY2 同构链路：仅 Y 超分 → UV 双线性对齐 → 可选伪彩 → 打包 YUY2 / 预览 JPEG
"""

import argparse
import json
import os
import re
import threading
import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_sock import Sock

import config
import camera_control
import false_color
import hat_backend
import yuv_codec
from capture import FrameSource, SharedState, list_cameras
from engine import ModelHub, matched_output_size, resolve_input_size

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.after_request
def _no_cache(resp):
    """禁止缓存动态页面/API（防止前端 fetch 拿到旧状态）"""
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp   # 模板热重载：修改 index.html 后无需重启
sock = Sock(app)

state = SharedState()
hub = None
source = None          # 全局视频源（拍照/录像/相机控制 API 使用）
_cached_cam_key = None         # 面板尺寸按相机分辨率缓存（更换相机时才重算）
_panel_size = (0, 0)


def _get_panel_size() -> tuple[int, int]:
    """当前相机对应的面板固定尺寸 = 匹配模型的 SR 输出尺寸。"""
    global _cached_cam_key, _panel_size
    key = tuple(state.cam_size)
    if key != _cached_cam_key:
        _cached_cam_key = key
        _panel_size = matched_output_size(hub.models_dir, key) if hub else (0, 0)
    return _panel_size


def _rescan_models() -> bool:
    """重新扫描模型目录；有变化时作废面板尺寸缓存。"""
    global _cached_cam_key
    if hub is None:
        return False
    changed = hub.rescan()
    if changed:
        _cached_cam_key = None
    return changed


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _selectable_names() -> list[str]:
    """原图以外的可选名：UNet .pth + 可用时的 HAT。"""
    names = list(hub.names) if hub else []
    if hat_backend.available()[0]:
        names.append(config.HAT_LABEL)
    return names


def _heavy_models() -> list[str]:
    return [config.HAT_LABEL] if hat_backend.available()[0] else []


def _is_heavy(name: str) -> bool:
    return hat_backend.is_heavy(name)


def _infer_export(name: str, frame: np.ndarray, seq: int) -> np.ndarray:
    """拍照/录像用：含 HAT 真推理。返回单通道 Y（SR 或原图）。"""
    if name == config.RAW_LABEL:
        return frame
    if _is_heavy(name):
        return hat_backend.infer(frame)
    sr, _ = hub.infer(name, frame, seq)
    return sr


def _track_size(model: str, cam_w: int, cam_h: int) -> tuple[int, int]:
    if model == config.RAW_LABEL:
        return cam_w, cam_h
    if _is_heavy(model):
        s = hat_backend.scale()
        return cam_w * s, cam_h * s
    _, scale = hub._get_engine(model)
    input_w, input_h = resolve_input_size(model)
    return input_w * scale, input_h * scale


def _resize_uv_to(uv: np.ndarray | None, out_h: int, out_w: int) -> np.ndarray:
    """将 UV (H,W,2) 双线性对齐到 Y_SR 尺寸。"""
    if uv is None:
        return np.full((out_h, out_w, 2), 128, dtype=np.uint8)
    if uv.shape[0] == out_h and uv.shape[1] == out_w:
        return uv
    return cv2.resize(uv, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def _normalize_mode(mode: str | None) -> str:
    m = false_color._MODE_ALIASES.get(str(mode or "lut").strip().lower(), "lut")
    return m


def _resolve_fc(spec: dict | None) -> dict:
    """单面板伪彩。未传时回退到全局设置（兼容旧请求）。"""
    if not isinstance(spec, dict) or not spec:
        st = false_color.get_state()
        enabled = bool(st["enabled"])
        palette = false_color.normalize_palette(st["palette"])
        mode = _normalize_mode(st.get("mode"))
    else:
        enabled = bool(spec.get("enabled"))
        palette = false_color.normalize_palette(spec.get("palette") or "ironbow")
        mode = _normalize_mode(spec.get("mode"))
    return {
        "enabled": enabled,
        "palette": palette,
        "mode": mode,
        "effective": palette if enabled else "white_hot",
    }


def _compose_export(y_sr: np.ndarray, uv_lq: np.ndarray | None,
                    y_lq: np.ndarray | None = None,
                    fc: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """返回 (预览BGR, YUY2)。

    lut：对 y_sr 直接查表。
    uv：先对 y_lq 做 LUT，再把伪彩 Y'UV' 缩放到 y_sr 的尺寸。
    """
    view = _resolve_fc(fc)
    pal = view["effective"]
    h, w = y_sr.shape[:2]
    uv_hq = _resize_uv_to(uv_lq, h, w)
    yuy2 = false_color.compose_yuy2(y_sr, uv_hq, palette=pal)
    if pal not in ("white_hot", "black_hot"):
        bgr = yuv_codec.yuy2_to_bgr(yuy2, w, h)
    elif pal == "black_hot":
        bgr = yuv_codec.gray_to_bgr(255 - y_sr)
    else:
        bgr = yuv_codec.gray_to_bgr(y_sr)
    return bgr, yuy2


def _jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def _render(name: str, frame: np.ndarray, seq: int,
            hat_hold: dict | None = None, fc: dict | None = None) -> bytes:
    """按面板选择渲染一帧为 JPEG。伪彩设置来自该面板，不共用全局。"""
    panel_w, panel_h = _get_panel_size()
    view = _resolve_fc(fc)
    pal = view["effective"]

    def _fit(bgr: np.ndarray, nearest: bool = False) -> np.ndarray:
        if panel_h and (bgr.shape[0], bgr.shape[1]) != (panel_h, panel_w):
            interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
            bgr = cv2.resize(bgr, (panel_w, panel_h), interpolation=interp)
        return bgr

    if name == config.RAW_LABEL or (view["mode"] == "uv" and not _is_heavy(name)):
        if view["mode"] == "uv":
            y_fc, u, v = false_color.lut_to_yuv(frame, pal)
            if name == config.RAW_LABEL:
                y_hr = y_fc
            else:
                y_hr, _ = hub.infer(name, y_fc, seq, cache_tag="uv:" + pal)
            bgr, _ = false_color.attach_uv(y_hr, u, v)
        else:
            bgr = false_color.apply_to_bgr(frame, pal)
        # 原图放大到面板用最近邻，避免双线性抹掉像素块
        return _jpeg(_fit(bgr, nearest=(name == config.RAW_LABEL)))
    if _is_heavy(name):
        if hat_hold is not None:
            if hat_hold.get("jpeg") is None:
                ph = hat_backend.make_placeholder(frame, (panel_w, panel_h))
                if ph.ndim == 2:
                    ph = false_color.apply_to_bgr(ph, pal)
                hat_hold["jpeg"] = _jpeg(ph)
            return hat_hold["jpeg"] or b""
        ph = hat_backend.make_placeholder(frame, (panel_w, panel_h))
        if ph.ndim == 2:
            ph = false_color.apply_to_bgr(ph, pal)
        return _jpeg(ph)
    sr, _ = hub.infer(name, frame, seq)
    bgr = false_color.apply_to_bgr(sr, pal)
    return _jpeg(_fit(bgr))


# ---------------------------------------------------------------------------
# WebSocket — 单一面板流（二进制 JPEG 帧）
# ---------------------------------------------------------------------------
@sock.route("/ws/panel")
def ws_panel(ws):
    selected = config.RAW_LABEL
    last_seq = 0
    fc_spec = {"enabled": False, "palette": "ironbow", "mode": "lut"}
    hat_hold = {"jpeg": None, "palette": None}
    while ws.connected:
        msg = ws.receive(timeout=0.02)
        if msg:
            try:
                d = json.loads(msg)
                allowed = [config.RAW_LABEL] + _selectable_names()
                if d.get("model") in allowed:
                    new_sel = d["model"]
                    if new_sel != selected:
                        selected = new_sel
                        hat_hold = {"jpeg": None, "palette": None}
                        last_seq = 0
                if isinstance(d.get("false_color"), dict):
                    fc_spec = d["false_color"]
                    hat_hold = {"jpeg": None, "palette": None}
                    last_seq = 0
            except Exception:
                pass

        frame, seq = state.get_raw()
        if frame is None:
            continue
        view = _resolve_fc(fc_spec)
        pal_key = f"{view['enabled']}:{view['effective']}:{view['mode']}"
        if hat_hold.get("palette") != pal_key:
            hat_hold = {"jpeg": None, "palette": pal_key}
            last_seq = 0

        if _is_heavy(selected) and hat_hold.get("jpeg") is not None:
            if seq == last_seq:
                continue
            last_seq = seq
            try:
                ws.send(hat_hold["jpeg"])
            except Exception:
                break
            continue
        if seq == last_seq:
            continue
        last_seq = seq
        try:
            data = _render(selected, frame, seq, hat_hold=hat_hold, fc=fc_spec)
            if hat_hold is not None:
                hat_hold["palette"] = pal_key
        except Exception as exc:
            config.log(f"渲染失败 ({selected}): {exc}")
            continue
        if data:
            try:
                ws.send(data)
            except Exception:
                break


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


def matching_models(cam_size: tuple | None = None) -> list[str]:
    """可选模型。cam_size 给定时只保留输入尺寸与相机一致的系列；HAT 始终保留。"""
    names = _selectable_names()
    if not cam_size or not cam_size[0]:
        return names
    cam_w, cam_h = int(cam_size[0]), int(cam_size[1])
    out = []
    for name in names:
        if _is_heavy(name):
            out.append(name)
            continue
        iw, ih = resolve_input_size(name)
        if (iw, ih) == (cam_w, cam_h):
            out.append(name)
    return out


@app.route("/api/models")
def api_models():
    # 每次拉取都重新扫描目录（页面刷新等），无需重启服务
    _rescan_models()
    # 等待相机分辨率确定（最多 5s），据此过滤匹配模型
    for _ in range(50):
        if state.cam_size[0]:
            break
        time.sleep(0.1)
    heavy = _heavy_models()
    all_names = _selectable_names()
    matched = matching_models(state.cam_size)
    return jsonify({
        "models": matched,
        "all_models": all_names,
        "matched_models": matched,
        "filter_by_camera": True,
        "heavy_models": heavy,
        "model_meta": {n: {"heavy": True, "offline_only": True} for n in heavy},
        "camera_size": list(state.cam_size),
        "panel_size": list(_get_panel_size()),
        "false_color": false_color.get_state(),
    })


@app.route("/api/status")
def api_status():
    raw, raw_seq = state.get_raw()
    heavy = _heavy_models()
    fc = false_color.get_state()
    return jsonify({
        "running": state.running,
        "raw_seq": raw_seq,
        "raw_fps": round(state.raw_fps, 1),
        "camera": raw is not None,
        "camera_size": list(state.cam_size),
        "panel_size": list(_get_panel_size()),
        "models": matching_models(state.cam_size),
        "all_models": _selectable_names(),
        "matched_models": matching_models(state.cam_size),
        "heavy_models": heavy,
        "ffc": camera_control.ffc_capable(),
        "false_color": fc,
    })


@app.route("/api/false_color", methods=["GET", "POST"])
def api_false_color():
    """读写伪彩开关与调色板。POST: {enabled, palette, keep_sensor_uv}"""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        st = false_color.set_state(
            enabled=data.get("enabled"),
            palette=data.get("palette"),
            keep_sensor_uv=data.get("keep_sensor_uv"),
            mode=data.get("mode"),
        )
        return jsonify({"status": "ok", **st})
    return jsonify({"status": "ok", **false_color.get_state()})


@app.route("/api/luts")
def api_luts():
    false_color.ensure_luts()
    return jsonify({
        "status": "ok",
        "lut_dir": false_color.lut_dir(),
        "files": false_color.list_lut_files(),
        "palettes": false_color.get_state()["palettes"],
    })


# ---------------------------------------------------------------------------
# 拍照 / 录像 / 相机控制
# ---------------------------------------------------------------------------
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "snapshots")
RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "recordings")
_recorders = {}          # 同步录像会话：{"session", "thread", "models"}；空 = 未在录
_hat_jobs = {}           # job_id -> 离线 HAT 后处理任务
_hat_jobs_lock = threading.Lock()


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_model_name(model: str) -> str:
    """模型名转安全文件名（全英文）：原图 → raw，其余去特殊字符。"""
    if model == config.RAW_LABEL:
        return "raw"
    if _is_heavy(model):
        return "hat_x4"
    return re.sub(r"[^\w.-]+", "_", model) or "model"


@app.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    """保存当前帧：Y→(SR)→伪彩预览图，可选同帧 YUY2。

    Request: {"model", "format": "png"|"jpg", "export_yuy2": true}
    """
    data = request.get_json(silent=True) or {}
    model = data.get("model", config.RAW_LABEL)
    fmt = (data.get("format") or "png").lower()
    if fmt not in ("png", "jpg", "jpeg"):
        fmt = "png"
    ext = "jpg" if fmt in ("jpg", "jpeg") else "png"
    export_yuy2 = data.get("export_yuy2", True)
    frame, uv, seq = state.get_raw_yuv()
    if frame is None:
        return jsonify({"status": "error", "message": "当前无帧（相机未连接）"}), 400
    try:
        view = _resolve_fc(data.get("false_color"))
        pal = view["effective"]
        if view["mode"] == "uv":
            y_fc, u, v = false_color.lut_to_yuv(frame, pal)
            if model == config.RAW_LABEL:
                y_out = y_fc
            elif _is_heavy(model):
                config.log(f"拍照 HAT 推理中… ({frame.shape[1]}x{frame.shape[0]})")
                y_out = hat_backend.infer(y_fc)
            else:
                y_out, _ = hub.infer(model, y_fc, seq, cache_tag="uv:" + pal)
            bgr, yuy2 = false_color.attach_uv(y_out, u, v)
        else:
            if _is_heavy(model):
                config.log(f"拍照 HAT 推理中… ({frame.shape[1]}x{frame.shape[0]})")
            y_out = _infer_export(model, frame, seq)
            bgr, yuy2 = _compose_export(y_out, uv, frame, data.get("false_color"))
    except Exception as exc:
        return jsonify({"status": "error", "message": f"渲染失败: {exc}"}), 400
    _ensure_dir(SNAPSHOT_DIR)
    tag = "gray" if not view["enabled"] else f"{view['mode']}_{view['palette']}"
    ts = _timestamp()
    base = f"{ts}_{_safe_model_name(model)}_{tag}"
    fn = f"{base}.{ext}"
    path = os.path.join(SNAPSHOT_DIR, fn)
    cv2.imwrite(path, bgr)
    yuy2_rel = None
    if export_yuy2:
        yuy2_fn = f"{base}.yuy2"
        yuy2_path = os.path.join(SNAPSHOT_DIR, yuy2_fn)
        with open(yuy2_path, "wb") as f:
            f.write(np.asarray(yuy2, dtype=np.uint8).tobytes())
        # sidecar 元数据
        meta = {
            "width": int(bgr.shape[1]),
            "height": int(bgr.shape[0]),
            "format": "YUY2",
            "palette": view["effective"],
            "false_color_mode": view["mode"],
            "false_color_enabled": view["enabled"],
            "model": model,
        }
        with open(os.path.join(SNAPSHOT_DIR, f"{base}.yuy2.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        yuy2_rel = os.path.join("output", "snapshots", yuy2_fn)
    h, w = bgr.shape[:2]
    _std = float(np.asarray(y_out).std())
    _mean = float(np.asarray(y_out).mean())
    warn = None
    if _std < 2:
        warn = "帧接近平坦（std=%.1f），请检查相机画面" % _std
    return jsonify({
        "status": "ok",
        "file": os.path.join("output", "snapshots", fn),
        "yuy2": yuy2_rel,
        "model": model,
        "size": [h, w],
        "mean": round(_mean, 1),
        "std": round(_std, 1),
        "warn": warn,
        "heavy": _is_heavy(model),
        "palette": view["effective"],
        "false_color_mode": view["mode"],
    })


def _open_video_writer(path_no_ext: str, fmt: str, fps: int, size: tuple[int, int]):
    """打开 VideoWriter，优先用不依赖 OpenH264 的编码器。

    返回 (writer, path, fmt, fourcc_name)。全部失败返回 (None, None, None, None)。
    MP4 不用 avc1/H.264（需额外 openh264 DLL），改用 mp4v。
    """
    fmt = (fmt or "avi").lower()
    if fmt not in ("avi", "mp4"):
        fmt = "avi"
    # (容器扩展名, fourcc字符串) — 按优先级尝试
    if fmt == "mp4":
        candidates = [
            ("mp4", "mp4v"),   # MPEG-4，OpenCV 自带，无需 OpenH264
            ("avi", "MJPG"),   # 回退：改写为 AVI
        ]
    else:
        candidates = [
            ("avi", "MJPG"),
            ("avi", "XVID"),
        ]

    vw, vh = size
    for ext, code in candidates:
        path = f"{path_no_ext}.{ext}"
        fourcc = cv2.VideoWriter_fourcc(*code)
        writer = cv2.VideoWriter(path, fourcc, fps, (vw, vh), isColor=True)
        if writer.isOpened():
            if ext != fmt:
                config.log(f"录像编码器回退: 请求 {fmt} → 实际 {ext}/{code}")
            else:
                config.log(f"录像编码器: {ext}/{code}")
            return writer, path, ext, code
        writer.release()
        try:
            if os.path.isfile(path) and os.path.getsize(path) == 0:
                os.remove(path)
        except OSError:
            pass
    return None, None, None, None


@app.route("/api/recording/start", methods=["POST"])
def api_recording_start():
    """开始录像：轻量模型同步写入同一相机帧序号。

    HAT 等重模型不参与实时录制；停止后再对「原图」录像做离线超分。
    Request: {"models": ["原图", "model_a", "HAT×4"], "fps": 帧率, "format": "avi"|"mp4"}
    """
    global _recorders
    if _recorders:
        return jsonify({"status": "error", "message": "录像已在进行中"}), 400
    data = request.get_json(silent=True) or {}
    fps = int(data.get("fps", 15))
    fmt = (data.get("format") or "avi").lower()
    export_yuy2 = bool(data.get("export_yuy2", True))
    if fmt not in ("avi", "mp4"):
        fmt = "avi"
    valid = [config.RAW_LABEL] + _selectable_names()
    jobs = []
    raw_panels = data.get("panels")
    if isinstance(raw_panels, list) and raw_panels:
        for item in raw_panels[:3]:
            if isinstance(item, dict) and item.get("model") in valid:
                fc = item.get("false_color") if isinstance(item.get("false_color"), dict) else {}
                jobs.append({"model": item["model"], "fc": fc})
    else:
        models_in = data.get("models") or [data.get("model", config.RAW_LABEL)]
        jobs = [{"model": m, "fc": {}} for m in dict.fromkeys(m for m in models_in if m in valid)][:3]
    if not jobs:
        return jsonify({"status": "error", "message": "无有效面板模型可录制"}), 400

    want_hat = any(_is_heavy(j["model"]) for j in jobs)
    live_jobs = [j for j in jobs if not _is_heavy(j["model"])]
    raw_auto = False
    if want_hat and not any(j["model"] == config.RAW_LABEL for j in live_jobs):
        live_jobs = [{"model": config.RAW_LABEL, "fc": {}}] + live_jobs
        raw_auto = True
        live_jobs = live_jobs[:3]
    if not live_jobs:
        return jsonify({"status": "error", "message": "没有可实时录制的内容（仅选了 HAT 时将自动录原图）"}), 400
    models = [j["model"] for j in jobs]

    frame, seq = state.get_raw()
    if frame is None:
        return jsonify({"status": "error", "message": "当前无帧（相机未连接）"}), 400
    cam_h, cam_w = frame.shape[:2]
    _ensure_dir(RECORDING_DIR)
    ts = _timestamp()

    if want_hat:
        config.log("录像含 HAT：实时阶段只录轻量轨；停止后对原图离线超分")

    tracks = []
    probe_model = live_jobs[0]["model"]
    probe_size = _track_size(probe_model, cam_w, cam_h)
    probe_path = os.path.join(RECORDING_DIR, f"{ts}_probe")
    probe_writer, _, used_fmt, used_fourcc = _open_video_writer(
        probe_path, fmt, fps, probe_size)
    if probe_writer is None:
        return jsonify({
            "status": "error",
            "message": "无法打开录像编码器（请改用 AVI，或安装可用的 OpenCV 视频后端）",
        }), 500
    probe_writer.release()
    try:
        probe_file = f"{probe_path}.{used_fmt}"
        if os.path.isfile(probe_file):
            os.remove(probe_file)
    except OSError:
        pass

    fourcc = cv2.VideoWriter_fourcc(*used_fourcc)
    hat_source_path = None
    live_models = []
    for i, job in enumerate(live_jobs):
        model = job["model"]
        view = _resolve_fc(job.get("fc"))
        tag = "gray" if not view["enabled"] else f"{view['mode']}_{view['palette']}"
        vw, vh = _track_size(model, cam_w, cam_h)
        path = os.path.join(
            RECORDING_DIR, f"{ts}_{i}_{_safe_model_name(model)}_{tag}.{used_fmt}")
        writer = cv2.VideoWriter(path, fourcc, fps, (vw, vh), isColor=True)
        if not writer.isOpened():
            for t in tracks:
                t["writer"].release()
            return jsonify({
                "status": "error",
                "message": f"VideoWriter 打开失败: {os.path.basename(path)}",
            }), 500
        tracks.append({
            "model": model, "writer": writer, "path": path,
            "frames": 0, "vw": vw, "vh": vh, "fc": job.get("fc") or {},
            "key": i,
        })
        live_models.append(model)
        if model == config.RAW_LABEL and hat_source_path is None:
            hat_source_path = path

    stop_evt = threading.Event()
    session = {
        "tracks": tracks,
        "fps": fps,
        "stop": stop_evt,
        "lock": threading.Lock(),
        "frames": 0,
        "last": 0,
        "done": False,
        "format": used_fmt,
        "fourcc": used_fourcc,
        "want_hat": want_hat,
        "hat_source_path": hat_source_path,
        "raw_auto": raw_auto,
        "requested": models,
        "export_yuy2": export_yuy2,
        "last_yuy2": {},
    }
    thread = threading.Thread(target=_rec_sync_loop, args=(session,), daemon=True)
    _recorders = {"session": session, "thread": thread, "models": live_models}
    thread.start()
    files = [os.path.join("output", "recordings", os.path.basename(t["path"])) for t in tracks]
    return jsonify({
        "status": "ok",
        "files": files,
        "models": live_models,
        "fps": fps,
        "format": used_fmt,
        "fourcc": used_fourcc,
        "heavy": want_hat,
        "hat_deferred": want_hat,
        "raw_auto": raw_auto,
    })


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return false_color.apply_to_bgr(img)
    return img


def _match_writer_size(img: np.ndarray, vw: int, vh: int) -> np.ndarray:
    """保证写入尺寸与 VideoWriter 一致，避免静默丢帧导致各路长度漂移。"""
    h, w = img.shape[:2]
    if (w, h) != (vw, vh):
        img = cv2.resize(img, (vw, vh), interpolation=cv2.INTER_LINEAR)
    return img


def _rec_sync_loop(session: dict) -> None:
    """单线程同步录像：同一 seq 渲染全部轻量模型后再一并写入（不含 HAT 推理）。"""
    lock = session["lock"]
    export_yuy2 = bool(session.get("export_yuy2"))
    last_yuy2 = {}  # model -> (path written once at end via session)
    while not session["stop"].is_set():
        f, uv, s = state.get_raw_yuv()
        if f is None or s == session["last"]:
            time.sleep(0.01)
            continue
        session["last"] = s
        f = f.copy()
        uv = None if uv is None else uv.copy()

        rendered = []
        ok = True
        for track in session["tracks"]:
            try:
                fc_view = _resolve_fc(track.get("fc"))
                if fc_view["mode"] == "uv":
                    pal = fc_view["effective"]
                    y_fc, u, v = false_color.lut_to_yuv(f, pal)
                    if track["model"] == config.RAW_LABEL:
                        y_out = y_fc
                    else:
                        y_out, _ = hub.infer(
                            track["model"], y_fc, s, cache_tag="uv:" + pal)
                    bgr, yuy2 = false_color.attach_uv(y_out, u, v)
                else:
                    if track["model"] == config.RAW_LABEL:
                        y_out = f
                    else:
                        y_out, _ = hub.infer(track["model"], f, s)
                    bgr, yuy2 = _compose_export(y_out, uv, f, track.get("fc"))
                bgr = _match_writer_size(bgr, track["vw"], track["vh"])
                rendered.append(bgr)
                if export_yuy2:
                    last_yuy2[track.get("key", track["model"])] = (
                        yuy2, bgr.shape[1], bgr.shape[0], y_out.shape)
            except Exception as exc:
                config.log(f"录像跳过 seq={s} model={track['model']}: {exc}")
                ok = False
                break
        if not ok or len(rendered) != len(session["tracks"]):
            continue

        try:
            for track, img in zip(session["tracks"], rendered):
                track["writer"].write(img)
        except Exception as exc:
            config.log(f"录像写入失败 seq={s}: {exc}（本序号不计入；文件可能已有多余帧）")
            continue

        with lock:
            session["frames"] += 1
            n = session["frames"]
            for track in session["tracks"]:
                track["frames"] = n
            session["last_yuy2"] = last_yuy2

    for track in session["tracks"]:
        try:
            track["writer"].release()
        except Exception as exc:
            config.log(f"释放 VideoWriter 异常: {exc}")
        config.log(f"录像已保存: {track['path']} ({track['frames']} 帧)")
    # 可选：每路落一份末帧 YUY2 便于板端对照
    if export_yuy2:
        for track in session["tracks"]:
            pack = session.get("last_yuy2", {}).get(track.get("key", track["model"]))
            if not pack:
                continue
            yuy2, _bw, _bh, yshape = pack
            yh, yw = int(yshape[0]), int(yshape[1])
            yuy2_path = os.path.splitext(track["path"])[0] + "_last.yuy2"
            with open(yuy2_path, "wb") as f:
                f.write(np.asarray(yuy2, dtype=np.uint8).tobytes())
            with open(yuy2_path + ".json", "w", encoding="utf-8") as mf:
                json.dump({
                    "width": yw,
                    "height": yh,
                    "format": "YUY2",
                    "palette": false_color.effective_palette(),
                    "false_color_mode": false_color.get_state().get("mode", "lut"),
                    "model": track["model"],
                    "note": "last frame of recording",
                }, mf, ensure_ascii=False, indent=2)
            track["yuy2"] = yuy2_path
            config.log(f"录像末帧 YUY2: {yuy2_path}")
    with lock:
        session["done"] = True


def _start_hat_post_job(src_path: str, fps: float, fmt: str, fourcc_name: str) -> dict:
    """对原图录像逐帧 HAT×4，写新文件；返回 job 字典。"""
    job_id = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}_{threading.get_ident()}"
    base = os.path.basename(src_path)
    # e.g. 20260924_120000_raw.avi -> 20260924_120000_hat_x4.avi
    if "_raw." in base:
        out_name = base.replace("_raw.", "_hat_x4.", 1)
    else:
        stem, _ext = os.path.splitext(base)
        out_name = f"{stem}_hat_x4.{fmt}"
    out_path = os.path.join(RECORDING_DIR, out_name)

    job = {
        "id": job_id,
        "status": "queued",
        "progress": 0,
        "current": 0,
        "total": 0,
        "src": src_path,
        "file": os.path.join("output", "recordings", out_name),
        "path": out_path,
        "message": "排队中",
        "error": None,
    }

    def worker():
        job["status"] = "running"
        job["message"] = "加载 HAT…"
        try:
            ok, detail = hat_backend.available()
            if not ok:
                raise RuntimeError(detail)
            # 预加载权重，避免进度条长时间停在 0
            hat_backend.infer(np.zeros((64, 64), dtype=np.uint8))
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            job["message"] = f"HAT 加载失败: {exc}"
            config.log(job["message"])
            return

        cap = cv2.VideoCapture(src_path)
        if not cap.isOpened():
            job["status"] = "error"
            job["error"] = f"无法打开源视频: {src_path}"
            job["message"] = job["error"]
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or float(fps or 15)
        job["total"] = max(total, 0)
        job["message"] = "HAT 超分中…"

        writer = None
        written = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                if bgr.ndim == 3:
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                else:
                    gray = bgr
                sr = hat_backend.infer(gray)
                sr_bgr = _to_bgr(sr)
                if writer is None:
                    h, w = sr_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                    writer = cv2.VideoWriter(out_path, fourcc, src_fps, (w, h), isColor=True)
                    if not writer.isOpened():
                        raise RuntimeError(f"无法创建 HAT 输出: {out_path}")
                writer.write(sr_bgr)
                written += 1
                job["current"] = written
                if total > 0:
                    job["progress"] = min(99, int(100.0 * written / total))
                else:
                    job["progress"] = min(99, written % 100)
            if writer is None:
                raise RuntimeError("源视频无有效帧")
            job["progress"] = 100
            job["status"] = "done"
            job["message"] = f"HAT 超分完成（{written} 帧）"
            config.log(f"HAT 离线超分完成: {out_path} ({written} 帧)")
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            job["message"] = f"HAT 超分失败: {exc}"
            config.log(job["message"])
        finally:
            cap.release()
            if writer is not None:
                writer.release()

    with _hat_jobs_lock:
        _hat_jobs[job_id] = job
    threading.Thread(target=worker, daemon=True).start()
    return job


@app.route("/api/recording/stop", methods=["POST"])
def api_recording_stop():
    """停止实时录像；若本次请求了 HAT，则启动离线超分任务。"""
    global _recorders
    if not _recorders:
        return jsonify({"status": "ok", "files": [], "synced_frames": 0})
    session = _recorders["session"]
    thread = _recorders["thread"]
    session["stop"].set()
    thread.join(timeout=60)
    if thread.is_alive():
        config.log("录像线程停止超时，仍返回当前已同步帧数")

    with session["lock"]:
        n = int(session["frames"])
        results = []
        for track in session["tracks"]:
            track["frames"] = n
            # 自动加入的原图仍返回，便于用户看到源；前端可标注
            results.append({
                "model": track["model"],
                "file": os.path.join("output", "recordings", os.path.basename(track["path"])),
                "frames": n,
                "auto_raw": bool(session.get("raw_auto") and track["model"] == config.RAW_LABEL),
                "yuy2": (
                    os.path.join("output", "recordings", os.path.basename(track["yuy2"]))
                    if track.get("yuy2") else None
                ),
            })
    want_hat = bool(session.get("want_hat"))
    hat_src = session.get("hat_source_path")
    fmt = session.get("format") or "avi"
    fourcc_name = session.get("fourcc") or "MJPG"
    fps = session.get("fps") or 15
    _recorders = {}

    post_hat = None
    if want_hat:
        if not hat_src or not os.path.isfile(hat_src):
            post_hat = {"status": "error", "message": "缺少原图录像，无法离线 HAT"}
        elif n <= 0:
            post_hat = {"status": "error", "message": "录像无帧，跳过 HAT"}
        else:
            job = _start_hat_post_job(hat_src, fps, fmt, fourcc_name)
            post_hat = {
                "status": "queued",
                "job_id": job["id"],
                "message": "已开始离线 HAT 超分",
            }

    return jsonify({
        "status": "ok",
        "files": results,
        "synced_frames": n,
        "post_hat": post_hat,
    })


@app.route("/api/recording/hat_progress", methods=["GET"])
def api_recording_hat_progress():
    """查询离线 HAT 超分进度。?job_id=..."""
    job_id = request.args.get("job_id") or ""
    with _hat_jobs_lock:
        job = _hat_jobs.get(job_id)
    if not job:
        return jsonify({"status": "error", "message": "任务不存在"}), 404
    return jsonify({
        "status": "ok",
        "job_id": job["id"],
        "job_status": job["status"],
        "progress": job["progress"],
        "current": job["current"],
        "total": job["total"],
        "message": job["message"],
        "file": job.get("file"),
        "error": job.get("error"),
    })


@app.route("/api/recording/status", methods=["GET"])
def api_recording_status():
    if _recorders:
        session = _recorders["session"]
        with session["lock"]:
            frames = int(session["frames"])
        return jsonify({"status": "ok", "recording": True,
                        "models": list(_recorders["models"]),
                        "frames": frames,
                        "hat_deferred": bool(session.get("want_hat"))})
    return jsonify({"status": "ok", "recording": False, "frames": 0})


@app.route("/api/camera/disconnect", methods=["POST"])
def api_camera_disconnect():
    """断开相机（释放设备，暂停采集）。"""
    source.pause()
    return jsonify({"status": "ok"})


@app.route("/api/camera/reconnect", methods=["POST"])
def api_camera_reconnect():
    """重新连接相机，并重新扫描模型目录。"""
    _rescan_models()
    source.resume()
    return jsonify({"status": "ok"})


@app.route("/api/camera/status", methods=["GET"])
def api_camera_status():
    running = state.cam_size[0] > 0
    paused = bool(getattr(source, "_paused", False)) if source else False
    return jsonify({"status": "ok", "connected": running,
                    "size": list(state.cam_size), "paused": paused,
                    "ffc": camera_control.ffc_capable()})


@app.route("/api/camera/ffc", methods=["POST"])
def api_camera_ffc():
    """手动快门校正：若本机安装了相机控制插件，会短暂让出 OpenCV 设备。"""
    if not camera_control.plugin_present():
        return jsonify({"status": "error", "message": "未安装相机控制插件"}), 404
    if not camera_control.ffc_capable():
        return jsonify({"status": "error", "message": "相机控制插件当前不可用"}), 400
    if source is None:
        return jsonify({"status": "error", "message": "视频源未启动"}), 400
    if source.mode != "camera":
        return jsonify({"status": "error", "message": "当前不是摄像头源，无法打快门"}), 400
    if _recorders:
        return jsonify({"status": "error", "message": "正在录像，请先停止再打快门"}), 400

    was_paused = source._paused
    released_device = False
    if camera_control.needs_device_release() and not was_paused:
        released = source.release_for_control(timeout=3.0)
        if not released:
            source.resume_after_control()
            return jsonify({"status": "error", "message": "无法让出摄像头（OpenCV 仍占用）"}), 500
        released_device = True
        time.sleep(0.3)
    try:
        camera_control.ffc_update()
        return jsonify({"status": "ok"})
    except camera_control.CameraControlError as exc:
        config.log(f"快门校正失败: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500
    except Exception as exc:
        config.log(f"快门校正异常: {exc}")
        return jsonify({"status": "error", "message": f"快门校正异常: {exc}"}), 500
    finally:
        if released_device:
            source.resume_after_control()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _port_in_use(port: int) -> bool:
    """检测端口是否被占用（防止重复启动多实例抢相机）。"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


def main():
    global hub, source

    parser = argparse.ArgumentParser(description="本地实时推理模拟 Web 应用")
    parser.add_argument("--cam", type=int, default=config.DEFAULT_CAM_INDEX, help="摄像头索引")
    parser.add_argument("--models-dir", type=str, default=config.MODELS_DIR, help="模型目录（扫描 .pth）")
    parser.add_argument("--device", type=str, default=config.DEFAULT_DEVICE, choices=["cpu", "cuda"], help="推理设备")
    parser.add_argument("--port", type=int, default=config.PORT, help="服务端口")
    parser.add_argument("--host", type=str, default=config.HOST, help="绑定地址")
    parser.add_argument("--source", type=str, default=None,
                        help="视频源覆盖: image:<path> 或 video:<path>（调试用，默认摄像头）")
    parser.add_argument("--list-cameras", action="store_true", help="列出可用摄像头并退出")
    parser.add_argument("--list-models", action="store_true", help="列出模型目录中的可用模型并退出")
    parser.add_argument("--hat-weights", type=str, default=config.HAT_WEIGHTS,
                        help="Real-HAT-GAN 权重路径（拍照即时；录像停后离线）")
    parser.add_argument("--fusion-root", type=str, default=config.FUSION_TOOL_ROOT,
                        help="Image-Tool 仓库本地克隆根目录（提供 network_hat.py）")
    args = parser.parse_args()
    config.HAT_WEIGHTS = args.hat_weights
    config.FUSION_TOOL_ROOT = args.fusion_root
    config.DEFAULT_DEVICE = args.device
    if not config.HAT_DEVICE:
        config.HAT_DEVICE = args.device

    # 端口冲突检测：已有实例在跑时直接退出，避免双实例抢相机/端口
    if _port_in_use(args.port):
        config.log(f"端口 {args.port} 已被占用（可能有实例在运行）。请先停止旧实例或换端口。")
        return

    # 抑制 OpenCV 日志（摄像头断开/热插拔时 WARN 刷屏），仅保留 ERROR 及以上
    import cv2.utils.logging as _cv2log
    _cv2log.setLogLevel(_cv2log.LOG_LEVEL_ERROR)

    if args.list_cameras:
        for cam in list_cameras():
            mark = "OK " if cam["ok"] else "N/A"
            size = f"{cam.get('width', '?')}x{cam.get('height', '?')}" if cam["ok"] else "-"
            print(f"  [{mark}] index={cam['index']}  {size}")
        return

    # 模型注册表（先于摄像头，模型错误早暴露）
    hub = ModelHub(args.models_dir, device=args.device)
    config.log(f"模型目录: {args.models_dir} → {hub.names if hub.names else '(空)'}")
    hat_ok, hat_detail = hat_backend.available()
    if hat_ok:
        config.log(f"HAT 可用（拍照即时 / 录像停后离线）: {hat_detail}")
    else:
        config.log(f"HAT 不可用: {hat_detail}")
    false_color.ensure_luts()
    if args.list_models:
        print("可用模型:", _selectable_names() or "(无)")
        print("伪彩 LUT:", false_color.lut_dir())
        return
    if not hub.names and not hat_ok:
        config.log("模型目录为空且无 HAT 权重，请检查 --models-dir / --hat-weights")
        return

    # 视频源（摄像头或文件）
    try:
        source = FrameSource(state, cam_index=args.cam, source=args.source)
        source.start()
    except Exception as exc:
        config.log(f"视频源启动失败: {exc}")
        return

    if camera_control.ffc_capable():
        config.log("相机控制: 快门校正已启用")
    elif camera_control.plugin_present():
        config.log("相机控制: 插件存在但当前不可用")

    config.log(f"服务启动: http://{args.host}:{args.port}  (摄像头 index={args.cam})")
    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        source.stop()
        config.log("已关闭")


if __name__ == "__main__":
    main()
