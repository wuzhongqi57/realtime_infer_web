"""本地实时推理模拟 Web 应用 — Flask + WebSocket 入口。

启动：conda run -n ir_realtime python app.py [--cam 0] [--port 5001]
打开：http://127.0.0.1:5001

架构：两个固定面板，每面板独立选择显示"原图"或某个模型的效果。
  - /ws/panel：每个面板一条 WebSocket，客户端可发送 {"model": <名字>} 切换
  - /api/models：返回可选模型列表
  - 同一模型同一帧的推理结果跨面板共享（ModelHub 帧缓存）
  - 原图 256×192 由前端最近邻放大到面板分辨率（1024×768）
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
def _jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    return buf.tobytes() if ok else b""


def _render(name: str, frame: np.ndarray, seq: int) -> bytes:
    """按面板选择渲染一帧为 JPEG。
    - '原图'：最近邻缩放到面板固定尺寸（与 SR 输出同尺寸，对比更明显）
    - 模型名：推理后发超分结果（本身就是面板尺寸）
    """
    if name == config.RAW_LABEL:
        w, h = _get_panel_size()
        if h and (frame.shape != (h, w)):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
        return _jpeg(frame)
    sr, _ = hub.infer(name, frame, seq)
    return _jpeg(sr)


# ---------------------------------------------------------------------------
# WebSocket — 单一面板流（二进制 JPEG 帧）
# ---------------------------------------------------------------------------
@sock.route("/ws/panel")
def ws_panel(ws):
    selected = config.RAW_LABEL
    last_seq = 0
    while ws.connected:
        # 非阻塞接收客户端选择消息（timeout 即轮询间隔）
        msg = ws.receive(timeout=0.02)
        if msg:
            try:
                d = json.loads(msg)
                if d.get("model") in ([config.RAW_LABEL] + hub.names):
                    selected = d["model"]
                    last_seq = 0          # 强制下一帧立即发送
            except Exception:
                pass

        frame, seq = state.get_raw()
        if frame is None or seq == last_seq:
            continue
        last_seq = seq
        try:
            data = _render(selected, frame, seq)
        except Exception as exc:
            # 推理/渲染失败不中断连接，打印后跳过本帧（保原图侧可用）
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


def matching_models() -> list[str]:
    """按当前相机分辨率过滤出匹配的模型（模型输入尺寸 == 相机尺寸）。

    相机未知或尺寸非标（无精确匹配）时全部列出，避免下拉表为空。
    """
    w, h = state.cam_size
    if not w:
        return hub.names
    exact = [info["name"] for info in hub.infos
             if resolve_input_size(info["name"]) == (w, h)]
    return exact if exact else hub.names


@app.route("/api/models")
def api_models():
    # 每次拉取都重新扫描目录（页面刷新等），无需重启服务
    _rescan_models()
    # 等待相机分辨率确定（最多 5s），据此过滤匹配模型
    for _ in range(50):
        if state.cam_size[0]:
            break
        time.sleep(0.1)
    return jsonify({"models": matching_models(), "camera_size": list(state.cam_size),
                    "panel_size": list(_get_panel_size())})


@app.route("/api/status")
def api_status():
    raw, raw_seq = state.get_raw()
    return jsonify({
        "running": state.running,
        "raw_seq": raw_seq,
        "raw_fps": round(state.raw_fps, 1),
        "camera": raw is not None,
        "camera_size": list(state.cam_size),
        "panel_size": list(_get_panel_size()),
        "models": matching_models(),
    })


# ---------------------------------------------------------------------------
# 拍照 / 录像 / 相机控制
# ---------------------------------------------------------------------------
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "snapshots")
RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "recordings")
_recorders = {}          # model -> (thread, rec)；每面板一路并行录制


def _ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_model_name(model: str) -> str:
    """模型名转安全文件名（全英文）：原图 → raw，其余去特殊字符。"""
    if model == config.RAW_LABEL:
        return "raw"
    return re.sub(r"[^\w.-]+", "_", model) or "model"


@app.route("/api/snapshot", methods=["POST"])
def api_snapshot():
    """保存当前帧（原图或指定模型效果）为 PNG。
    Request: {"model": "原图" | 模型名}   (默认原图)
    """
    data = request.get_json(silent=True) or {}
    model = data.get("model", config.RAW_LABEL)
    fmt = (data.get("format") or "png").lower()
    if fmt not in ("png", "jpg", "jpeg"):
        fmt = "png"
    ext = "jpg" if fmt in ("jpg", "jpeg") else "png"
    frame, seq = state.get_raw()
    if frame is None:
        return jsonify({"status": "error", "message": "当前无帧（相机未连接）"}), 400
    try:
        if model == config.RAW_LABEL:
            img = frame
        else:
            img, _ = hub.infer(model, frame, seq)   # 取数组（勿用 _render，其返回 JPEG bytes）
    except Exception as exc:
        return jsonify({"status": "error", "message": f"渲染失败: {exc}"}), 400
    _ensure_dir(SNAPSHOT_DIR)
    fn = f"{_timestamp()}_{_safe_model_name(model)}.{ext}"
    path = os.path.join(SNAPSHOT_DIR, fn)
    cv2.imwrite(path, img)
    h, w = img.shape[:2]
    # 帧统计（诊断：纯噪声/平坦帧提示）
    import numpy as _np
    _std = float(_np.asarray(img).std())
    _mean = float(_np.asarray(img).mean())
    warn = None
    if _std < 2:
        warn = "帧接近平坦（std=%.1f），请检查相机画面" % _std
    return jsonify({"status": "ok", "file": os.path.join("output", "snapshots", fn),
                    "model": model, "size": [h, w], "mean": round(_mean, 1), "std": round(_std, 1),
                    "warn": warn})


@app.route("/api/recording/start", methods=["POST"])
def api_recording_start():
    """开始录像：同时录制所有面板对应的视频（每面板一路）。
    Request: {"models": ["原图", "new_ir256_dim8", ...]}  {"fps": 帧率}
    """
    global _recorders
    if _recorders:
        return jsonify({"status": "error", "message": "录像已在进行中"}), 400
    data = request.get_json(silent=True) or {}
    models = data.get("models") or [data.get("model", config.RAW_LABEL)]
    fps = int(data.get("fps", 15))
    fmt = (data.get("format") or "avi").lower()
    if fmt not in ("avi", "mp4"):
        fmt = "avi"
    fourcc = (cv2.VideoWriter_fourcc(*"MJPG") if fmt == "avi"
              else cv2.VideoWriter_fourcc(*"avc1"))   # MP4 用 H.264(avc1)，播放器兼容性最好
    valid = [config.RAW_LABEL] + (hub.names if hub else [])
    # 去重 + 过滤非法模型
    models = list(dict.fromkeys(m for m in models if m in valid))
    if not models:
        return jsonify({"status": "error", "message": "无有效面板模型可录制"}), 400
    frame, seq = state.get_raw()
    if frame is None:
        return jsonify({"status": "error", "message": "当前无帧（相机未连接）"}), 400
    h, w = frame.shape[:2]
    _ensure_dir(RECORDING_DIR)
    ts = _timestamp()
    for model in models:
        # 每路 writer 用该路输出尺寸：原图=raw 尺寸，模型=SR 尺寸（scale×输入）
        if model == config.RAW_LABEL:
            vw, vh = w, h
        else:
            _, scale = hub._get_engine(model)
            vw, vh = w * scale, h * scale
        fn = f"{ts}_{_safe_model_name(model)}.{fmt}"
        path = os.path.join(RECORDING_DIR, fn)
        writer = cv2.VideoWriter(path, fourcc, fps, (vw, vh), isColor=True)
        if not writer.isOpened():
            return jsonify({"status": "error", "message": f"VideoWriter 打开失败: {fn}"}), 500
        stop_evt = threading.Event()
        rec = {"writer": writer, "path": path, "model": model, "fps": fps,
               "stop": stop_evt, "frames": 0, "last": 0}
        _recorders[model] = (threading.Thread(target=_rec_loop, args=(rec,), daemon=True), rec)
    for t, _ in _recorders.values():
        t.start()
    files = [os.path.join("output", "recordings", os.path.basename(r["path"]))
             for _, r in _recorders.values()]
    return jsonify({"status": "ok", "files": files, "models": list(_recorders.keys()), "fps": fps})


def _rec_loop(rec: dict) -> None:
    """单路录像线程：按 rec['model'] 渲染帧写入（灰度转 BGR，MJPG 需 3 通道）。"""
    while not rec["stop"].is_set():
        f, s = state.get_raw()
        if f is None or s == rec["last"]:
            time.sleep(0.02)
            continue
        rec["last"] = s
        try:
            if rec["model"] == config.RAW_LABEL:
                img = f
            else:
                img, _ = hub.infer(rec["model"], f, s)   # 取数组（勿用 _render）
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)   # MJPG 编码器需 3 通道
            rec["writer"].write(img)
            rec["frames"] += 1
        except Exception:
            pass
    rec["writer"].release()
    config.log(f"录像已保存: {rec['path']} ({rec['frames']} 帧)")


@app.route("/api/recording/stop", methods=["POST"])
def api_recording_stop():
    """停止全部录像，返回各文件路径与帧数。"""
    global _recorders
    results = []
    for model, (t, rec) in list(_recorders.items()):
        rec["stop"].set()
        t.join(timeout=3)
        results.append({"model": model,
                        "file": os.path.join("output", "recordings", os.path.basename(rec["path"])),
                        "frames": rec["frames"]})
    _recorders = {}
    return jsonify({"status": "ok", "files": results})


@app.route("/api/recording/status", methods=["GET"])
def api_recording_status():
    if _recorders:
        return jsonify({"status": "ok", "recording": True,
                        "models": list(_recorders.keys()),
                        "frames": sum(r["frames"] for _, r in _recorders.values())})
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
    return jsonify({"status": "ok", "connected": running,
                    "size": list(state.cam_size), "paused": source._paused})


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
    args = parser.parse_args()

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
    config.log(f"模型目录: {args.models_dir} → {hub.names if hub.names else '(空!)'}")
    if args.list_models:
        print("可用模型:", hub.names or "(无)")
        return
    if not hub.names:
        config.log("模型目录为空，请检查 --models-dir")
        return

    # 视频源（摄像头或文件）
    try:
        source = FrameSource(state, cam_index=args.cam, source=args.source)
        source.start()
    except Exception as exc:
        config.log(f"视频源启动失败: {exc}")
        return

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
