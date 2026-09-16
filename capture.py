"""视频源采集 — 摄像头 / 图像文件 / 视频文件，单生产者线程。

把最新一帧灰度图 (H,W) 写入 SharedState，帧序号单调递增。
摄像头分辨率自适应：启动时按设备实际输出读取（256×192 或 640×512 均可）。

红外 UVC 相机（VID_3474 系列）实测取流要求：
- 后端必须用 MSMF（DSHOW 无法打开该设备）
- 必须设 CAP_PROP_CONVERT_RGB=0 才能取到帧（默认 8-bit 协商返回
  MF_E_NO_VIDEO_SAMPLE_AVAILABLE，无帧）
- 帧格式为 YUY2：256×192 → 98304 字节，640×512 → 655360 字节；
  亮度 Y 在每 2 字节首位
- 上电后约 1 秒（前若干帧）为空帧，需过滤
"""

import threading
import time

import cv2
import numpy as np

import config


class SharedState:
    """多线程共享的最新帧缓冲区。所有读写都经锁保护。"""

    def __init__(self):
        self.running = True
        self._lock = threading.Lock()
        self.raw_frame = None      # 最新原图灰度 uint8 (H,W)，分辨率随相机
        self.raw_seq = 0           # 原图帧序号（单调递增）
        self.raw_fps = 0.0         # 实际采集帧率
        self.cam_size = (0, 0)     # 相机实际分辨率 (W,H)
        self.sr_frame = None       # 最新超分结果 uint8 (H*scale,W*scale)
        self.sr_seq = 0            # 已处理的帧序号
        self.sr_ms = 0.0           # 最近一次推理耗时(ms)
        self.infer_fps = 0.0       # 实际推理帧率

    # -- 原图读写 --
    def set_raw(self, frame: np.ndarray) -> None:
        with self._lock:
            self.raw_frame = frame
            self.raw_seq += 1

    def get_raw(self) -> tuple:
        with self._lock:
            return self.raw_frame, self.raw_seq

    # -- 超分结果读写 --
    def set_sr(self, frame: np.ndarray, seq: int, ms: float) -> None:
        with self._lock:
            self.sr_frame = frame
            self.sr_seq = seq
            self.sr_ms = ms

    def get_sr(self) -> tuple:
        with self._lock:
            return self.sr_frame, self.sr_seq, self.sr_ms


def list_cameras(max_index: int = 6) -> list[dict]:
    """枚举摄像头（MSMF），返回 [{index, ok, width, height}]。"""
    result = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        ok, frame = cap.read()
        entry = {"index": i, "ok": ok}
        if ok and frame is not None:
            entry["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            entry["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        result.append(entry)
    return result


def extract_gray(frame: np.ndarray, cam_w: int, cam_h: int):
    """从相机原始帧提取 8-bit 灰度图 (cam_h, cam_w)。

    支持两种输入：
      - 原始 YUY2 缓冲 (1, W*H*2) uint8  → 取 Y 亮度平面
      - 已是灰度 (H,W) uint8             → 直接返回
    返回 None 表示空帧/平坦帧（需跳过）。
    """
    n = int(frame.size)
    if n == cam_w * cam_h * 2:                       # YUY2
        b = frame.tobytes()
        y = np.frombuffer(b[0::2], np.uint8).reshape(cam_h, cam_w)
    elif n == cam_w * cam_h:                          # 已是灰度
        y = frame.reshape(cam_h, cam_w)
    else:
        return None

    # 空帧过滤：上电预热期全 0 / 全 128 的平坦帧跳过
    if y.max() == y.min():
        return None
    return y


class FrameSource:
    """视频源。mode: 'camera' | 'image' | 'video'。运行在独立线程，持续写入 state。"""

    def __init__(self, state: SharedState, cam_index: int = 0, source: str = None):
        self.state = state
        self.cam_index = cam_index
        self.source = source          # None → camera；"image:<path>" | "video:<path>"
        self.mode = "camera"
        self._cap = None
        self._static_frame = None
        self._thread = None
        self._fps_window = []
        self._last_t = time.perf_counter()
        self.cam_w = 0                # 相机实际分辨率（打开后读取）
        self.cam_h = 0
        self._paused = False          # 手动断开（暂停采集）标志
        self._pause_lock = threading.Lock()
        self._control_hold = threading.Event()  # 快门校正期间让出 VideoCapture

        if source:
            if source.startswith("image:"):
                self.mode = "image"
                path = source[len("image:"):]
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(f"无法读取图像源: {path}")
                self._static_frame = img
                self.cam_w, self.cam_h = img.shape[1], img.shape[0]
                self.state.cam_size = (self.cam_w, self.cam_h)
                config.log(f"图像源: {path} ({self.cam_w}x{self.cam_h})")
            elif source.startswith("video:"):
                self.mode = "video"
                path = source[len("video:"):]
                self._cap = cv2.VideoCapture(path)
                if not self._cap.isOpened():
                    raise FileNotFoundError(f"无法打开视频源: {path}")
                self.cam_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.cam_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.state.cam_size = (self.cam_w, self.cam_h)
                config.log(f"视频源: {path} ({self.cam_w}x{self.cam_h})")
            else:
                raise ValueError("--source 必须是 'image:<path>' 或 'video:<path>'")

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.mode == "camera":
            # 热插拔：不在此阻塞/抛错，由采集线程负责打开与自动重连
            config.log(f"摄像头线程启动: index={self.cam_index}（热插拔自动重连）")
        self._thread = threading.Thread(target=self._loop, daemon=True, name="frame-source")
        self._thread.start()

    def _open_camera(self) -> bool:
        """打开摄像头并读取实际分辨率。返回是否成功。"""
        if self._cap is not None:
            return True
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_MSMF)
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS_TARGET)
        self.cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap = cap
        with self.state._lock:
            self.state.cam_size = (self.cam_w, self.cam_h)
        config.log(f"摄像头已打开: index={self.cam_index}, {self.cam_w}x{self.cam_h}")
        return True

    def _loop(self) -> None:
        fail = 0
        while self.state.running:
            # 手动断开：释放采集、挂起线程，等 resume() 恢复
            if self._paused:
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                with self.state._lock:
                    self.state.cam_size = (0, 0)
                config.log("摄像头已手动断开")
                time.sleep(0.2)
                continue
            if self._control_hold.is_set():
                if self._cap is not None:
                    self._cap.release()
                    self._cap = None
                    config.log("摄像头已让出（快门校正）")
                time.sleep(0.05)
                continue
            gray = None
            if self.mode == "camera":
                if self._cap is None:
                    if self._open_camera():
                        fail = 0
                        continue                     # 刚打开，立即取帧
                    time.sleep(1.0)                  # 重连间隔
                    continue
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    fail += 1
                    if fail >= 10:                   # 持续失败 → 判定断开，释放重连
                        config.log("摄像头断开，尝试重连...")
                        self._cap.release()
                        self._cap = None
                        fail = 0
                        with self.state._lock:
                            self.state.cam_size = (0, 0)
                    else:
                        time.sleep(0.2)              # 短暂失败退避，避免忙循环
                    continue
                fail = 0
                gray = extract_gray(frame, self.cam_w, self.cam_h)
                if gray is None:                     # 预热期空帧，跳过
                    time.sleep(0.02)
                    continue
            elif self.mode == "image":
                gray = self._static_frame
                time.sleep(1.0 / 15.0)               # 静态图模拟 ~15fps
            else:  # video
                ok, frame = self._cap.read()
                if not ok:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # 循环播放
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if gray is None:
                continue
            self.state.set_raw(gray)
            self._update_fps()

    def _update_fps(self) -> None:
        now = time.perf_counter()
        self._fps_window.append(1.0 / max(now - self._last_t, 1e-6))
        self._last_t = now
        if len(self._fps_window) > 30:
            self._fps_window.pop(0)
        with self.state._lock:
            self.state.raw_fps = sum(self._fps_window) / len(self._fps_window)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None

    def pause(self) -> None:
        """手动断开相机：释放设备并暂停采集线程（不退出线程）。"""
        with self._pause_lock:
            self._paused = True
        config.log("camera pause requested")

    def resume(self) -> None:
        """重新连接相机：恢复采集线程（热插拔逻辑自动重开设备）。"""
        with self._pause_lock:
            self._paused = False
        if self.mode != "camera":
            # 静态图/视频源：恢复分辨率标记
            with self.state._lock:
                self.state.cam_size = (self.cam_w, self.cam_h)
        config.log("camera resume requested")

    def release_for_control(self, timeout: float = 3.0) -> bool:
        """暂停 OpenCV 取流并释放设备，供机芯 USB 命令通道占用。不改 cam_size。"""
        self._control_hold.set()
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < timeout:
            if self._cap is None:
                return True
            time.sleep(0.05)
        return self._cap is None

    def resume_after_control(self) -> None:
        """快门校正结束后恢复 OpenCV 采集。"""
        self._control_hold.clear()
