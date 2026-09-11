# IR 实时推理模拟 Web 应用

本地 USB 红外摄像头 → 逐帧超分推理 → 浏览器双面板实时对比不同模型效果。

- **摄像头**：分辨率自适应（256×192 或 640×512 均可，启动时按实际设备读取）
- **模型**：目录 `E:\wuzq\models` 下所有 `.pth`（自动从 checkpoint 推导架构）
  - **ir256 系列**：×4，输入 256×192 → 输出 1024×768（new/pre × dim8/dim16）
  - **ir640 系列**：×2，输入 640×512 → 输出 1280×1024（new/pre × dim8）
  - 推理前自动把相机帧缩放到各模型期望输入尺寸（640 模型←640×512，256 模型←256×192）
- **技术栈**：Flask + WebSocket（原生，无 CDN 依赖）+ OpenCV + PyTorch
- **交互**：支持 **1~3 个面板**自由切换（页头下拉选择）；每个面板画布固定为匹配模型的 SR 输出尺寸（640 相机→1280×1024，256 相机→1024×768），所有面板始终一致；每面板独立下拉选择显示「原图」或模型效果
- **自动匹配**：启动时自动识别摄像头分辨率（256×192 或 640×512），下拉表**只列出与该分辨率匹配的模型**（ir256 系列 ← 256×192 相机，ir640 系列 ← 640×512 相机），不再全部列出

## 环境（一次性）

```bash
conda create -n ir_realtime python=3.12 -y
conda run -n ir_realtime pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## 启动

```bash
conda run -n ir_realtime python app.py --cam 0
```

浏览器打开 <http://127.0.0.1:5001>。

### 常用参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--cam N` | 摄像头索引 | 0 |
| `--models-dir PATH` | 模型目录（扫描 .pth） | `E:\wuzq\models` |
| `--device cpu/cuda` | 推理设备 | cpu |
| `--port N` | 服务端口 | 5001 |
| `--source image:<path>` / `video:<path>` | 用静态图/视频模拟摄像头（调试） | — |
| `--list-cameras` | 列出可用摄像头 | — |
| `--list-models` | 列出模型目录中的可用模型 | — |

## 架构

```
FrameSource（采集线程） → SharedState（互斥锁） → ModelHub（多模型推理）
                                                    │
                      /ws/panel（每面板一条 WS）◄────┘
                        ├─ 客户端发送 {"model": "原图"|模型名} 切换
                        └─ 服务器推 JPEG 帧（二进制）
```

- **面板**：画布固定为匹配模型的 SR 输出尺寸（640 相机→1280×1024，256 相机→1024×768），更换相机时才改变。原图由服务器用最近邻缩放到该尺寸，模型输出本身即该尺寸，两面板内容同尺寸并排对比。选择「原图」时前端用最近邻（`imageSmoothingEnabled=false` + `image-rendering: pixelated`），模型效果用双线性平滑
- **模型输入适配**：每个模型有期望输入尺寸（`config.MODEL_INPUT_SIZES` 按模型名匹配），推理前把相机帧缩放过去；降采样用 INTER_AREA、升采样用 INTER_LINEAR
- **多模型共享**：同一模型同一帧只推理一次，多个面板选同一模型时直接复用缓存
- **无 FPS 显示**：界面只展示画面和模型选择，不显示帧率

## 摄像头实测要点（IR UVC，VID_3474 系列）

该红外相机直接 `cv2.VideoCapture(0)` 无法出帧，实测需满足三个条件（已固化在 `capture.py`）：

1. **后端必须用 MSMF**（`cv2.CAP_MSMF`）——DSHOW 后端打不开该设备
2. **必须设 `CAP_PROP_CONVERT_RGB=0`**——默认 8-bit 协商返回 `MF_E_NO_VIDEO_SAMPLE_AVAILABLE` 无帧；设 0 后取到 YUY2 原始帧（256×192→98304 字节，640×512→655360 字节）
3. **上电预热约 1 秒**——前若干帧为空帧，`extract_gray()` 自动过滤（max==min 跳过），取 Y 亮度平面作为灰度图

实测：640×512 相机采集正常，ir640 dim8 模型单帧 ~25 ms（CPU，~40 fps）；ir256 dim8 模型单帧 ~7 ms。

## 已知事项

- 默认端口 5001（5000 已被 image_fusion_tool 占用），可用 `--port` 覆盖
- 若需 GPU 推理，重装 CUDA 版 torch：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch`（约 2.5GB）
- `pre_*` 权重含 numpy 标量，加载时自动回退 `weights_only=False`（信任来源）
