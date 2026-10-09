# IR Realtime Infer Web

本地 USB 红外摄像头采集 → 逐帧超分推理 → 浏览器多面板实时对比。

适合快速查看「原图 vs 模型效果」，支持同时开最多 3 个面板，各自独立选择显示内容。

## 功能概览

- **摄像头自适应**：启动时读取实际分辨率，自动调整显示
- **跨分辨率支持**：即使相机分辨率与模型期望输入不同，所有模型仍可选择。程序会自动缩放相机帧到模型输入尺寸再推理，并将结果缩放到统一的面板尺寸以便对比
- **多面板对比**：1～3 个面板（页头可调），每个面板可单独选「原图」或某个已加载模型
- **多路拍照/录像**：按当前各面板所选内容并行保存，最多 3 路（同模型去重）
- **HAT×4（可选）**：下拉可选，但**不做实时预览**（仅首帧占位）；**拍照**即时跑 Real-HAT-GAN；**录像**时不超分，停止后对已录原图离线超分并显示进度。权重默认取自 [Image-Tool](https://github.com/wuzhongqi57/Image-Tool) 仓库的 `model_zoo/hat/`，不随本仓库分发
- **YUY2 超分+伪彩**：仅 Y 超分 → UV 双线性对齐 → 可选伪彩（白热/黑热/铁虹/熔岩/彩虹）→ 打包 YUY2；调色板查找表见 `luts/`
- **模型目录扫描**：指定目录下的权重文件会自动出现在面板下拉框
- **调试源**：也可用静态图或视频模拟摄像头（不必接真机）
- **可选相机控制插件**：本机若提供 `plugins/local.py`，工具栏才显示「快门校正」等机芯命令（见 `plugins/README.md`）
- **技术栈**：Flask + WebSocket、OpenCV、PyTorch（无 CDN 依赖）

## 环境（一次性）

```bash
conda create -n ir_realtime python=3.12 -y
conda run -n ir_realtime pip install -r requirements.txt
```

国内镜像可选：

```bash
conda run -n ir_realtime pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## 准备模型

1. 将 `.pth` 权重放到任意本地目录（权重**不随本仓库分发**）
2. 启动时用 `--models-dir` 指向该目录

程序会按权重文件名自动识别输入尺寸，并把相机帧缩放到模型期望尺寸后再推理。

## 启动

```bash
conda run -n ir_realtime python app.py --cam 0 --models-dir /path/to/models
```

浏览器打开 <http://127.0.0.1:5001>。

GPU 示例：

```bash
conda run -n ir_realtime python app.py --cam 0 --models-dir /path/to/models --device cuda
```

Windows 也可双击 `run.bat` / `run_gpu.bat`（请先按脚本内说明改好模型目录与环境名）。

### 常用参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--cam N` | 摄像头索引 | `0` |
| `--models-dir PATH` | 模型目录（扫描 `.pth`） | 见 `config.py` |
| `--device cpu/cuda` | 推理设备 | `cpu` |
| `--port N` | 服务端口 | `5001` |
| `--source image:<path>` / `video:<path>` | 用静态图/视频模拟摄像头 | — |
| `--list-cameras` | 列出可用摄像头 | — |
| `--list-models` | 列出可用模型（含 HAT） | — |
| `--hat-weights PATH` | Real-HAT-GAN 权重 | Image-Tool 仓库 `model_zoo/hat/...` |
| `--fusion-root PATH` | Image-Tool 仓库本地克隆（提供 `network_hat.py`） | 同级 `image_fusion_tool` 目录 |

### 性能调优参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--fp16` / `--no-fp16` | 启用/禁用 FP16 半精度推理（仅 CUDA） | `--fp16`（CUDA 时默认开启） |
| `--stream-size WxH` | 直播流最大尺寸（如 `1280x1024`），超出时等比缩放 | `1280x1024` |
| `--stream-size 0` | 禁用流缩放，以面板原尺寸推流 | — |
| `--stats` | 启动时显示统计叠加层 | 关闭 |

FP16 推理在 CUDA 上可减少约 50% 推理时间，输出差异通常 < 0.5 灰度级。
流缩放将 2560x2048 面板缩至 1280x1024 后再 JPEG 编码，可减少编码时间约 70%、带宽约 60%。

## 界面说明

- 页头可切换面板数量（1～3）
- 每个面板下拉框选择「原图」或某个模型
- 画布尺寸 = 相机分辨率 × 最大放大倍数，便于并排对比
- 原图放大到面板用最近邻；模型结果缩放到面板仍用双线性
- 选中 `HAT×4` 时直播仅显示占位提示；拍照会即时超分；录像过程中不跑 HAT（避免卡死），停止后弹出进度条对原图录像离线超分
- 页头可开关**伪彩**并选择调色板；关伪彩=白热（仅 Y）；拍照默认附带 `.yuy2` 原始帧，便于逐字节对照
- **📊 统计**按钮或 `?stats=1` URL 参数可显示每面板的 FPS、渲染耗时、推理耗时及端到端延迟

## 伪彩查找表

调色板查找表随仓库提供：`luts/<调色板>.bin`（256×3 字节，每项 B,G,R）与同内容的 `luts/<调色板>.csv`。
如需重新生成（程序启动时也会自动补齐缺失文件）：

```bash
conda run -n ir_realtime python -c "import false_color; false_color.ensure_luts()"
```

## 架构（简要）

```
采集线程 → 共享帧缓冲 → 多模型推理缓存
                ↓
         每面板一条 WebSocket
                ↓
         浏览器 JPEG 实时画面
```

同一模型、同一帧只推理一次；多个面板选同一模型时复用结果。

## 摄像头说明（Windows 红外 UVC）

部分红外相机用默认 OpenCV 设置可能无法出帧。本仓库的采集逻辑已按常见 IR UVC 设备做过适配，主要包括：

1. 优先使用 MSMF 后端
2. 关闭自动 RGB 转换，按原始 YUY2 取 **Y+UV**（超分只用 Y；伪彩按文档链路）
3. 上电后短暂预热并丢弃空帧

若仍无画面：先用 `--list-cameras` 确认索引，再试 `--cam N`，或改用 `--source image:...` / `video:...` 验证推理链路。

## 性能基准测试

使用 `tools/bench_stream.py` 可在本地测量各阶段耗时：

```bash
# 测试所有模型（CPU）
python tools/bench_stream.py --models-dir /path/to/models

# 测试特定模型，含 FP16（GPU）
python tools/bench_stream.py --models-dir /path/to/models --model mymodel --device cuda --fp16
```

输出包括：
- 各色彩映射（ironbow、white_hot 等）的 LUT 应用时间
- 不同尺寸的 resize 和 JPEG 编码时间
- FP32/FP16 推理时间及精度差异

## 已知事项

- 默认端口 `5001`（可用 `--port` 修改）
- GPU 需安装对应 CUDA 版 PyTorch
- 仅加载你信任来源的权重文件
- HAT 依赖 `einops` 与 [Image-Tool](https://github.com/wuzhongqi57/Image-Tool) 的 `network_hat.py`；权重约 162MB，不入库
- HAT 全帧约数秒～十余秒/帧，**不能**当实时预览

## 许可证与用途

本仓库提供本地演示与对比工具。模型权重、训练配置与内部命名不在公开文档中说明；请使用你有权使用的权重。
