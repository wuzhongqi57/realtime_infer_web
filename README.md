# IR Realtime Infer Web

本地 USB 红外摄像头采集 → 逐帧超分推理 → 浏览器多面板实时对比。

适合快速查看「原图 vs 模型效果」，支持同时开最多 3 个面板，各自独立选择显示内容。

## 功能概览

- **摄像头自适应**：启动时读取实际分辨率，自动调整显示
- **跨分辨率支持**：即使相机分辨率与模型期望输入不同，所有模型仍可选择。程序会自动缩放相机帧到模型输入尺寸再推理，并将结果缩放到统一的面板尺寸以便对比
- **多面板对比**：1～3 个面板，每个面板可单独选「原图」或某个已加载模型
- **模型目录扫描**：指定目录下的权重文件会自动出现在面板下拉框
- **调试源**：也可用静态图或视频模拟摄像头（不必接真机）
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
| `--list-models` | 列出模型目录中的可用模型 | — |

## 界面说明

- 页头可切换面板数量（1～3）
- 每个面板下拉框选择「原图」或某个模型
- 画布尺寸 = 相机分辨率 × 最大放大倍数，便于并排对比
- 「原图」使用最近邻放大（保留像素感）
- 模型结果使用平滑缩放（跨分辨率模型的输出会自适应缩放到面板尺寸）

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
2. 关闭自动 RGB 转换，按原始 YUY2 取灰度
3. 上电后短暂预热并丢弃空帧

若仍无画面：先用 `--list-cameras` 确认索引，再试 `--cam N`，或改用 `--source image:...` / `video:...` 验证推理链路。

## 已知事项

- 默认端口 `5001`（可用 `--port` 修改）
- GPU 需安装对应 CUDA 版 PyTorch
- 仅加载你信任来源的权重文件

## 许可证与用途

本仓库提供本地演示与对比工具。模型权重、训练配置与内部命名不在公开文档中说明；请使用你有权使用的权重。
