# 相机控制插件（可选）

本仓库的采集与推理不依赖任何厂商 SDK。若本机需要「快门校正」一类机芯命令，把实现放到 **`plugins/local.py`**（该文件已被 gitignore，不会进入公开仓库）。

启动时只会加载 `local.py`，不会把 `example_stub.py` 或其他文件当生产插件。

## 约定

```python
# plugins/local.py（本机私有，不入库）

NEEDS_DEVICE_RELEASE = True  # 默认 True：调用前释放 OpenCV 占用的 USB

def available() -> tuple[bool, str]:
    """能否执行控制；不能则返回 (False, 原因)。"""

def ffc_update() -> None:
    """执行一次快门/FFC。失败抛异常。"""
```

- `available()` 为假时，工具栏不显示「快门校正」按钮。
- `NEEDS_DEVICE_RELEASE=True` 时，服务会先让出 OpenCV 设备，再调用 `ffc_update()`。
- 厂商 SDK 路径、VID/PID 等只写在 `local.py` 或环境变量里，不要改核心 `config.py`。

可参考同目录的 `example_stub.py`（不会被自动加载）。
