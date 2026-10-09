#!/usr/bin/env python3
"""Benchmark script for IR super-resolution streaming pipeline.

Measures per-stage timing on a synthetic 640x512 grayscale frame:
  - Preprocessing (resize to model input)
  - Inference (FP32 and FP16 if available)
  - False color (various colormaps)
  - Resize to panel/stream size
  - JPEG encode at different sizes

Usage:
    python tools/bench_stream.py --models-dir /path/to/models
    python tools/bench_stream.py --models-dir /path/to/models --model mymodel
    python tools/bench_stream.py --models-dir /path/to/models --device cuda --fp16

The script works without a camera or CUDA - it will skip GPU tests if unavailable.
"""

import argparse
import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np


def generate_synthetic_frame(width: int = 640, height: int = 512) -> np.ndarray:
    """Generate a synthetic grayscale frame with realistic IR-like content."""
    y = np.linspace(0, 1, height).reshape(-1, 1)
    x = np.linspace(0, 1, width).reshape(1, -1)
    frame = (y * x * 255).astype(np.float32)
    noise = np.random.normal(0, 20, (height, width)).astype(np.float32)
    frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
    for _ in range(5):
        cx, cy = np.random.randint(50, width - 50), np.random.randint(50, height - 50)
        Y, X = np.ogrid[:height, :width]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        hotspot = np.clip(255 - dist * 2, 0, 255).astype(np.uint8)
        frame = np.maximum(frame, hotspot)
    return frame


def benchmark_colormap(frame: np.ndarray, colormap: str, iterations: int = 100) -> float:
    """Benchmark colormap application, return median ms per frame."""
    import false_color
    false_color.ensure_luts()
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = false_color.apply_to_bgr(frame, colormap)
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times)


def benchmark_resize(frame: np.ndarray, target_size: tuple, iterations: int = 100) -> float:
    """Benchmark resize operation, return median ms per frame."""
    times = []
    src_h, src_w = frame.shape[:2]
    tgt_w, tgt_h = target_size
    interp = cv2.INTER_AREA if src_w > tgt_w else cv2.INTER_LINEAR
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = cv2.resize(frame, target_size, interpolation=interp)
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times)


def benchmark_jpeg(frame: np.ndarray, quality: int = 80, iterations: int = 100) -> tuple:
    """Benchmark JPEG encode, return (median_ms, avg_size_kb)."""
    times = []
    sizes = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        times.append((time.perf_counter() - t0) * 1000)
        if ok:
            sizes.append(len(buf) / 1024)
    return np.median(times), np.mean(sizes) if sizes else 0


def benchmark_palette_ordering(full_size: tuple, stream_size: tuple,
                               colormap: str = "ironbow", iterations: int = 50) -> dict:
    """Compare palette-at-full-res vs palette-after-downscale.

    Returns dict with timing for both approaches and the speedup factor.
    """
    import false_color
    false_color.ensure_luts()

    full_w, full_h = full_size
    stream_w, stream_h = stream_size

    frame_full = generate_synthetic_frame(full_w, full_h)

    times_full_first = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        bgr_full = false_color.apply_to_bgr(frame_full, colormap)
        bgr_stream = cv2.resize(bgr_full, (stream_w, stream_h), interpolation=cv2.INTER_AREA)
        times_full_first.append((time.perf_counter() - t0) * 1000)

    times_downscale_first = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        gray_stream = cv2.resize(frame_full, (stream_w, stream_h), interpolation=cv2.INTER_AREA)
        bgr_stream = false_color.apply_to_bgr(gray_stream, colormap)
        times_downscale_first.append((time.perf_counter() - t0) * 1000)

    palette_full_ms = np.median(times_full_first)
    palette_after_ms = np.median(times_downscale_first)
    speedup = palette_full_ms / palette_after_ms if palette_after_ms > 0 else 0

    return {
        "full_size": f"{full_w}x{full_h}",
        "stream_size": f"{stream_w}x{stream_h}",
        "colormap": colormap,
        "palette_at_full_ms": palette_full_ms,
        "palette_after_downscale_ms": palette_after_ms,
        "speedup": speedup,
    }


def benchmark_inference(model_name: str, model_path: str, frame: np.ndarray,
                        device: str, use_fp16: bool, iterations: int = 20) -> dict:
    """Benchmark model inference, return timing dict."""
    import torch
    from engine import build_model, resolve_input_size

    results = {}
    try:
        model, first_dim, scale = build_model(model_path)
    except Exception as e:
        return {"error": str(e)}

    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()

    input_w, input_h = resolve_input_size(model_name)
    results["input_size"] = f"{input_w}x{input_h}"
    results["output_size"] = f"{input_w * scale}x{input_h * scale}"
    results["scale"] = scale

    if frame.shape != (input_h, input_w):
        interp = cv2.INTER_AREA if frame.shape[0] >= input_h else cv2.INTER_LINEAR
        input_frame = cv2.resize(frame, (input_w, input_h), interpolation=interp)
    else:
        input_frame = frame

    tensor = torch.from_numpy(input_frame.astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device_obj)

    with torch.no_grad():
        for _ in range(3):
            _ = model(tensor)
    if device == "cuda":
        torch.cuda.synchronize()

    times_fp32 = []
    for _ in range(iterations):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(tensor)
        if device == "cuda":
            torch.cuda.synchronize()
        times_fp32.append((time.perf_counter() - t0) * 1000)

    results["fp32_ms"] = np.median(times_fp32)

    t0 = time.perf_counter()
    sr = out.squeeze().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
    results["copy_back_ms"] = (time.perf_counter() - t0) * 1000
    results["sr_frame"] = sr

    if use_fp16 and device == "cuda":
        try:
            model_fp16 = model.half()
            tensor_fp16 = tensor.half()

            with torch.no_grad():
                for _ in range(3):
                    _ = model_fp16(tensor_fp16)
            torch.cuda.synchronize()

            times_fp16 = []
            for _ in range(iterations):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out_fp16 = model_fp16(tensor_fp16)
                torch.cuda.synchronize()
                times_fp16.append((time.perf_counter() - t0) * 1000)

            results["fp16_ms"] = np.median(times_fp16)

            sr_fp16 = out_fp16.float().squeeze().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
            diff = np.abs(sr.astype(float) - sr_fp16.astype(float))
            results["fp16_max_diff"] = float(diff.max())
            results["fp16_mean_diff"] = float(diff.mean())
        except Exception as e:
            results["fp16_error"] = str(e)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark IR super-resolution streaming pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--models-dir", type=str, required=True,
                        help="Directory containing .pth model files")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model name to benchmark (default: all)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Inference device (default: cpu)")
    parser.add_argument("--fp16", action="store_true",
                        help="Also benchmark FP16 inference (CUDA only)")
    parser.add_argument("--input-size", type=str, default="640x512",
                        help="Input frame size WxH (default: 640x512)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of iterations for each benchmark")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print summary table")
    args = parser.parse_args()

    try:
        input_w, input_h = map(int, args.input_size.lower().split("x"))
    except ValueError:
        print(f"Error: Invalid --input-size format: {args.input_size}")
        return 1

    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if args.device == "cuda" and not cuda_available:
            print("Warning: CUDA requested but not available, falling back to CPU")
            args.device = "cpu"
    except ImportError:
        print("Warning: PyTorch not available, skipping inference benchmarks")

    from engine import discover_models
    models = discover_models(args.models_dir)
    if not models:
        print(f"Error: No models found in {args.models_dir}")
        return 1

    if args.model:
        models = [m for m in models if m["name"] == args.model]
        if not models:
            print(f"Error: Model '{args.model}' not found")
            return 1

    if not args.quiet:
        print(f"Benchmark Configuration:")
        print(f"  Input frame: {input_w}x{input_h} grayscale")
        print(f"  Device: {args.device}" + (" (CUDA available)" if cuda_available else ""))
        print(f"  FP16: {args.fp16 and args.device == 'cuda'}")
        print(f"  Models: {len(models)}")
        print(f"  Iterations: {args.iterations}")
        print()

    frame = generate_synthetic_frame(input_w, input_h)

    if not args.quiet:
        print("=== False Color Benchmarks ===")

    panel_frame = cv2.resize(frame, (input_w * 4, input_h * 4), interpolation=cv2.INTER_NEAREST)
    colormaps = ["white_hot", "black_hot", "ironbow", "lava", "rainbow"]
    colormap_results = {}

    for cm in colormaps:
        ms = benchmark_colormap(panel_frame, cm, args.iterations)
        colormap_results[cm] = ms
        if not args.quiet:
            print(f"  {cm:12s}: {ms:6.2f} ms ({panel_frame.shape[1]}x{panel_frame.shape[0]})")

    if not args.quiet:
        print()
        print("=== Resize Benchmarks ===")

    resize_tests = [
        ("panel→stream", (input_w * 4, input_h * 4), (1280, 1024)),
        ("panel→half", (input_w * 4, input_h * 4), (input_w * 2, input_h * 2)),
    ]

    for name, src_size, tgt_size in resize_tests:
        test_frame = cv2.resize(frame, src_size, interpolation=cv2.INTER_NEAREST)
        ms = benchmark_resize(test_frame, tgt_size, args.iterations)
        if not args.quiet:
            print(f"  {name:15s}: {ms:6.2f} ms ({src_size[0]}x{src_size[1]} → {tgt_size[0]}x{tgt_size[1]})")

    if not args.quiet:
        print()
        print("=== Palette Ordering Benchmarks ===")
        print("  Comparing: colorize-then-downscale vs downscale-then-colorize")
        print()

    palette_tests = [
        ((2560, 2048), (1280, 1024), "ironbow"),
        ((2560, 2048), (1280, 1024), "rainbow"),
        ((2560, 2048), (960, 768), "ironbow"),
        ((1280, 1024), (640, 512), "ironbow"),
    ]

    for full_size, stream_size, cmap in palette_tests:
        result = benchmark_palette_ordering(full_size, stream_size, cmap, args.iterations)
        if not args.quiet:
            print(f"  {result['full_size']} → {result['stream_size']} ({cmap}):")
            print(f"    Palette at full res:     {result['palette_at_full_ms']:6.2f} ms")
            print(f"    Palette after downscale: {result['palette_after_downscale_ms']:6.2f} ms")
            print(f"    Speedup: {result['speedup']:.1f}x faster")
            print()

    if not args.quiet:
        print("=== JPEG Encode Benchmarks ===")

    jpeg_tests = [
        ("panel_full", (input_w * 4, input_h * 4)),
        ("stream_1280", (1280, 1024)),
        ("stream_960", (960, 768)),
    ]

    for name, size in jpeg_tests:
        test_frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        test_frame_bgr = cv2.cvtColor(test_frame, cv2.COLOR_GRAY2BGR)
        ms, kb = benchmark_jpeg(test_frame_bgr, iterations=args.iterations)
        if not args.quiet:
            print(f"  {name:15s}: {ms:6.2f} ms, {kb:6.1f} KB ({size[0]}x{size[1]})")

    if not args.quiet:
        print()
        print("=== Inference Benchmarks ===")

    for model_info in models:
        name = model_info["name"]
        path = model_info["path"]

        if not args.quiet:
            print(f"\n  Model: {name}")

        results = benchmark_inference(name, path, frame, args.device, args.fp16, args.iterations)

        if "error" in results:
            print(f"    Error: {results['error']}")
            continue

        if not args.quiet:
            print(f"    Input: {results['input_size']}, Output: {results['output_size']}, Scale: x{results['scale']}")
            print(f"    FP32 inference: {results['fp32_ms']:.2f} ms")
            print(f"    Copy back: {results['copy_back_ms']:.2f} ms")

            if "fp16_ms" in results:
                print(f"    FP16 inference: {results['fp16_ms']:.2f} ms")
                print(f"    FP16 max diff: {results['fp16_max_diff']:.2f} grey levels")
                print(f"    FP16 mean diff: {results['fp16_mean_diff']:.4f} grey levels")
            elif "fp16_error" in results:
                print(f"    FP16 error: {results['fp16_error']}")

        if "sr_frame" in results:
            import false_color
            false_color.ensure_luts()
            sr = results["sr_frame"]
            sr_bgr = false_color.apply_to_bgr(sr, "ironbow")

            ironbow_ms = benchmark_colormap(sr, "ironbow", args.iterations)
            jpeg_full_ms, jpeg_full_kb = benchmark_jpeg(sr_bgr, iterations=args.iterations)

            sr_stream = cv2.resize(sr_bgr, (1280, 1024), interpolation=cv2.INTER_AREA)
            jpeg_stream_ms, jpeg_stream_kb = benchmark_jpeg(sr_stream, iterations=args.iterations)

            if not args.quiet:
                print(f"    Ironbow LUT: {ironbow_ms:.2f} ms ({sr.shape[1]}x{sr.shape[0]})")
                print(f"    JPEG full: {jpeg_full_ms:.2f} ms, {jpeg_full_kb:.1f} KB")
                print(f"    JPEG stream: {jpeg_stream_ms:.2f} ms, {jpeg_stream_kb:.1f} KB (1280x1024)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Input frame: {input_w}x{input_h}")
    print(f"Device: {args.device}")
    print()
    print("Colormap timings (on 4x panel):")
    for cm, ms in colormap_results.items():
        print(f"  {cm}: {ms:.2f} ms")
    print()
    print("Key optimizations:")
    print("  1. Downscale grayscale FIRST, then apply palette (cv2.LUT)")
    print("     - Colorizing after downscale can be 3-5x faster for x4 models")
    print("     - cv2.LUT is constant-time regardless of image complexity")
    print("  2. Use stream downscaling to reduce encode/transfer cost")
    print("     - Full panel JPEG can be 5-10x slower than stream-sized JPEG")

    return 0


if __name__ == "__main__":
    sys.exit(main())
