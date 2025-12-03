#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export MonoUNI ONNX model to TensorRT engine
- Supports FP16/FP32; FP16 traces upstream from vis_depth and forces FP32 to avoid overflow
- Supports dynamic batch (fixed input resolution 512x960)
"""
import os
import sys
import argparse
import tensorrt as trt

# Project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def load_tensorrt_plugins():
    """Load TensorRT plugin libraries"""
    import ctypes
    plugin_paths = [
        '/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so',
        '/usr/local/lib/libnvinfer_plugin.so',
        'libnvinfer_plugin.so'
    ]
    for p in plugin_paths:
        try:
            ctypes.CDLL(p)
            print(f"[OK] Loaded TensorRT plugin library: {p}")
            break
        except OSError:
            continue
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    print("[OK] TensorRT plugins initialized")


def build_engine(
    onnx_path,
    engine_path,
    precision="fp16",
    max_batch_size=1,
    workspace_size=4,
    input_size=(512, 960),
):
    """Build a TensorRT engine from ONNX"""
    print("\n" + "=" * 60)
    print("Building TensorRT Engine")
    print("=" * 60)
    print(f"ONNX model: {onnx_path}")
    print(f"Engine output: {engine_path}")
    print(f"Precision: {precision.upper()}")
    print(f"Max batch size: {max_batch_size}")
    print(f"Workspace size: {workspace_size} GB")

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    load_tensorrt_plugins()

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print("\nParsing ONNX model...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX model")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False
    print("[OK] ONNX model parsed successfully")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30))

    # Optimization profile (fixed H,W, dynamic batch)
    h, w = input_size
    profile = builder.create_optimization_profile()
    shape_inputs = {
        "image": (max_batch_size, 3, h, w),
        "coord_ranges": (max_batch_size, 2, 2),
        "calibs": (max_batch_size, 3, 4),
        "calib_pitch_sin": (max_batch_size, 1),
        "calib_pitch_cos": (max_batch_size, 1),
    }
    for name, shape in shape_inputs.items():
        if network.get_input(0) is None:
            continue
        min_shape = tuple([1] + list(shape)[1:])
        profile.set_shape(name, min_shape, shape, shape)
    config.add_optimization_profile(profile)

    # Select precision
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        config.clear_flag(trt.BuilderFlag.TF32)
        # Enforce FP32 requests where specified
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        print("[OK] FP16 enabled (TF32 disabled)")
    else:
        print("[OK] FP32 mode")

    # Helper: try to force a layer to FP32
    def set_layer_fp32(layer, reason):
        allowed = {
            trt.LayerType.CONVOLUTION,
            trt.LayerType.DECONVOLUTION,
            trt.LayerType.ACTIVATION,
            trt.LayerType.ELEMENTWISE,
            trt.LayerType.SOFTMAX,
            trt.LayerType.REDUCE,
            trt.LayerType.TOPK,
            trt.LayerType.SCALE,
            trt.LayerType.MATRIX_MULTIPLY,
            trt.LayerType.UNARY,
            trt.LayerType.POOLING,
        }
        if layer.type not in allowed:
            return
        try:
            layer.precision = trt.DataType.FLOAT
            for i in range(layer.num_outputs):
                t = layer.get_output(i)
                if t and t.dtype in (trt.DataType.FLOAT, trt.DataType.HALF):
                    layer.set_output_type(i, trt.DataType.FLOAT)
            print(f"  [fp32] {layer.name} ({layer.type}) <- {reason}")
        except Exception as e:
            print(f"  [fp32-skip] {layer.name}: {e}")

    if precision == "fp16":
        # Auto: trace upstream only from vis_depth and force FP32 to avoid overflow
        tensor_to_layer = {}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for j in range(layer.num_outputs):
                t = layer.get_output(j)
                if t and t.name:
                    tensor_to_layer[t.name] = layer

        def collect_upstream(out_names):
            collected = set()
            q = list(out_names)
            while q:
                tname = q.pop()
                layer = tensor_to_layer.get(tname)
                if not layer or layer in collected:
                    continue
                collected.add(layer)
                for k in range(layer.num_inputs):
                    tin = layer.get_input(k)
                    if tin and not tin.is_network_input and tin.name:
                        q.append(tin.name)
            return collected

        target_outputs = {"vis_depth"}

        print("\nAuto FP32 (upstream of vis_depth only):")
        for layer in collect_upstream(target_outputs):
            set_layer_fp32(layer, "depth-upstream")

    print("\nBuilding engine... (may take minutes)")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return False

    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"[OK] Engine saved: {engine_path}")
    print(f"[OK] Engine size: {size_mb:.2f} MB")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Export MonoUNI ONNX model to TensorRT engine")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model file")
    parser.add_argument("--output", type=str, default=None, help="Engine output path")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"],
                        help="Precision mode (default: fp16)")
    parser.add_argument("--max-batch-size", type=int, default=1, help="Max batch size (default: 1)")
    parser.add_argument("--workspace", type=int, default=4, help="Workspace size (GB, default: 4)")
    parser.add_argument("--input-size", type=str, default="512,960", help="Input H,W (default: 512,960)")
    return parser.parse_args()


def main():
    args = parse_args()

    h, w = map(int, args.input_size.split(","))

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.onnx))[0]
        if args.precision == "fp16":
            suffix = "_fp16"
        else:
            suffix = "_fp32"
        engine_path = os.path.join(ROOT_DIR, "checkpoints", base + suffix + ".engine")
    else:
        engine_path = args.output

    ok = build_engine(
        onnx_path=args.onnx,
        engine_path=engine_path,
        precision=args.precision,
        max_batch_size=args.max_batch_size,
        workspace_size=args.workspace,
        input_size=(h, w),
    )

    if ok:
        print("\n" + "=" * 60)
        print("Export completed successfully!")
        print("=" * 60)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
