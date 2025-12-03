# MonoUNI (TensorRT / ONNX / DAIR-V2X)

Derived from the NeurIPS 2023 MonoUNI monocular 3D detector (upstream: https://github.com/Traffic-X/MonoUNI). This repo focuses on practical tooling and dataset coverage for DAIR-V2X and KITTI, with both Python and C++ pipelines.

## Highlights
- DAIR-V2X support end-to-end: configs, dataloader, evaluation.
- Ground-plane utilities: plane-equation scripts, dense depth cube generation, and visualization helpers.
- ONNX workflow: export script and ONNX evaluation script.
- TensorRT workflow: export + eval; FP32 is near-lossless, FP16 has some drop and currently little speedup—community help wanted.
- C++ implementations provided; related repos (to be open-sourced): [monouni_cpp_onnx](https://github.com/monouni_cpp_onnx), [monouni_cpp_trt](https://github.com/monouni_cpp_trt).

## Quick start (Python)
```bash
# Export ONNX
python scripts/export_onnx.py --config config/config_kitti.yaml --checkpoint <ckpt> --output <out.onnx>

# Export TensorRT (FP32 or FP16)
python scripts/export_tensorrt.py --onnx <out.onnx> --precision fp32 --max-batch-size 1
python scripts/export_tensorrt.py --onnx <out.onnx> --precision fp16 --max-batch-size 1

# Evaluate TensorRT
python scripts/eval_tensorrt.py --config config/config_kitti.yaml --engine <engine> --batch-size 1
```

## Precision notes
- FP32: accuracy is effectively unchanged versus PyTorch.
- FP16: measurable accuracy loss; current build shows minimal speedup. Contributions to improve FP16 stability/perf are welcome.

## Thanks and contributions
Issues and PRs are appreciated—especially for FP16 stability/speed and broader dataset coverage.***
