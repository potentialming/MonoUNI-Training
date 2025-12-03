#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export MonoUNI model to ONNX format (no batch-size arg)
- Wrapper returns a tuple aligned with output_names
- Uses batch=1 dummy inputs for export; ONNX keeps dynamic batch dim
- ROI-type outputs tag dim-0 as num_dets; feature-map outputs tag dim-0 as batch_size
- Default opset_version=13
"""
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# Add project root to sys.path
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)  # scripts/ -> project root
sys.path.insert(0, ROOT_DIR)

from lib.helpers.model_helper import build_model


class MonoUNIWrapper(nn.Module):
    """
    Wrap MonoUNI to expose a fixed, ordered tuple of outputs for ONNX export.
    This exports 8 common outputs: heatmap/offset_2d/size_2d (feature-map) +
    offset_3d/size_3d/heading/vis_depth/att_depth (ROI).
    To export more (ins_depth/uncertainty/depth_bin/train_tag, etc.), extend the return tuple and output_names.
    """
    def __init__(self, model: nn.Module, mode: str = 'test', K: int = 50):
        super().__init__()
        assert mode in ['train', 'val', 'test']
        self.model = model
        self.mode = mode
        self.K = K

    def forward(self,
                image,               # [B, 3, H, W]
                coord_ranges,        # [B, 2, 2]
                calibs,              # [B, 3, 4]
                calib_pitch_sin,     # [B, 1]
                calib_pitch_cos):    # [B, 1]
        ret = self.model(
            image,
            coord_ranges,
            calibs,
            targets=None,
            K=self.K,
            mode=self.mode,
            calib_pitch_sin=calib_pitch_sin,
            calib_pitch_cos=calib_pitch_cos
        )

        # Pack into a fixed-order tuple (aligned with output_names)
        heatmap    = ret['heatmap']      # [B, C, H/4, W/4] (example)
        offset_2d  = ret['offset_2d']    # [B, 2, H/4, W/4]
        size_2d    = ret['size_2d']      # [B, 2, H/4, W/4]
        offset_3d  = ret['offset_3d']    # [N, 2]
        size_3d    = ret['size_3d']      # [N, 3]
        heading    = ret['heading']      # [N, 24]
        vis_depth  = ret['vis_depth']    # [N, 7, 7] or [N, 5, 7, 7]
        att_depth  = ret['att_depth']    # [N, 7, 7] or [N, 5, 7, 7]

        return (heatmap, offset_2d, size_2d, offset_3d, size_3d, heading, vis_depth, att_depth)


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support multiple checkpoint key layouts
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Strip DDP prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    print("Checkpoint loaded successfully!")
    return model


def export_onnx(
    model: nn.Module,
    output_path: str,
    input_size=(512, 960),
    opset_version: int = 13,
    simplify: bool = True
):
    print(f"\nExporting model to ONNX format...")
    print(f"Input size (HxW): {input_size}")
    print(f"Output path: {output_path}")

    device = next(model.parameters()).device
    B = 1  # Use batch=1 dummy; ONNX still keeps a dynamic batch axis
    C = 3
    H, W = input_size

    # -------------------------------
    # Build dummy inputs (batch=1)
    # -------------------------------
    image = torch.randn((B, C, H, W), dtype=torch.float32, device=device)

    # coord_ranges: [[x_min, y_min], [x_max, y_max]] in pixel coordinates
    coord_ranges = torch.tensor(
        [[[0.0, 0.0], [float(W), float(H)]]],
        dtype=torch.float32, device=device
    )  # [1, 2, 2]

    calibs = torch.randn(B, 3, 4, dtype=torch.float32, device=device)
    calib_pitch_sin = torch.randn(B, 1, dtype=torch.float32, device=device)
    calib_pitch_cos = torch.randn(B, 1, dtype=torch.float32, device=device)

    # -------------------------------
    # Dry-run raw model (dict) to log keys/shapes
    # -------------------------------
    print("\nTesting forward pass (raw model)...")
    model.eval()
    with torch.no_grad():
        try:
            raw_ret = model.model(
                image, coord_ranges, calibs,
                targets=None, K=50, mode='test',
                calib_pitch_sin=calib_pitch_sin,
                calib_pitch_cos=calib_pitch_cos
            )
            print("[OK] Raw forward pass. Output keys:", list(raw_ret.keys()))
            for k, v in raw_ret.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        except Exception as e:
            print(f"Warning: Raw forward failed: {e}")

    # -------------------------------
    # Export to ONNX (Wrapper returns tuple)
    # -------------------------------
    print("\nExporting to ONNX...")
    input_names = ['image', 'coord_ranges', 'calibs', 'calib_pitch_sin', 'calib_pitch_cos']
    output_names = [
        'heatmap', 'offset_2d', 'size_2d',
        'offset_3d', 'size_3d', 'heading',
        'vis_depth', 'att_depth'
    ]

    # Tag dynamic axes for feature-map outputs (batch_size) vs ROI outputs (num_dets)
    dynamic_axes = {
        # inputs
        'image': {0: 'batch_size'},  # add {2: 'in_h', 3: 'in_w'} to allow variable resolution
        'coord_ranges': {0: 'batch_size'},
        'calibs': {0: 'batch_size'},
        'calib_pitch_sin': {0: 'batch_size'},
        'calib_pitch_cos': {0: 'batch_size'},

        # feature-map type outputs
        'heatmap': {0: 'batch_size'},
        'offset_2d': {0: 'batch_size'},
        'size_2d': {0: 'batch_size'},

        # ROI type outputs (first dim is num_dets ~ B*K)
        'offset_3d': {0: 'num_dets'},
        'size_3d': {0: 'num_dets'},
        'heading': {0: 'num_dets'},
        'vis_depth': {0: 'num_dets'},
        'att_depth': {0: 'num_dets'},
    }

    try:
        torch.onnx.export(
            model,  # wrapper
            (image, coord_ranges, calibs, calib_pitch_sin, calib_pitch_cos),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        print(f"[OK] Model exported to: {output_path}")
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[OK] ONNX file size: {file_size_mb:.2f} MB")
    except Exception as e:
        import traceback
        print(f"[FAIL] ONNX export failed: {e}")
        traceback.print_exc()
        raise

    # -------------------------------
    # Optional: simplify ONNX
    # -------------------------------
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            print("\nSimplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            simplified_model, check = onnx_simplify(onnx_model)
            if check:
                simplified_path = output_path.replace('.onnx', '_simplified.onnx')
                onnx.save(simplified_model, simplified_path)
                size_mb = os.path.getsize(simplified_path) / (1024 * 1024)
                print(f"[OK] Simplified model saved to: {simplified_path}")
                print(f"[OK] Simplified file size: {size_mb:.2f} MB")
            else:
                print("[WARN] onnx-simplifier check failed (kept original ONNX).")
        except ImportError:
            print("\nNote: to simplify the model, install:")
            print("  pip install onnx onnx-simplifier")
        except Exception as e:
            print(f"Warning: simplification failed: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description='Export MonoUNI model to ONNX format (no batch-size arg)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file (default: config/config.yaml)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ONNX path (default: same dir as checkpoint, same name)')
    parser.add_argument('--input-size', type=str, default='512,960',
                        help='Input size as H,W (default: 512,960)')
    parser.add_argument('--opset-version', type=int, default=13,
                        help='ONNX opset version (default: 13)')
    parser.add_argument('--no-simplify', action='store_true',
                        help='Do not run onnx-simplifier')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--K', type=int, default=50,
                        help='TopK proposals used in test path (default: 50)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Switch to project root for relative paths
    os.chdir(ROOT_DIR)

    # Basic checks
    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        return
    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        return

    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Input size
    in_h, in_w = map(int, args.input_size.split(','))
    input_size = (in_h, in_w)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build model (cls_mean_size must match dataset classes)
    print("\nBuilding model...")
    cls_mean_size = np.array([
        [1.288762253204939, 1.6939648801353426, 4.25589251897889],   # car
        [1.7199308570318539, 1.7356837654961508, 4.641152817981265], # big_vehicle
        [2.682263889273618, 2.3482764551684268, 6.940250839428722],  # pedestrian? (example)
        [2.9588510594399073, 2.5199248789610693, 10.542197736838778] # cyclist?  (example)
    ], dtype=np.float32)

    model_core = build_model(cfg['model'], cls_mean_size)
    model_core = model_core.to(device)

    # Load checkpoint
    model_core = load_checkpoint(model_core, args.checkpoint, device)

    # Wrap for test path with fixed K
    wrapped = MonoUNIWrapper(model_core, mode='test', K=args.K).to(device)
    wrapped.eval()

    # Resolve output path
    if args.output is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = os.path.join(ckpt_dir, f"{ckpt_name}.onnx")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Export
    export_onnx(
        wrapped,
        args.output,
        input_size=input_size,
        opset_version=args.opset_version,
        simplify=not args.no_simplify
    )

    print("\n" + "=" * 60)
    print("Export completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
