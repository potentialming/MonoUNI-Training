#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate MonoUNI TensorRT engine (TensorRT 10 API, primary CUDA context, plugin bootstrap)

- TRT 10 Python API: num_io_tensors / get_tensor_name / set_input_shape /
  set_tensor_address / execute_async_v3
- Uses primary CUDA context (avoid Numba "non-primary context" error)
- Proactively loads TRT shared libs and initializes plugin registry
"""

import os
import sys
import argparse
import yaml
import tqdm
import numpy as np
import ctypes
import glob

# Project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

import torch
from torch.utils.data import DataLoader

from numba import cuda as nb_cuda
import pycuda.driver as cuda  # no autoinit
import tensorrt as trt

from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections
import lib.eval_tools.eval as eval


# -------- Primary CUDA Context --------

class CUDAPrimaryContext:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.ctx = None

    def __enter__(self):
        nb_cuda.select_device(self.device_id)
        cuda.init()
        dev = cuda.Device(self.device_id)
        self.ctx = dev.retain_primary_context()
        self.ctx.push()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.ctx is not None:
            self.ctx.pop()
            self.ctx.detach()
        self.ctx = None


# -------- TRT plugin bootstrap --------

def _dlopen(path: str):
    return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

def _find_trt_libs_from_python_pkg():
    try:
        import tensorrt_libs
        libdir = os.path.dirname(tensorrt_libs.__file__)
        cand = {
            "libnvinfer":        sorted(glob.glob(os.path.join(libdir, "libnvinfer.so*")), reverse=True),
            "libnvinfer_plugin": sorted(glob.glob(os.path.join(libdir, "libnvinfer_plugin.so*")), reverse=True),
            "libnvinfer_builder_resource": sorted(glob.glob(os.path.join(libdir, "libnvinfer_builder_resource.so*")), reverse=True),
        }
        return cand
    except Exception:
        return {}

def ensure_trt_plugins(logger=None, verbose=True):
    if verbose: print("[TRT-Plugin] Calling init_libnvinfer_plugins(...)")
    try:
        trt.init_libnvinfer_plugins(logger or trt.Logger(trt.Logger.ERROR), "")
    except Exception as e:
        if verbose: print(f"[TRT-Plugin] init_libnvinfer_plugins failed (continue): {e}")

    reg = trt.get_plugin_registry()
    creators = reg.plugin_creator_list
    names = [(c.name, c.plugin_version, c.plugin_namespace) for c in creators]
    have_roi = [x for x in names if x[0] == "ROIAlign_TRT"]

    if not have_roi:
        if verbose: print("[TRT-Plugin] ROIAlign_TRT not found; dlopen libs from tensorrt_libs ...")
        libs = _find_trt_libs_from_python_pkg()
        for key in ["libnvinfer", "libnvinfer_builder_resource", "libnvinfer_plugin"]:
            for p in libs.get(key, []):
                try:
                    if verbose: print(f"[TRT-Plugin] dlopen: {p}")
                    _dlopen(p)
                except Exception as e:
                    if verbose: print(f"[TRT-Plugin] dlopen failed for {p}: {e}")
        try:
            if verbose: print("[TRT-Plugin] Re-calling init_libnvinfer_plugins(...)")
            trt.init_libnvinfer_plugins(logger or trt.Logger(trt.Logger.ERROR), "")
        except Exception as e:
            if verbose: print(f"[TRT-Plugin] init_libnvinfer_plugins failed again: {e}")
        reg = trt.get_plugin_registry()
        creators = reg.plugin_creator_list
        names = [(c.name, c.plugin_version, c.plugin_namespace) for c in creators]
        have_roi = [x for x in names if x[0] == "ROIAlign_TRT"]

    if verbose:
        roi_related = [x for x in names if "ROI" in x[0]]
        print(f"[TRT-Plugin] ROI-related creators: {roi_related}")
    return have_roi  # list of tuples (name, version, ns)


# -------- TensorRT 10 Runtime Wrapper --------

class TRTModelWrapper:
    """
    TensorRT 10 wrapper (no legacy bindings).
    Inputs:
      image [B,3,H,W], coord_ranges [B,2,2], calibs [B,3,4], calib_pitch_sin [B,1], calib_pitch_cos [B,1]
    Outputs: 8 heads (heatmap, offset_2d, size_2d, offset_3d, size_3d, heading, vis_depth, att_depth)
    """

    OUTPUTS = ["heatmap", "offset_2d", "size_2d", "offset_3d", "size_3d", "heading", "vis_depth", "att_depth"]

    def __init__(self, engine_path: str, device_stream=None):
        assert os.path.exists(engine_path), f"Engine not found: {engine_path}"
        print(f"Loading TensorRT engine from: {engine_path}")

        logger = trt.Logger(trt.Logger.ERROR)
        roi_creators = ensure_trt_plugins(logger=logger, verbose=True)

        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            eng_bytes = f.read()
        self.engine = runtime.deserialize_cuda_engine(eng_bytes)
        if self.engine is None:
            versions = ", ".join([f"{n} v{v}" for n, v, _ in roi_creators])
            raise RuntimeError(
                "Failed to deserialize engine. Likely plugin version mismatch "
                f"(runtime has: {versions or 'None'})."
            )

        self.context = self.engine.create_execution_context()
        assert self.context is not None, "Failed to create execution context"

        # Cache I/O tensor names and modes (TRT 10)
        self.inputs = []
        self.outputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)

        print("Inputs:", self.inputs)
        print("Outputs:", self.outputs)

        # Device buffers & sizes
        self.dptr = {}    # name -> DeviceAllocation
        self.dbytes = {}  # name -> int (bytes)
        self.stream = device_stream or cuda.Stream()

    @staticmethod
    def _np_f32(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.ascontiguousarray(x.astype(np.float32))

    def _alloc(self, name, nbytes):
        cur = self.dbytes.get(name, 0)
        if cur < nbytes or name not in self.dptr:
            if name in self.dptr:
                try:
                    self.dptr[name].free()
                except Exception:
                    pass
            self.dptr[name] = cuda.mem_alloc(nbytes)
            self.dbytes[name] = nbytes
        return self.dptr[name]

    def __call__(self, inputs, coord_ranges, calibs, K=50, mode='val',
                 calib_pitch_sin=None, calib_pitch_cos=None):

        # Host numpy
        image = self._np_f32(inputs)
        coord_ranges = self._np_f32(coord_ranges)
        calibs = self._np_f32(calibs)
        if calib_pitch_sin is None or calib_pitch_cos is None:
            raise ValueError("calib_pitch_sin/cos required")
        calib_pitch_sin = self._np_f32(calib_pitch_sin)
        calib_pitch_cos = self._np_f32(calib_pitch_cos)
        if calib_pitch_sin.ndim == 1:
            calib_pitch_sin = calib_pitch_sin.reshape(-1, 1)
        if calib_pitch_cos.ndim == 1:
            calib_pitch_cos = calib_pitch_cos.reshape(-1, 1)

        B, C, H, W = image.shape

        # Set input shapes
        self.context.set_input_shape("image", (B, C, H, W))
        self.context.set_input_shape("coord_ranges", (B, 2, 2))
        self.context.set_input_shape("calibs", (B, 3, 4))
        self.context.set_input_shape("calib_pitch_sin", (B, 1))
        self.context.set_input_shape("calib_pitch_cos", (B, 1))

        # H2D & set tensor addresses
        host_inputs = {
            "image": image,
            "coord_ranges": coord_ranges,
            "calibs": calibs,
            "calib_pitch_sin": calib_pitch_sin,
            "calib_pitch_cos": calib_pitch_cos,
        }
        for name, arr in host_inputs.items():
            d = self._alloc(name, arr.nbytes)
            cuda.memcpy_htod_async(d, arr, self.stream)
            self.context.set_tensor_address(name, int(d))

        # Prepare outputs by current binding shapes
        host_outputs = {}
        for name in self.outputs:
            shp = tuple(self.context.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            arr = np.empty(shp, dtype=dtype)
            host_outputs[name] = arr
            d = self._alloc(name, arr.nbytes)
            self.context.set_tensor_address(name, int(d))

        # Execute
        ok = self.context.execute_async_v3(self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3() failed")
        self.stream.synchronize()

        # D2H
        for name, arr in host_outputs.items():
            d = self.dptr[name]
            cuda.memcpy_dtoh_async(arr, d, self.stream)
        self.stream.synchronize()

        # Convert to torch; cast to float32 to match downstream expectations
        out = {}
        for k, v in host_outputs.items():
            t = torch.from_numpy(v)
            if t.dtype != torch.float32:
                t = t.float()
            out[k] = t

        # Reshape heads
        batch_size = out["heatmap"].shape[0]

        for k in ["vis_depth", "att_depth"]:
            if k in out:
                shape = out[k].shape
                if len(shape) == 3 and shape[0] != batch_size:
                    kk = shape[0] // batch_size
                    out[k] = out[k].view(batch_size, kk, 7, 7)

        for k in ["offset_3d", "size_3d", "heading"]:
            if k in out:
                shape = out[k].shape
                if shape[0] != batch_size:
                    kk = shape[0] // batch_size
                    rest = shape[1:] if len(shape) > 1 else ()
                    out[k] = out[k].view(batch_size, kk, *rest)

        # Required by extract_dets_from_outputs
        kk = 50
        out["ins_depth_uncer"] = torch.ones(batch_size * kk * 7 * 7, dtype=torch.float32) * 0.1
        return out


# -------- Tester --------

class TRTTester:
    def __init__(self, cfg, trt_model, data_loader, logger=None):
        self.cfg = cfg['tester']
        self.eval_cls = cfg['dataset']['eval_cls']
        self.root_dir = cfg['dataset']['root_dir']

        dataset_type = cfg['dataset'].get('type', 'rope3d').lower()
        if dataset_type == 'kitti':
            self.label_dir = os.path.join(self.root_dir, 'training', 'label_2')
            self.calib_dir = os.path.join(self.root_dir, 'training', 'calib')
            self.de_norm_dir = os.path.join(self.root_dir, 'training', 'denorm')
        else:
            self.label_dir = os.path.join(self.root_dir, 'label_2_4cls_filter_with_roi_for_eval')
            self.calib_dir = os.path.join(self.root_dir, 'calib')
            self.de_norm_dir = os.path.join(self.root_dir, 'denorm')

        self.model = trt_model
        self.data_loader = data_loader
        self.logger = logger
        self.class_name = data_loader.dataset.class_name

        print("\nTRT Tester initialized:")
        print(f"  Dataset type: {dataset_type}")
        print(f"  Root dir: {self.root_dir}")
        print(f"  Label dir: {self.label_dir}")
        print(f"  Eval classes: {self.eval_cls}")
        print(f"  Threshold: {self.cfg['threshold']}")

    def test(self):
        results = {}
        pbar = tqdm.tqdm(total=len(self.data_loader), leave=True, desc='TRT Evaluation Progress')

        for _, batch_data in enumerate(self.data_loader):
            inputs, calibs, coord_ranges, _, info, calib_pitch_cos, calib_pitch_sin = batch_data

            outputs = self.model(
                inputs,
                coord_ranges,
                calibs,
                K=50,
                mode='val',
                calib_pitch_sin=calib_pitch_sin,
                calib_pitch_cos=calib_pitch_cos
            )

            dets = extract_dets_from_outputs(outputs, calibs, K=50).detach().cpu().numpy()

            calibs_list = [self.data_loader.dataset.get_calib(i) for i in info['img_id']]
            denorms = [self.data_loader.dataset.get_denorm(i) for i in info['img_id']]

            info['img_id'] = info['img_id']
            info['img_size'] = info['img_size'].detach().cpu().numpy()
            info['bbox_downsample_ratio'] = info['bbox_downsample_ratio'].detach().cpu().numpy()

            cls_mean_size = self.data_loader.dataset.cls_mean_size
            dets_decoded = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs_list,
                denorms=denorms,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg['threshold']
            )

            results.update(dets_decoded)
            pbar.update()

        pbar.close()

        out_dir = os.path.join(self.cfg['out_dir'])
        self.save_results(results, out_dir)

        use_roi_filter = self.cfg.get('use_roi_filter', True)
        print("\nRunning evaluation...")
        print(f"  Label dir: {self.label_dir}")
        print(f"  Results dir: {os.path.join(out_dir, 'data')}")
        print(f"  Use ROI filter: {use_roi_filter}")

        eval_results = eval.do_repo3d_eval(
            self.logger,
            self.label_dir,
            os.path.join(out_dir, 'data'),
            self.calib_dir,
            self.de_norm_dir,
            self.eval_cls,
            ap_mode=40,
            use_roi_filter=use_roi_filter
        )
        return eval_results

    def save_results(self, results, output_dir='./outputs'):
        output_dir = os.path.join(output_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving results to: {output_dir}")

        for img_id in results.keys():
            out_path = os.path.join(output_dir, img_id + '.txt')
            with open(out_path, 'w') as f:
                for i in range(len(results[img_id])):
                    class_name = self.class_name[int(results[img_id][i][0])]
                    f.write('{} 0.0 0'.format(class_name))
                    for j in range(1, len(results[img_id][i])):
                        f.write(' {:.2f}'.format(results[img_id][i][j]))
                    f.write('\n')
        print(f"Saved {len(results)} result files")


# -------- DataLoader / Logger / CLI --------

def create_dataloader(cfg):
    dataset_type = cfg['dataset'].get('type', 'rope3d').lower()
    if dataset_type == 'kitti':
        from lib.datasets.kitti import KITTI
        dataset = KITTI(cfg['dataset']['root_dir'], 'val', cfg['dataset'])
    else:
        from lib.datasets.rope3d import Rope3D
        dataset = Rope3D(cfg['dataset']['root_dir'], 'val', cfg['dataset'])

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg['dataset'].get('batch_size', 1),
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    print("\nDataLoader created:")
    print(f"  Dataset: {dataset_type}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Batch size: {cfg['dataset'].get('batch_size', 1)}")
    print(f"  Number of batches: {len(dataloader)}")
    return dataloader


class SimpleLogger:
    def info(self, message):
        print(message)


def parse_args():
    pa = argparse.ArgumentParser(description='Evaluate MonoUNI TensorRT engine (TRT10, primary context, plugins)')
    pa.add_argument('--config', type=str, default='config/config.yaml', help='Path to config')
    pa.add_argument('--engine', type=str, required=True, help='Path to .engine')
    pa.add_argument('--output-dir', type=str, default=None, help='Override output dir')
    pa.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    pa.add_argument('--threshold', type=float, default=None, help='Override threshold')
    pa.add_argument('--dataset-type', type=str, default=None, choices=['rope3d', 'kitti'], help='Override dataset type')
    pa.add_argument('--dataset-root', type=str, default=None, help='Override dataset root')
    pa.add_argument('--device', type=int, default=0, help='CUDA device id')
    return pa.parse_args()


def main():
    args = parse_args()

    with CUDAPrimaryContext(device_id=args.device):
        os.chdir(ROOT_DIR)
        print(f"Working directory: {os.getcwd()}")

        if not os.path.exists(args.engine):
            print(f"Error: engine not found: {args.engine}")
            return
        if not os.path.exists(args.config):
            print(f"Error: config not found: {args.config}")
            return

        print(f"\nLoading config from: {args.config}")
        with open(args.config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        if args.output_dir:   cfg['tester']['out_dir'] = args.output_dir
        if args.batch_size:   cfg['dataset']['batch_size'] = args.batch_size
        if args.threshold:    cfg['tester']['threshold'] = args.threshold
        if args.dataset_type: cfg['dataset']['type'] = args.dataset_type
        if args.dataset_root: cfg['dataset']['root_dir'] = args.dataset_root

        print("\nConfiguration:")
        print(f"  Dataset: {cfg['dataset']['type']}")
        print(f"  Dataset root: {cfg['dataset']['root_dir']}")
        print(f"  Batch size: {cfg['dataset'].get('batch_size', 1)}")
        print(f"  Threshold: {cfg['tester']['threshold']}")
        print(f"  Output dir: {cfg['tester']['out_dir']}")

        print("\n" + "="*60)
        print("Loading TensorRT Engine")
        print("="*60)
        trt_model = TRTModelWrapper(args.engine)

        print("\n" + "="*60)
        print("Creating DataLoader")
        print("="*60)
        dataloader = create_dataloader(cfg)

        logger = SimpleLogger()

        print("\n" + "="*60)
        print("Starting Evaluation (TRT)")
        print("="*60)
        tester = TRTTester(cfg, trt_model, dataloader, logger)

        results = tester.test()

        print("\n" + "="*60)
        print("Evaluation Complete!")
        print("="*60)
        return results


if __name__ == '__main__':
    main()
