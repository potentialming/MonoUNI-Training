"""
Evaluate MonoUNI ONNX model
Based on tester_helper.py but using ONNX Runtime for inference
"""
import os
import sys
import argparse
import yaml
import tqdm
import numpy as np
import onnxruntime as ort

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

import torch
from torch.utils.data import DataLoader

from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections
import lib.eval_tools.eval as eval


class ONNXModelWrapper:
    """Wrapper for ONNX model to match PyTorch model interface"""
    
    def __init__(self, onnx_path, device='cuda'):
        """
        Initialize ONNX Runtime session
        
        Args:
            onnx_path: Path to ONNX model file
            device: 'cuda' or 'cpu'
        """
        print(f"Loading ONNX model from: {onnx_path}")
        
        # Set up ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input/output info
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"Model loaded successfully!")
        print(f"Input names: {self.input_names}")
        print(f"Output names: {self.output_names}")
        
        # Check which provider is actually being used
        print(f"Execution providers: {self.session.get_providers()}")
        
    def __call__(self, inputs, coord_ranges, calibs, K=50, mode='val', 
                 calib_pitch_sin=None, calib_pitch_cos=None):
        """
        Run inference on ONNX model
        
        Args:
            inputs: [B, 3, H, W] input image tensor
            coord_ranges: [B, 2, 2] coordinate ranges
            calibs: [B, 3, 4] camera calibration matrices
            K: number of top detections
            mode: inference mode (not used in ONNX)
            calib_pitch_sin: [B, 1] sine of camera pitch
            calib_pitch_cos: [B, 1] cosine of camera pitch
            
        Returns:
            Dictionary of predictions (same format as PyTorch model)
        """
        # Convert PyTorch tensors to numpy
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.cpu().numpy()
        if isinstance(coord_ranges, torch.Tensor):
            coord_ranges = coord_ranges.cpu().numpy()
        if isinstance(calibs, torch.Tensor):
            calibs = calibs.cpu().numpy()
        if isinstance(calib_pitch_sin, torch.Tensor):
            calib_pitch_sin = calib_pitch_sin.cpu().numpy()
        if isinstance(calib_pitch_cos, torch.Tensor):
            calib_pitch_cos = calib_pitch_cos.cpu().numpy()
        
        # Ensure pitch tensors have the right shape [B, 1]
        if calib_pitch_sin.ndim == 1:
            calib_pitch_sin = calib_pitch_sin.reshape(-1, 1)
        if calib_pitch_cos.ndim == 1:
            calib_pitch_cos = calib_pitch_cos.reshape(-1, 1)
        
        # Prepare input dictionary
        input_dict = {
            'image': inputs.astype(np.float32),
            'coord_ranges': coord_ranges.astype(np.float32),
            'calibs': calibs.astype(np.float32),
            'calib_pitch_sin': calib_pitch_sin.astype(np.float32),
            'calib_pitch_cos': calib_pitch_cos.astype(np.float32)
        }
        
        # Run inference - get all outputs
        # ONNX exports: output[12]='1591' is offset_3d, output[13]='1332' is size_3d
        all_outputs = self.session.run(None, input_dict)
        
        # Map outputs by index (new ONNX export has 8 outputs)
        output_mapping = {
            0: 'heatmap',
            1: 'offset_2d',
            2: 'size_2d',
            3: 'offset_3d',
            4: 'size_3d',
            5: 'heading',
            6: 'vis_depth',
            7: 'att_depth'
        }
        
        # Convert to dictionary
        output_dict = {}
        for idx, name in output_mapping.items():
            tensor = torch.from_numpy(all_outputs[idx])
            output_dict[name] = tensor
        
        # Get batch size for reshaping
        batch_size = output_dict['heatmap'].shape[0]
        
        # Reshape depth outputs from [batch*K, 7, 7] to [batch, K, 7, 7]
        for key in ['vis_depth', 'att_depth']:
            if key in output_dict:
                shape = output_dict[key].shape
                if len(shape) == 3 and shape[0] != batch_size:
                    K = shape[0] // batch_size
                    output_dict[key] = output_dict[key].view(batch_size, K, 7, 7)
        
        # Reshape other outputs from [batch*K, ...] to [batch, K, ...]
        for key in ['offset_3d', 'size_3d', 'heading']:
            if key in output_dict:
                shape = output_dict[key].shape
                if shape[0] != batch_size:
                    K = shape[0] // batch_size
                    remaining_dims = shape[1:] if len(shape) > 1 else ()
                    output_dict[key] = output_dict[key].view(batch_size, K, *remaining_dims)
        
        # Add ins_depth_uncer (required by extract_dets_from_outputs)
        K = 50
        output_dict['ins_depth_uncer'] = torch.ones(batch_size * K * 7 * 7) * 0.1
        
        return output_dict


class ONNXTester:
    """ONNX Model Tester - similar to Tester in tester_helper.py"""
    
    def __init__(self, cfg, onnx_model, data_loader, logger=None):
        """
        Initialize ONNX Tester
        
        Args:
            cfg: Configuration dictionary
            onnx_model: ONNXModelWrapper instance
            data_loader: PyTorch DataLoader
            logger: Logger instance (optional)
        """
        self.cfg = cfg['tester']
        self.eval_cls = cfg['dataset']['eval_cls']
        self.root_dir = cfg['dataset']['root_dir']
        
        # Set label directory based on dataset type
        dataset_type = cfg['dataset'].get('type', 'rope3d').lower()
        if dataset_type == 'kitti':
            self.label_dir = os.path.join(self.root_dir, 'training', 'label_2')
            self.calib_dir = os.path.join(self.root_dir, 'training', 'calib')
            self.de_norm_dir = os.path.join(self.root_dir, 'training', 'denorm')
        else:
            self.label_dir = os.path.join(self.root_dir, 'label_2_4cls_filter_with_roi_for_eval')
            self.calib_dir = os.path.join(self.root_dir, 'calib')
            self.de_norm_dir = os.path.join(self.root_dir, 'denorm')
        
        self.model = onnx_model
        self.data_loader = data_loader
        self.logger = logger
        self.class_name = data_loader.dataset.class_name
        
        print(f"\nONNX Tester initialized:")
        print(f"  Dataset type: {dataset_type}")
        print(f"  Root dir: {self.root_dir}")
        print(f"  Label dir: {self.label_dir}")
        print(f"  Eval classes: {self.eval_cls}")
        print(f"  Threshold: {self.cfg['threshold']}")
        
    def test(self):
        """Run evaluation on test/validation dataset"""
        results = {}
        progress_bar = tqdm.tqdm(
            total=len(self.data_loader), 
            leave=True, 
            desc='ONNX Evaluation Progress'
        )
        
        for batch_idx, batch_data in enumerate(self.data_loader):
            # Unpack batch data
            inputs, calibs, coord_ranges, _, info, calib_pitch_cos, calib_pitch_sin = batch_data
            
            # Run ONNX inference
            outputs = self.model(
                inputs, 
                coord_ranges, 
                calibs, 
                K=50, 
                mode='val',
                calib_pitch_sin=calib_pitch_sin,
                calib_pitch_cos=calib_pitch_cos
            )
            
            # Extract detections (same as PyTorch version)
            
            dets = extract_dets_from_outputs(outputs, calibs, K=50)
            # Debug: Output TopK for image 000018
            if info["img_id"][0] == "000018":
                class_names = ["car", "big_vehicle", "pedestrian", "cyclist"]
                print("\nPython TopK detections for 000018:")
                for k in range(50):
                    cls_id = int(dets[0, k, 0].item())
                    score = dets[0, k, 1].item()
                    print(f"{class_names[cls_id]} {score:.6f}")
            dets = dets.detach().cpu().numpy()
            
            # Get calibrations and denorms
            calibs_list = [
                self.data_loader.dataset.get_calib(index) 
                for index in info['img_id']
            ]
            denorms = [
                self.data_loader.dataset.get_denorm(index) 
                for index in info['img_id']
            ]
            
            # Prepare info dict
            info['img_id'] = info['img_id']
            info['img_size'] = info['img_size'].detach().cpu().numpy()
            info['bbox_downsample_ratio'] = info['bbox_downsample_ratio'].detach().cpu().numpy()
            
            # Decode detections
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
            progress_bar.update()
        
        progress_bar.close()
        
        # Save results
        out_dir = os.path.join(self.cfg['out_dir'])
        self.save_results(results, out_dir)
        
        # Run evaluation
        use_roi_filter = self.cfg.get('use_roi_filter', True)
        
        print(f"\nRunning evaluation...")
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
        """Save detection results to files"""
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


def create_dataloader(cfg):
    """Create data loader based on dataset configuration"""
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
    
    print(f"\nDataLoader created:")
    print(f"  Dataset: {dataset_type}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Batch size: {cfg['dataset'].get('batch_size', 1)}")
    print(f"  Number of batches: {len(dataloader)}")
    
    return dataloader


class SimpleLogger:
    """Simple logger for printing messages"""
    
    def info(self, message):
        print(message)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate MonoUNI ONNX model performance'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--onnx',
        type=str,
        required=True,
        help='Path to ONNX model file (.onnx)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Confidence threshold (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference'
    )
    parser.add_argument(
        '--dataset-type',
        type=str,
        default=None,
        choices=['rope3d', 'kitti'],
        help='Dataset type (overrides config)'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        default=None,
        help='Dataset root directory (overrides config)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Change to project root
    os.chdir(ROOT_DIR)
    print(f"Working directory: {os.getcwd()}")
    
    # Check ONNX model exists
    if not os.path.exists(args.onnx):
        print(f"Error: ONNX model not found: {args.onnx}")
        return
    
    # Check config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    # Load configuration
    print(f"\nLoading config from: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    
    # Override config with command line arguments
    if args.output_dir:
        cfg['tester']['out_dir'] = args.output_dir
    if args.batch_size:
        cfg['dataset']['batch_size'] = args.batch_size
    if args.threshold:
        cfg['tester']['threshold'] = args.threshold
    if args.dataset_type:
        cfg['dataset']['type'] = args.dataset_type
    if args.dataset_root:
        cfg['dataset']['root_dir'] = args.dataset_root
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {cfg['dataset']['type']}")
    print(f"  Dataset root: {cfg['dataset']['root_dir']}")
    print(f"  Batch size: {cfg['dataset'].get('batch_size', 1)}")
    print(f"  Threshold: {cfg['tester']['threshold']}")
    print(f"  Output dir: {cfg['tester']['out_dir']}")
    print(f"  Device: {args.device}")
    
    # Load ONNX model
    print(f"\n{'='*60}")
    print("Loading ONNX Model")
    print('='*60)
    onnx_model = ONNXModelWrapper(args.onnx, device=args.device)
    
    # Create data loader
    print(f"\n{'='*60}")
    print("Creating DataLoader")
    print('='*60)
    dataloader = create_dataloader(cfg)
    
    # Create logger
    logger = SimpleLogger()
    
    # Create tester
    print(f"\n{'='*60}")
    print("Starting Evaluation")
    print('='*60)
    tester = ONNXTester(cfg, onnx_model, dataloader, logger)
    
    # Run evaluation
    results = tester.test()
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print('='*60)
    
    return results


if __name__ == '__main__':
    main()
