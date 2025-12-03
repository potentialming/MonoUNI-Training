"""
Generate denorm (ground plane equation) files for KITTI format dataset.
The denorm file contains 4 values (a, b, c, d) representing the ground plane equation: ax + by + cz + d = 0
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def equation_plane(points):
    """
    Calculate plane equation from 3 points.
    Plane equation: ax + by + cz + d = 0
    """
    x1, y1, z1 = points[0, 0], points[0, 1], points[0, 2]
    x2, y2, z2 = points[1, 0], points[1, 1], points[1, 2]
    x3, y3, z3 = points[2, 0], points[2, 1], points[2, 2]
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return np.array([a, b, c, d])


def get_denorm_from_calib(calib_file):
    """
    Extract denorm (ground plane equation) from KITTI calib file.
    Uses Tr_velo_to_cam transformation matrix.
    """
    # Read calibration file
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    
    # Parse Tr_velo_to_cam
    tr_velo_to_cam = None
    for line in lines:
        if line.startswith('Tr_velo_to_cam:'):
            values = line.strip().split()[1:]
            tr_velo_to_cam = np.array([float(v) for v in values]).reshape(3, 4)
            break
    
    if tr_velo_to_cam is None:
        raise ValueError(f"Tr_velo_to_cam not found in {calib_file}")
    
    # Build lidar2cam transformation matrix (4x4)
    lidar2cam = np.eye(4)
    lidar2cam[:3, :3] = tr_velo_to_cam[:3, :3]  # Rotation
    lidar2cam[:3, 3] = tr_velo_to_cam[:3, 3]    # Translation
    
    # Define ground plane points in LiDAR coordinate (z=0 plane)
    ground_points_lidar = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ])
    
    # Convert to homogeneous coordinates
    ground_points_lidar = np.concatenate(
        (ground_points_lidar, np.ones((ground_points_lidar.shape[0], 1))), 
        axis=1
    )
    
    # Transform ground points to camera coordinate
    ground_points_cam = np.matmul(lidar2cam, ground_points_lidar.T).T
    
    # Calculate plane equation and negate (following the original implementation)
    denorm = -1 * equation_plane(ground_points_cam)
    
    return denorm


def generate_denorm_files(kitti_root):
    """
    Generate denorm files for all samples in KITTI format dataset.
    
    Args:
        kitti_root: Root directory of KITTI format dataset
    """
    kitti_root = Path(kitti_root)
    
    # Define paths
    calib_dir = kitti_root / "training" / "calib"
    denorm_dir = kitti_root / "training" / "denorm"
    
    # Check if calib directory exists
    if not calib_dir.exists():
        print(f"Error: Calibration directory not found: {calib_dir}")
        return
    
    # Create denorm directory
    denorm_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all calibration files
    calib_files = sorted(calib_dir.glob("*.txt"))
    
    if len(calib_files) == 0:
        print(f"Error: No calibration files found in {calib_dir}")
        return
    
    print(f"Found {len(calib_files)} calibration files")
    print(f"Generating denorm files to {denorm_dir}")
    
    # Process each calibration file
    success_count = 0
    for calib_file in tqdm(calib_files, desc="Generating denorm files"):
        try:
            # Get denorm from calibration
            denorm = get_denorm_from_calib(calib_file)
            
            # Save denorm file with the same name
            denorm_file = denorm_dir / calib_file.name
            
            # Write denorm to file (4 values in one line, space-separated)
            with open(denorm_file, 'w') as f:
                f.write(' '.join([f'{v:.10f}' for v in denorm]))
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing {calib_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Successfully generated {success_count}/{len(calib_files)} denorm files")
    print(f"Denorm files saved to: {denorm_dir}")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate denorm (ground plane equation) files for KITTI format dataset"
    )
    parser.add_argument(
        "--kitti-root",
        type=str,
        required=True,
        help="Root directory of KITTI format dataset"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_denorm_files(args.kitti_root)
