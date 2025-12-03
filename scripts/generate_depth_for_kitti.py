"""
Generate depth maps for KITTI format dataset using denorm files.
This script creates dense depth maps based on 3D object annotations and ground plane equations.
This version is heavily optimized based on high-performance rasterization techniques.

Usage:
    python generate_depth_for_kitti.py --kitti-root /path/to/kitti/dataset [--thresh 0.5] [--workers 4]
"""

import os
import sys
import argparse
import cv2
import numpy as np
import math
import shutil
import logging
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from typing import Tuple, List, Sequence


# --- Setup and Data Structures (No changes needed here) ---

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
sys.dont_write_bytecode = True
# Color map after unified class mapping
COLOR_LIST = {
    "car": (0, 0, 255),           # blue
    "big_vehicle": (0, 255, 255), # cyan
    "pedestrian": (0, 128, 255),  # orange/blue mix
    "cyclist": (0, 128, 128)      # dark cyan
}

class ObjectData:
    def __init__( self, obj_type="unset", truncated=-1, occluded=-1, alpha=-10, x1=-1, y1=-1, x2=-1, y2=-1, h=-1, w=-1, l=-1, X=-1000, Y=-1000, Z=-1000, ry=-10, score=1.0 ):
        self.obj_type, self.truncated, self.occluded, self.alpha = obj_type.lower(), truncated, occluded, alpha
        self.x1, self.y1, self.x2, self.y2 = int(x1), int(y1), int(x2), int(y2)
        self.h, self.w, self.l = h, w, l
        self.X, self.Y, self.Z = X, Y, Z
        self.ry, self.score = ry, score
    def __str__(self): return "\n".join(f"{k}: {v}" for k, v in vars(self).items())

def load_kitti_labels(label_file):
    objects = []
    if not os.path.exists(label_file): return objects
    with open(label_file, 'r') as f:
        for line in f:
            fields = line.strip().split(' ')
            if len(fields) < 15: continue
            try:
                objects.append(ObjectData(
                    obj_type=fields[0], truncated=float(fields[1]), occluded=int(float(fields[2])),
                    alpha=float(fields[3]), x1=float(fields[4]), y1=float(fields[5]), x2=float(fields[6]),
                    y2=float(fields[7]), h=float(fields[8]), w=float(fields[9]), l=float(fields[10]),
                    X=float(fields[11]), Y=float(fields[12]), Z=float(fields[13]), ry=float(fields[14]),
                    score=float(fields[15]) if len(fields) > 15 else 1.0 ))
            except (ValueError, IndexError): continue
    return objects

def load_denorm_data(denorm_file):
    with open(denorm_file, 'r') as f:
        values = f.readline().strip().split()
        if len(values) != 4: raise ValueError(f"Invalid denorm format in {denorm_file}")
        return np.array([float(v) for v in values[:3]])

def load_kitti_calib(calib_file):
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith('P2:'):
                values = line.split()[1:]
                if len(values) != 12: raise ValueError(f"Invalid P2 matrix in {calib_file}")
                return np.array([float(v) for v in values]).reshape(3, 4)
    raise ValueError(f"P2 matrix not found in {calib_file}")

def compute_c2g_transformation(denorm):
    ground_z_axis = denorm / np.linalg.norm(denorm)
    cam_x_axis = np.array([1.0, 0.0, 0.0])
    ground_x_axis = cam_x_axis - np.dot(cam_x_axis, ground_z_axis) * ground_z_axis
    ground_x_axis /= np.linalg.norm(ground_x_axis)
    ground_y_axis = np.cross(ground_z_axis, ground_x_axis)
    ground_y_axis /= np.linalg.norm(ground_y_axis)
    return np.vstack([ground_x_axis, ground_y_axis, ground_z_axis])

def compute_plane_equation(p1, p2, p3):
    v1, v2 = p2 - p1, p3 - p1
    normal = np.cross(v1, v2)
    a, b, c = normal
    d = -np.dot(normal, p1)
    return a, b, c, d

def project_3d_to_2d(obj_center, w, h, l, ry, c2g_trans, p2):
    obj_center = np.array(obj_center).reshape(3, 1)
    obj_center_ground = (c2g_trans @ obj_center).flatten()
    theta_cam = np.array([math.cos(ry), 0, -math.sin(ry)]).reshape(3, 1)
    theta_ground = c2g_trans @ theta_cam
    yaw_ground = math.atan2(theta_ground[1, 0], theta_ground[0, 0])
    cos_yaw, sin_yaw = math.cos(yaw_ground), math.sin(yaw_ground)
    rot_matrix = np.array([[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]])
    
    # Ensure the object bottom-center projects to (X, Y, Z) on ground
    # (X, Y, Z) -> obj_center_ground -> (x_g, y_g, z_g)
    # Object center in ground coords: (x_g, y_g, z_g)
    # Want bottom center anchored at (x_g, y_g, z_g)
    # 8 corners in object coords (origin at bottom center)
    corners_obj = np.array([
        [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2], # x
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2], # y
        [0, 0, 0, 0, h, h, h, h]                   # z (0 = bottom, h = top)
    ])
    
    # Rotate corners then translate to ground coords (x_g, y_g, z_g)
    corners_ground = rot_matrix @ corners_obj + obj_center_ground.reshape(3, 1)

    g2c_trans = np.linalg.inv(c2g_trans)
    corners_cam = g2c_trans @ corners_ground
    proj_points = p2 @ np.vstack([corners_cam, np.ones((1, 8))])
    
    # Depth must be positive
    if np.any(proj_points[2] <= 0):
        return None, None
        
    corners_2d = (proj_points[:2] / proj_points[2]).T
    if np.any(np.isnan(corners_2d)) or np.any(np.isinf(corners_2d)): return None, None
    return corners_2d.astype(np.int32), corners_cam.T

# --- NEW High-Performance Rasterization Functions ---

def accumulate_face_depth(
    depth_map: np.ndarray,
    plane: Tuple[float, float, float, float],
    polygon: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> None:
    """
    Renders the depth of a single face onto the main depth map using the
    highly efficient bounding-box-cropping technique.
    """
    # 1. Bounding Box Cropping
    h, w = depth_map.shape
    # np.clip keeps the code concise
    x_min, y_min = np.clip(polygon.min(axis=0), 0, [w - 1, h - 1])
    x_max, y_max = np.clip(polygon.max(axis=0), 0, [w - 1, h - 1])

    if x_min > x_max or y_min > y_max:
        return

    # 2. Create a tiny, local mask
    local_w, local_h = x_max - x_min + 1, y_max - y_min + 1
    local_mask = np.zeros((local_h, local_w), dtype=np.uint8)
    
    # Shift polygon to local coordinates
    shifted_poly = polygon - np.array([x_min, y_min], dtype=np.int32)
    
    # 3. Use faster fillConvexPoly on the small mask
    cv2.fillConvexPoly(local_mask, shifted_poly, 1)
    
    # Get local coordinates of pixels within the face
    ys, xs = np.nonzero(local_mask)
    if xs.size == 0:
        return

    # Convert back to global image coordinates
    xs_world = xs + x_min
    ys_world = ys + y_min

    # 4. Vectorized depth calculation for these pixels
    a, b, c, d = plane
    denominator = a * ((xs_world - cx) / fx) + b * ((ys_world - cy) / fy) + c
    
    valid_mask = np.abs(denominator) > 1e-8
    if not np.any(valid_mask):
        return

    # Filter coordinates and calculate depth
    xs_world, ys_world = xs_world[valid_mask], ys_world[valid_mask]
    depth = -d / denominator[valid_mask]
    
    # Filter for positive depth
    positive_depth_mask = depth > 0
    if not np.any(positive_depth_mask):
        return

    xs_world = xs_world[positive_depth_mask]
    ys_world = ys_world[positive_depth_mask]
    depth = depth[positive_depth_mask]

    # 5. Direct, efficient update of the main depth map (Z-buffering)
    existing_depth = depth_map[ys_world, xs_world]
    # Update where new depth is smaller (closer) or where there was no depth before
    update_mask = (existing_depth == 0) | (depth < existing_depth)
    
    depth_map[ys_world[update_mask], xs_world[update_mask]] = depth[update_mask]

# --- MODIFIED process_single_image to use the new functions ---

def process_single_image(args: Tuple) -> bool:
    """
    Main processing function for one image. It now directly calls the
    high-performance rasterization logic.
    """
    sample_name, kitti_root, depth_dir, score_thresh, img_width, img_height = args
    
    try:
        kitti_root = Path(kitti_root)
        label_file = kitti_root / "training" / "label_2" / f"{sample_name}.txt"
        calib_file = kitti_root / "training" / "calib" / f"{sample_name}.txt"
        denorm_file = kitti_root / "training" / "denorm" / f"{sample_name}.txt"
        
        if not all(f.exists() for f in [calib_file, denorm_file]):
            logging.warning(f"Missing calib/denorm for {sample_name}")
            return False
        
        # Label file may be missing (no objects)
        objects = load_kitti_labels(str(label_file))
        denorm = load_denorm_data(str(denorm_file))
        p2 = load_kitti_calib(str(calib_file))
        c2g_trans = compute_c2g_transformation(denorm)
        
        depth_map = np.zeros((img_height, img_width), dtype=np.float32)

        for obj in objects:
            if obj.score < score_thresh or obj.obj_type not in COLOR_LIST or (obj.w <= 0.05 and obj.h <= 0.05 and obj.l <= 0.05):
                continue
            
            try:
                verts_2d, verts_3d = project_3d_to_2d(
                    [obj.X, obj.Y, obj.Z], obj.w, obj.h, obj.l, obj.ry, c2g_trans, p2
                )
                if verts_2d is None:
                    continue

                faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
                fx, fy, cx, cy = p2[0, 0], p2[1, 1], p2[0, 2], p2[1, 2]

                for face_indices in faces:
                    # Get 3D points for the plane and 2D points for the polygon
                    p1, p2_pt, p3 = verts_3d[face_indices[0]], verts_3d[face_indices[1]], verts_3d[face_indices[2]]
                    plane_eq = compute_plane_equation(p1, p2_pt, p3)
                    polygon_2d = verts_2d[face_indices]
                    
                    # Accumulate depth for this face directly onto the main depth map
                    accumulate_face_depth(depth_map, plane_eq, polygon_2d, fx, fy, cx, cy)

            except Exception:
                # Log or handle per-object errors if necessary, but continue processing
                continue

        depth_u16 = np.clip(depth_map * 256.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(Path(depth_dir) / f"{sample_name}.png"), depth_u16)
        return True
        
    except Exception as e:
        logging.error(f"FATAL error processing {sample_name}: {e}")
        return False

# --- Main execution block and argument parsing ---

def generate_depth_maps(kitti_root, score_thresh=0.5, num_workers=None, img_width=1920, img_height=1080):
    kitti_root, label_dir, depth_dir = Path(kitti_root), Path(kitti_root) / "training" / "label_2", Path(kitti_root) / "training" / "box3d_depth_dense"
    if not label_dir.exists(): raise FileNotFoundError(f"Label directory not found: {label_dir}")
    if depth_dir.exists(): shutil.rmtree(depth_dir)
    depth_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created depth directory: {depth_dir}")
    sample_names = [f.stem for f in sorted(label_dir.glob("*.txt"))]
    if not sample_names: raise ValueError(f"No label files found in {label_dir}")
    
    # Log using detected dimensions
    logging.info(f"Found {len(sample_names)} samples to process at {img_width}x{img_height}")
    
    if num_workers is None: num_workers = max(1, multiprocessing.cpu_count() - 1)
    logging.info(f"Using {num_workers} workers for parallel processing")
    args_list = [(name, str(kitti_root), str(depth_dir), score_thresh, img_width, img_height) for name in sample_names]
    success_count, failed_count = 0, 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image, args): args for args in args_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating depth maps"):
            try:
                if future.result(): success_count += 1
                else: failed_count += 1
            except Exception as e:
                logging.error(f"Future execution failed for {futures[future][0]}: {e}")
                failed_count += 1
    logging.info(f"Depth map generation completed. Successful: {success_count}, Failed: {failed_count}")
    return success_count, failed_count

def parse_args():
    parser = argparse.ArgumentParser(description="Generate depth maps for KITTI format dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--kitti-root", type=str, required=True, help="Root directory of KITTI format dataset")
    parser.add_argument("--thresh", type=float, default=0.5, help="Minimum score threshold for objects")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count - 1)")
    parser.add_argument("--img-width", type=int, default=1920, help="Image width in pixels (FALLBACK ONLY)")
    parser.add_argument("--img-height", type=int, default=1080, help="Image height in pixels (FALLBACK ONLY)")
    return parser.parse_args()

def main():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    args = parse_args()
    kitti_root = Path(args.kitti_root)
    
    # --- Directory validation (updated) ---
    training_dir = kitti_root / "training"
    label_dir = training_dir / "label_2"
    calib_dir = training_dir / "calib"
    denorm_dir = training_dir / "denorm"
    image_dir = training_dir / "image_2"  # newly added input check

    # Verify all required input directories
    required_dirs = {"label_2": label_dir, "calib": calib_dir, "denorm": denorm_dir, "image_2": image_dir}
    missing_dirs = [name for name, path in required_dirs.items() if not path.exists()]
    
    if missing_dirs:
        logging.error(f"Required directories not found in {training_dir}: {', '.join(missing_dirs)}")
        return 1

    # --- Auto-detect image size ---
    img_height, img_width = args.img_height, args.img_width  # 1) start with fallback values from args
    
    try:
        # 2) pick first label file to get a sample id
        first_label_file = next(sorted(label_dir.glob("*.txt")), None)
        if first_label_file:
            sample_stem = first_label_file.stem
            
            # 3) find matching image file (.png/.jpg/.jpeg)
            img_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate_path = image_dir / f"{sample_stem}{ext}"
                if candidate_path.exists():
                    img_path = candidate_path
                    break
            
            # 4) read its dimensions if found
            if img_path:
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_height, img_width = img.shape[:2]  # (height, width)
                    logging.info(f"Auto-detected image size from {img_path.name}: {img_width}x{img_height}")
                else:
                    logging.warning(f"Could not read sample image {img_path}. Using fallback size {img_width}x{img_height}.")
            else:
                logging.warning(f"No sample image found for {sample_stem}. Using fallback size {img_width}x{img_height}.")
        else:
            logging.warning(f"No label files found in {label_dir} to detect image size. Using fallback size {img_width}x{img_height}.")
    except Exception as e:
        logging.error(f"Error during image size detection: {e}. Using fallback size {img_width}x{img_height}.")
    # --- End of image size detection ---

    logging.info(f"Starting KITTI Depth Map Generation (High-Performance Version)")
    
    # 5) Pass detected (or fallback) size into main worker
    generate_depth_maps(
        kitti_root=args.kitti_root, 
        score_thresh=args.thresh, 
        num_workers=args.workers,
        img_width=img_width,  # use detected width
        img_height=img_height # use detected height
    )
    return 0

if __name__ == "__main__":
    exit(main())
