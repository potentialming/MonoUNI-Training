"""
Visualize depth map alignment with images, 2D boxes, and 3D boxes
"""
import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

# Assume visual_utils sits in 'scripts/data_converter'
# Add project root to sys.path
try:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from scripts.data_converter.visual_utils import (
        load_calib, 
        compute_box_3d_camera, 
        project_to_image,
        draw_box_3d
    )
except ImportError:
    print("Warning: Could not import 'visual_utils'. 3D box drawing might fail.")
    print("Please ensure this script is run from a directory where 'scripts/data_converter/visual_utils.py' is accessible.")
    # Define placeholder to avoid runtime failure
    def load_calib(calib_file):
        print("ERROR: load_calib is not loaded.")
        return None, np.eye(3, 4), np.zeros(3)
    def compute_box_3d_camera(dim, loc, ry, denorm):
        print("ERROR: compute_box_3d_camera is not loaded.")
        return np.zeros((8, 3))
    def project_to_image(box_3d, P2):
        print("ERROR: project_to_image is not loaded.")
        return np.zeros((8, 2))
    def draw_box_3d(image, box_2d, c=(255,255,255)):
        print("ERROR: draw_box_3d is not loaded.")
        return image


# Color map for unified object types
COLOR_MAP = {
    "car": (0, 255, 0),           # Green
    "big_vehicle": (0, 255, 255), # Cyan
    "pedestrian": (255, 255, 0),  # Yellow
    "cyclist": (0, 0, 255),       # Red
}


def load_label(label_file):
    """Load KITTI format label file"""
    boxes = []
    with open(label_file, 'r') as f:
        for line in f:
            fields = line.strip().split(' ')
            if len(fields) < 15:
                continue
            try:
                obj_type = fields[0]
                x1, y1, x2, y2 = float(fields[4]), float(fields[5]), float(fields[6]), float(fields[7])
                h, w, l = float(fields[8]), float(fields[9]), float(fields[10])
                x, y, z = float(fields[11]), float(fields[12]), float(fields[13])
                ry = float(fields[14])
                
                boxes.append({
                    'type': obj_type,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'dim': [h, w, l],
                    'location': [x, y, z],
                    'rotation_y': ry
                })
            except Exception:
                continue # skip malformed lines
    return boxes


def draw_3d_boxes_on_image(image, boxes, P2, denorm):
    """Draw 3D bounding boxes on image"""
    for box in boxes:
        obj_type = box['type']
        dim = np.array(box['dim'])
        location = np.array(box['location'])
        rotation_y = box['rotation_y']
        
        # Get color for this object type
        color = COLOR_MAP.get(obj_type, (255, 255, 255))
        
        # Compute 3D box corners in camera coordinates
        box_3d = compute_box_3d_camera(dim, location, rotation_y, denorm)
        
        # Project to image
        box_2d = project_to_image(box_3d, P2)
        
        # Skip if projection failed (NaN values)
        if np.any(np.isnan(box_2d)) or np.any(np.isinf(box_2d)):
            continue
        
        # Draw 3D box
        try:
            image = draw_box_3d(image, box_2d, c=color)
        except Exception as e:
            # Skip boxes that fail to draw
            continue
    
    return image


def visualize_sample(image_path, depth_path, label_path, calib_path, output_path):
    """Visualize a single sample with depth, 2D boxes, and 3D boxes"""
    
    # Load image
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return False
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return False
    
    # Load depth map
    if not os.path.exists(depth_path):
        print(f"Depth map not found: {depth_path}")
        return False
    depth_u16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_u16 is None:
        print(f"Failed to load depth map: {depth_path}")
        return False
    
    # Convert depth to float (divide by 256 to get actual depth in meters)
    depth = depth_u16.astype(np.float32) / 256.0
    
    # Load calibration
    if not os.path.exists(calib_path):
        print(f"Calibration not found: {calib_path}")
        return False
    try:
        K, P2, denorm = load_calib(calib_path)
    except Exception as e:
        print(f"Failed to load calib/denorm: {e}")
        return False
        
    # Load labels
    boxes = []
    if os.path.exists(label_path):
        boxes = load_label(label_path)
    
    # Create visualization
    h, w = image.shape[:2]
    
    # Normalize depth for visualization (0-255)
    depth_viz = np.zeros_like(depth)
    mask = depth > 0
    if mask.any():
        depth_min = depth[mask].min()
        depth_max = np.percentile(depth[mask], 99) # 99th percentile for better contrast
        if depth_max <= depth_min:
            depth_max = depth[mask].max()
            
        depth_viz[mask] = (depth[mask] - depth_min) / (depth_max - depth_min) * 255
        np.clip(depth_viz, 0, 255, out=depth_viz) # keep values in 0-255
        
    depth_viz = depth_viz.astype(np.uint8)
    
    # Apply colormap to depth
    depth_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
    depth_colored[~mask] = 0  # Set no-depth areas to black
    
    # Create overlay image
    overlay = cv2.addWeighted(image, 0.6, depth_colored, 0.4, 0)
    
    # Create copies for different visualizations
    image_with_3d = image.copy()
    depth_with_3d = depth_colored.copy()
    overlay_with_3d = overlay.copy()
    
    # Draw 3D boxes on all three images
    image_with_3d = draw_3d_boxes_on_image(image_with_3d, boxes, P2, denorm)
    depth_with_3d = draw_3d_boxes_on_image(depth_with_3d, boxes, P2, denorm)
    overlay_with_3d = draw_3d_boxes_on_image(overlay_with_3d, boxes, P2, denorm)
    
    # Draw 2D bounding boxes and labels on overlay only
    for box in boxes:
        x1, y1, x2, y2 = box['bbox']
        obj_type = box['type']
        color = COLOR_MAP.get(obj_type, (255, 255, 255))
        
        # Draw 2D box on overlay
        cv2.rectangle(overlay_with_3d, (x1, y1), (x2, y2), color, 1)
        
        # Add label
        label_text = obj_type
        cv2.putText(overlay_with_3d, label_text, (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Check if depth exists in box region and add depth info
        box_depth = depth[y1:y2, x1:x2]
        box_mask = box_depth > 0
        if box_mask.any():
            avg_depth = box_depth[box_mask].mean()
            depth_text = f"{avg_depth:.1f}m"
            cv2.putText(overlay_with_3d, depth_text, (x1, y2+15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Concatenate images horizontally
    result = np.hstack([image_with_3d, depth_with_3d, overlay_with_3d])
    
    # Add titles
    result_with_title = np.zeros((result.shape[0] + 30, result.shape[1], 3), dtype=np.uint8)
    result_with_title[30:, :] = result
    
    cv2.putText(result_with_title, "Image + 3D Boxes", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result_with_title, "Depth + 3D Boxes", (w + 10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(result_with_title, "Overlay + 3D Boxes + Labels", (2*w + 10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save result
    cv2.imwrite(output_path, result_with_title)
    print(f"Saved visualization to: {output_path}")
    
    # Print statistics
    print(f"\nStatistics for {os.path.basename(image_path)}:")
    print(f"  Image size: {w} x {h}")
    print(f"  Number of objects: {len(boxes)}")
    if mask.any():
        print(f"  Depth range: {depth[mask].min():.2f}m - {depth[mask].max():.2f}m")
        print(f"  Pixels with depth: {mask.sum()} / {depth.size} ({100*mask.sum()/depth.size:.2f}%)")
    else:
        print(f"  No depth data found!")
    
    # Check alignment for each box
    if len(boxes) > 0:
        print(f"\n  Box depth analysis:")
        for i, box in enumerate(boxes[:10]):  # Only show first 10 for brevity
            x1, y1, x2, y2 = box['bbox']
            box_depth = depth[y1:y2, x1:x2]
            box_mask = box_depth > 0
            if box_mask.any():
                avg_depth = box_depth[box_mask].mean()
                coverage = 100 * box_mask.sum() / max(1, box_mask.size) # avoid division by zero
                print(f"    {i+1}. {box['type']}: avg_depth={avg_depth:.2f}m, coverage={coverage:.1f}%")
            else:
                print(f"    {i+1}. {box['type']}: NO DEPTH DATA!")
        if len(boxes) > 10:
            print(f"    ... and {len(boxes) - 10} more objects")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize depth map alignment with images and 3D boxes")
    parser.add_argument("--dataset-root", type=str, required=True, help="Dataset root directory (KITTI format)")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset name for output folder (auto-detected if not specified)")
    
    # --- MODIFIED ARGUMENTS ---
    parser.add_argument("--sample-ids", type=str, nargs='+', default=None,
                        help="Specific sample IDs to visualize (e.g., 000000 000001). Overrides --num.")
    parser.add_argument("--num", type=int, default=50,
                        help="Number of samples to visualize (e.g., 20). "
                             "Used only if --sample-ids is NOT provided.")
    # --- END MODIFICATIONS ---
    
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for visualizations")
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    
    # Auto-detect dataset name from path if not specified
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        dataset_name = dataset_root.name
    
    # Find label directory (could be label_2 or label_2_4cls_for_train)
    label_dir = dataset_root / "training" / "label_2"
    if not label_dir.exists():
        label_dir = dataset_root / "training" / "label_2_4cls_for_train"
    
    image_dir = dataset_root / "training" / "image_2"
    depth_dir = dataset_root / "training" / "box3d_depth_dense"
    calib_dir = dataset_root / "training" / "calib"
    
    # Output to project directory with dataset-specific subfolder
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Assume project_root is defined correctly at top of script
        try:
             output_dir = project_root / "output" / "depth_visualization" / dataset_name
        except NameError:
             print("Warning: 'project_root' not defined. Saving to current directory.")
             output_dir = Path(".") / "output" / "depth_visualization" / dataset_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Visualizing depth alignment with 3D boxes for dataset: {dataset_name}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Check if required directories exist
    missing_dirs = []
    for name, path in [("image", image_dir), ("depth", depth_dir), ("label", label_dir), ("calib", calib_dir)]:
        if not path.exists():
            missing_dirs.append(f"{name} ({path})")
    
    if missing_dirs:
        print(f"Error: Missing required directories:")
        for dir_info in missing_dirs:
            print(f"  - {dir_info}")
        return
        
    # --- NEW LOGIC: Determine samples to process ---
    
    # 1. Get all available sample IDs from the label directory
    all_label_files = sorted(label_dir.glob("*.txt"))
    all_sample_ids = [f.stem for f in all_label_files]
    
    if not all_sample_ids:
        print(f"Error: No label files (*.txt) found in {label_dir}")
        return

    # 2. Determine which samples to process
    sample_list_to_process = []
    if args.sample_ids:
        # Prefer --sample-ids if provided
        sample_list_to_process = args.sample_ids
        print(f"Visualizing {len(sample_list_to_process)} specific samples from --sample-ids...")
    else:
        # Otherwise fall back to --num
        num_to_take = min(args.num, len(all_sample_ids))
        if args.num > len(all_sample_ids):
            print(f"Warning: Requested {args.num} samples, but only {len(all_sample_ids)} found.")
        
        sample_list_to_process = all_sample_ids[:num_to_take]
        print(f"Visualizing first {len(sample_list_to_process)} samples (from --num={args.num})...")

    if not sample_list_to_process:
        print("No samples selected to visualize.")
        return
    
    # --- END NEW LOGIC ---

    success_count = 0
    # 3. Loop over the determined list
    for sample_id in sample_list_to_process:
        print(f"\nProcessing sample: {sample_id}")
        print("-"*80)
        
        # Find image file
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = image_dir / f"{sample_id}{ext}"
            if candidate.exists():
                image_path = str(candidate)
                break
        
        if image_path is None:
            print(f"Image not found for {sample_id}")
            continue
        
        depth_path = str(depth_dir / f"{sample_id}.png")
        label_path = str(label_dir / f"{sample_id}.txt")
        calib_path = str(calib_dir / f"{sample_id}.txt")
        output_path = str(output_dir / f"{sample_id}_3d_alignment.jpg")
        
        if visualize_sample(image_path, depth_path, label_path, calib_path, output_path):
            success_count += 1
    
    print("\n" + "="*80)
    print(f"Successfully visualized {success_count}/{len(sample_list_to_process)} samples")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
