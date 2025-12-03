import argparse
import os
import sys
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.data_converter.visual_utils import load_calib, compute_box_3d_camera, project_to_image, draw_box_3d

import warnings
warnings.filterwarnings("ignore")

# Color map for different object types
color_map = {
    "car": (0, 255, 0),           # Green
    "Car": (0, 255, 0),           
    "big_vehicle": (0, 255, 255), # Yellow
    "Bus": (0, 255, 255),         
    "pedestrian": (255, 255, 0),  # Cyan
    "Pedestrian": (255, 255, 0),  
    "cyclist": (0, 0, 255),       # Red
    "Cyclist": (0, 0, 255),       
}

def draw_3d_box_on_image_custom(image, label_2_file, P2, denorm, thickness=2):
    """Draw 3D bounding boxes on image from KITTI format label file."""
    if not os.path.exists(label_2_file):
        return image
        
    with open(label_2_file) as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            line_list = line.split(' ')
            object_type = line_list[0]
            
            # Skip if object type not in color map
            if object_type not in color_map:
                continue
            
            # Parse KITTI format: type truncated occluded alpha bbox(4) dimensions(3) location(3) rotation_y score
            try:
                dim = np.array(line_list[8:11]).astype(float)       # h, w, l
                location = np.array(line_list[11:14]).astype(float) # x, y, z
                rotation_y = float(line_list[14])
                
                # Compute 3D box corners in camera coordinates
                box_3d = compute_box_3d_camera(dim, location, rotation_y, denorm)
                
                # Project to 2D image
                box_2d = project_to_image(box_3d, P2)
                
                # Draw 3D box
                image = draw_box_3d(image, box_2d, c=color_map[object_type])
                
            except (ValueError, IndexError) as e:
                continue
    
    return image

def add_text_label(image, text, position='top'):
    """Add text label to image."""
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Set position
    if position == 'top':
        x = 10
        y = 40
    else:  # bottom
        x = 10
        y = h - 20
    
    # Draw background rectangle
    cv2.rectangle(image, (x - 5, y - text_height - 5), 
                  (x + text_width + 5, y + baseline + 5), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(image, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    return image

def visual_results(data_root, results_dir, output_dir, show_gt=False, max_images=None):
    """
    Visualize detection results on images.
    
    Args:
        data_root: Path to KITTI format dataset root
        results_dir: Path to detection results directory
        output_dir: Path to save visualized images
        show_gt: Whether to show ground truth in split screen (top: GT, bottom: predictions)
        max_images: Maximum number of images to visualize (None for all)
    """
    if not os.path.exists(data_root):
        raise ValueError(f"data_root not found: {data_root}")
    if not os.path.exists(results_dir):
        raise ValueError(f"results_dir not found: {results_dir}")
    
    image_path = os.path.join(data_root, "training/image_2")
    calib_path = os.path.join(data_root, "training/calib")
    gt_label_path = os.path.join(data_root, "training/label_2") if show_gt else None
    
    # Get all result files
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.txt')]
    result_files.sort()
    
    if max_images is not None:
        result_files = result_files[:max_images]
    
    print(f"Found {len(result_files)} result files to visualize")
    if show_gt:
        print("Visualization mode: Split screen (Top: Ground Truth, Bottom: Predictions)")
    else:
        print("Visualization mode: Predictions only")
    
    processed = 0
    for i, result_file in enumerate(result_files):
        image_id = result_file.replace('.txt', '')
        
        # Find image file
        image_2_file = None
        for ext in ['.png', '.jpg']:
            candidate = os.path.join(image_path, image_id + ext)
            if os.path.exists(candidate):
                image_2_file = candidate
                break
        
        if image_2_file is None:
            print(f"Warning: image file not found for {image_id}, skipping")
            continue
        
        calib_file = os.path.join(calib_path, image_id + ".txt")
        if not os.path.exists(calib_file):
            print(f"Warning: calib file not found for {image_id}, skipping")
            continue
        
        result_label_file = os.path.join(results_dir, result_file)
        
        # Load image and calibration
        image = cv2.imread(image_2_file)
        if image is None:
            print(f"Warning: failed to load image {image_2_file}, skipping")
            continue
            
        _, P2, denorm = load_calib(calib_file)
        
        if show_gt and gt_label_path:
            # Create split screen visualization
            gt_label_file = os.path.join(gt_label_path, result_file)
            
            # Draw GT on top image
            image_gt = image.copy()
            if os.path.exists(gt_label_file):
                image_gt = draw_3d_box_on_image_custom(image_gt, gt_label_file, P2, denorm, thickness=2)
            image_gt = add_text_label(image_gt, "Ground Truth", position='top')
            
            # Draw predictions on bottom image
            image_pred = image.copy()
            image_pred = draw_3d_box_on_image_custom(image_pred, result_label_file, P2, denorm, thickness=2)
            image_pred = add_text_label(image_pred, "Predictions", position='top')
            
            # Concatenate vertically
            final_image = np.vstack([image_gt, image_pred])
        else:
            # Only show predictions
            final_image = draw_3d_box_on_image_custom(image, result_label_file, P2, denorm, thickness=2)
        
        # Save output
        output_file = os.path.join(output_dir, image_id + ".jpg")
        cv2.imwrite(output_file, final_image)
        
        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed}/{len(result_files)} images")
    
    print(f"All done! Visualized {processed} images")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MonoUNI detection results")
    parser.add_argument("--data-root", type=str,
                        default="/home/liming/datasets/dair-v2x-i-kitti",
                        help="Path to KITTI format dataset root")
    parser.add_argument("--results-dir", type=str,
                        required=True,
                        help="Path to detection results directory")
    parser.add_argument("--output-dir", type=str,
                        default="./visual_results",
                        help="Path to save visualized images")
    parser.add_argument("--show-gt", action="store_true",
                        help="Show ground truth in split screen (top: GT, bottom: predictions)")
    parser.add_argument("--max-images", type=int,
                        default=None,
                        help="Maximum number of images to visualize (default: all)")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    visual_results(args.data_root, args.results_dir, args.output_dir, args.show_gt, args.max_images)
