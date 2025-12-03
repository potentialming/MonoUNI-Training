import argparse
import os
import sys
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.data_converter.visual_utils import *

import warnings
warnings.filterwarnings("ignore")

def kitti_visual_tool(data_root, demo_dir):
    if not os.path.exists(data_root):
        raise ValueError("data_root Not Found")
    image_path = os.path.join(data_root, "training/image_2")
    calib_path = os.path.join(data_root, "training/calib")
    label_path = os.path.join(data_root, "training/label_2")
    image_ids = []
    for image_file in os.listdir(image_path):
        image_ids.append(image_file.split(".")[0])
    
    print(f"Found {len(image_ids)} images to visualize")
    
    for i in range(len(image_ids)):
        if os.path.exists(os.path.join(image_path, str(image_ids[i]) + ".png")):
            image_2_file = os.path.join(image_path, str(image_ids[i]) + ".png")
        elif os.path.exists(os.path.join(image_path, str(image_ids[i]) + ".jpg")):
            image_2_file = os.path.join(image_path, str(image_ids[i]) + ".jpg")
        else:
            print(f"Error: image file not found for {image_ids[i]}")
            continue
        
        calib_file = os.path.join(calib_path, str(image_ids[i]) + ".txt")
        label_2_file = os.path.join(label_path, str(image_ids[i]) + ".txt")
        
        image = cv2.imread(image_2_file)
        _, P2, denorm = load_calib(calib_file)
        image = draw_3d_box_on_image(image, label_2_file, P2, denorm)
        output_file = os.path.join(demo_dir, str(image_ids[i]) + ".jpg")
        cv2.imwrite(output_file, image)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_ids)} images")
    
    print(f"All done! Results saved to {demo_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset in KITTI format Checking ...")
    parser.add_argument("--data_root", type=str,
                        default="./datasets/dair-v2x-i-kitti",
                        help="Path to Dataset root in KITTI format")
    parser.add_argument("--demo_dir", type=str,
                        default="./demo_kitti_visual",
                        help="Path to demo directions")
    args = parser.parse_args()
    os.makedirs(args.demo_dir, exist_ok=True)
    kitti_visual_tool(args.data_root, args.demo_dir)
