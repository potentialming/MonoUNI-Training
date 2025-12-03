#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert rope3D dataset to KITTI format with parallel processing (optimized).

Optimizations:
1) Image handling switches from (imread -> imwrite .png) to (shutil.copyfile),
   avoiding expensive PNG encoding (tradeoff: output images are .jpg instead of .png).
2) File-check logic in process_one_sample is streamlined.
"""

import os
import sys
import argparse
import json
import csv
import math
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------- Category mapping ----------------
# Unified mapping: lowercase; truck/bus -> big_vehicle
category_map = {
    'car': 'car', 'van': 'car',
    'truck': 'big_vehicle', 'bus': 'big_vehicle',
    'pedestrian': 'pedestrian',
    'cyclist': 'cyclist', 'motorcyclist': 'cyclist', 'tricyclist': 'cyclist'
}

# ---------------- Utility helpers ----------------
def parse_option():
    parser = argparse.ArgumentParser('Convert rope3D dataset to standard KITTI format', add_help=False)
    parser.add_argument('--source-root', type=str, default="data/rope3d", help='root path to rope3d dataset')
    parser.add_argument('--target-root', type=str, default="data/rope3d-kitti", help='root path to rope3d dataset in kitti format')
    parser.add_argument('--num-workers', type=int, default=None, help='number of parallel workers; default: os.cpu_count()')
    args = parser.parse_args()
    return args

def safe_copy(file_src, file_dest):
    # Ensure destination directory exists
    os.makedirs(os.path.dirname(file_dest), exist_ok=True)
    if not os.path.exists(file_dest):
        shutil.copyfile(file_src, file_dest)

def load_denorm(denorm_file):
    with open(denorm_file, 'r') as f:
        line = f.readline().strip()
    denorm = np.array([float(item) for item in line.split()], dtype=np.float64)
    return denorm

def get_cam2velo(denorm_file):
    denorm = load_denorm(denorm_file)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0],
                   [0.0, -1.0, 0.0]], dtype=np.float32)
    Rz = np.array([[0.0, 1.0, 0.0],
                   [-1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)

    origin_vector = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    target_vector = -1.0 * np.array([denorm[0], denorm[1], denorm[2]], dtype=np.float64)
    target_vector_norm = target_vector / (np.linalg.norm(target_vector) + 1e-12)
    sita = math.acos(np.clip(np.inner(target_vector_norm, origin_vector), -1.0, 1.0))
    n_vector = np.cross(target_vector_norm, origin_vector)
    n_norm = np.linalg.norm(n_vector)
    if n_norm < 1e-12:
        cam2velo = np.eye(3, dtype=np.float32)
    else:
        n_vector = (n_vector / n_norm).astype(np.float32)
        cam2velo, _ = cv2.Rodrigues(n_vector * sita)
        cam2velo = cam2velo.astype(np.float32)

    cam2velo = Rx @ cam2velo
    cam2velo = Rz @ cam2velo

    Ax, By, Cz, D = float(denorm[0]), float(denorm[1]), float(denorm[2]), float(denorm[3])
    mod_area = math.sqrt(Ax*Ax + By*By + Cz*Cz) + 1e-12
    d = abs(D) / mod_area

    Tr_cam2velo = np.eye(4, dtype=np.float32)
    Tr_cam2velo[:3, :3] = cam2velo
    Tr_cam2velo[:3, 3]  = [0.0, 0.0, d]
    Tr_velo2cam = np.linalg.inv(Tr_cam2velo)
    return Tr_velo2cam

def convert_calib(src_calib_file, src_denorm_file, dest_calib_file):
    with open(src_calib_file, 'r') as f:
        line0 = f.readline().strip()
    obj = line0.split()[1:]
    if len(obj) != 12:
        raise ValueError(f"P2 expects 12 values in {src_calib_file}")
    P2 = np.array(obj, dtype=np.float32)

    Tr_velo_to_cam = get_cam2velo(src_denorm_file)
    kitti_calib = dict()
    kitti_calib["P0"] = np.zeros((3, 4), dtype=np.float32)
    kitti_calib["P1"] = np.zeros((3, 4), dtype=np.float32)
    kitti_calib["P2"] = P2
    kitti_calib["P3"] = np.zeros((3, 4), dtype=np.float32)
    kitti_calib["R0_rect"] = np.identity(3, dtype=np.float32)
    kitti_calib["Tr_velo_to_cam"] = Tr_velo_to_cam[:3, :].astype(np.float32)
    kitti_calib["Tr_imu_to_velo"] = np.zeros((3, 4), dtype=np.float32)

    with open(dest_calib_file, "w") as calib_file:
        for (key, val) in kitti_calib.items():
            val = val.flatten()
            val_str = "%.12e" % val[0]
            for v in val[1:]:
                val_str += " %.12e" % v
            calib_file.write("%s: %s\n" % (key, val_str))

def alpha2roty(alpha, pos):
    ry = alpha + np.arctan2(pos[0], pos[2])
    if ry > np.pi:
        ry -= 2 * np.pi
    if ry < -np.pi:
        ry += 2 * np.pi
    return ry

def convert_label(src_label_file, dest_label_file):
    if not os.path.exists(src_label_file):
        # Allow empty labels: write empty file
        open(dest_label_file, 'w').close()
        return

    new_lines = []
    with open(src_label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls_type = parts[0]
            mapped = category_map.get(cls_type.lower(), None)
            if mapped is None:
                # Skip classes not in mapping
                continue
            parts[0] = mapped

            # Force truncated to zero
            try:
                truncated = int(float(parts[1]))
            except:
                truncated = 0
            parts[1] = '0.0'  # always set to 0.0

            # Normalize alpha/ry
            try:
                alpha = float(parts[3])
            except:
                alpha = 0.0
            try:
                pos = np.array([float(parts[11]), float(parts[12]), float(parts[13])], dtype=np.float32)
            except:
                continue
            if float(np.sum(np.abs(pos))) < 1e-9:
                continue
            try:
                ry = float(parts[14])
            except:
                ry = 0.0

            if alpha > np.pi:
                alpha -= 2 * np.pi
                ry = alpha2roty(alpha, pos)

            parts[3]  = str(alpha)
            parts[14] = str(ry)
            new_lines.append(' '.join(parts))

    with open(dest_label_file, 'w') as f:
        for ln in new_lines:
            f.write(ln + "\n")

# ---------------- Subprocess: handle one sample ----------------
def process_one_sample(task):
    """
    task: dict {
        'index': <src id>,
        'dst_id': <int id>,
        'src_image_path', 'src_label_path', 'src_calib_path', 'src_denorm_path',
        'dst_img', 'dst_label', 'dst_calib', 'dst_denorm'
    }
    """
    # Avoid OpenCV multithreading conflicts inside worker
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    idx = task['index']
    try:
        # Check source files
        si = task['src_image_path']
        sl = task['src_label_path']
        sc = task['src_calib_path']
        sd = task['src_denorm_path']
        
        # Label may be missing; other files must exist
        for p in (si, sc, sd):
            if not os.path.exists(p):
                return (idx, False, f"missing {p}")

        # === Optimization ===
        # 1. Image: direct file copy (much faster than decode/encode PNG)
        shutil.copyfile(si, task['dst_img'])
        # ============

        # 2. 标定
        convert_calib(sc, sd, task['dst_calib'])

        # 3. 标签
        convert_label(sl, task['dst_label'])

        # 4. denorm
        safe_copy(sd, task['dst_denorm'])

        return (idx, True, "")
    except Exception as e:
        return (idx, False, str(e))

# ---------------- Main pipeline (parallel) ----------------
def main_one_split(src_root, dest_root, split='train', start_img_id=0, num_workers=None):
    """
    Adapt to the dataset layout and process one split (train/val) in parallel.
    Returns (map_token2id, next_img_id)
    """
    os.makedirs(dest_root, exist_ok=True)
    out_img_dir   = os.path.join(dest_root, "training/image_2")
    out_lab_dir   = os.path.join(dest_root, "training/label_2")
    out_calib_dir = os.path.join(dest_root, "training/calib")
    out_den_dir   = os.path.join(dest_root, "training/denorm")
    for d in (out_img_dir, out_lab_dir, out_calib_dir, out_den_dir):
        os.makedirs(d, exist_ok=True)

    # Source directories
    src_image_dir = os.path.join(src_root, "image_2")
    src_label_dir = os.path.join(src_root, "label_2")
    src_calib_dir = os.path.join(src_root, "calib")
    src_den_dir   = os.path.join(src_root, "denorm")
    split_txt = os.path.join(src_root, "ImageSets", f"{split}.txt")
    if not os.path.exists(split_txt):
        print(f"[{split}] Split file not found: {split_txt}")
        return {}, start_img_id

    idx_list = [x.strip() for x in open(split_txt).readlines() if x.strip()]
    
    # Keep samples with existing images (JPG)
    valid_idx = []
    for idx in idx_list:
        if os.path.exists(os.path.join(src_image_dir, idx + ".jpg")):
            valid_idx.append(idx)
        else:
            # Try .png as fallback
            if os.path.exists(os.path.join(src_image_dir, idx + ".png")):
                 valid_idx.append(idx)
            else:
                 print(f"[{split}] Warning: image missing: {idx}.jpg/png")

    print(f"[{split}] Found {len(valid_idx)} valid samples")

    # Assign unique sequential target IDs
    tasks = []
    map_token2id = {}
    cur_id = start_img_id

    for idx in valid_idx:
        dst_id_str = f"{cur_id:06d}"
        
        # Determine source image extension
        src_img_path = os.path.join(src_image_dir, idx + ".jpg")
        if not os.path.exists(src_img_path):
            src_img_path = os.path.join(src_image_dir, idx + ".png")  # fallback to png

        tasks.append({
            'index': idx,
            'dst_id': cur_id,
            'src_image_path': src_img_path,
            'src_label_path': os.path.join(src_label_dir, idx + ".txt"),
            'src_calib_path': os.path.join(src_calib_dir, idx + ".txt"),
            'src_denorm_path': os.path.join(src_den_dir,   idx + ".txt"),
            
            # === Optimization ===
            # Force output extension to .jpg
            'dst_img':   os.path.join(out_img_dir,   dst_id_str + ".jpg"),
            # ============
            
            'dst_label': os.path.join(out_lab_dir,   dst_id_str + ".txt"),
            'dst_calib': os.path.join(out_calib_dir, dst_id_str + ".txt"),
            'dst_denorm':os.path.join(out_den_dir,   dst_id_str + ".txt"),
        })
        map_token2id[idx] = dst_id_str
        cur_id += 1

    if len(tasks) == 0:
        return {}, start_img_id

    # Run in parallel
    workers = num_workers if (num_workers and num_workers > 0) else os.cpu_count()
    if workers is None or workers < 1:
        workers = 1

    ok = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one_sample, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"[{split}] Converting", unit="file"):
            idx, success, msg = fut.result()
            if success:
                ok += 1
            else:
                print(f"[{split}] Warning: {idx} failed: {msg}")

    print(f"[{split}] Done: {ok}/{len(tasks)} succeeded.")
    return map_token2id, cur_id

def main():
    args = parse_option()
    source_root, target_root = args.source_root, args.target_root
    workers = args.num_workers

    print(f"Converting rope3D dataset from {source_root} to {target_root}")
    map_token2id = {}

    # Training split
    print("Processing training split (parallel)...")
    m_train, next_id = main_one_split(source_root, target_root, 'train', 0, workers)
    map_token2id.update(m_train)

    # Validation split
    print("Processing validation split (parallel)...")
    m_val, next_id = main_one_split(source_root, target_root, 'val', next_id, workers)
    map_token2id.update(m_val)

    # Save mapping
    os.makedirs(target_root, exist_ok=True)
    map_file = os.path.join(target_root, 'map_token2id.json')
    with open(map_file, 'w') as f:
        json.dump(map_token2id, f)
    print(f"Conversion completed! Total {len(map_token2id)} samples processed.")
    print(f"Token to ID mapping saved to {map_file}")
    
    # Generate ImageSets files
    imageset_dir = os.path.join(target_root, 'ImageSets')
    os.makedirs(imageset_dir, exist_ok=True)
    
    # Write train.txt
    train_ids = sorted([v for k, v in m_train.items()])
    with open(os.path.join(imageset_dir, 'train.txt'), 'w') as f:
        for tid in train_ids:
            f.write(tid + '\n')
    
    # Write val.txt
    val_ids = sorted([v for k, v in m_val.items()])
    with open(os.path.join(imageset_dir, 'val.txt'), 'w') as f:
        for vid in val_ids:
            f.write(vid + '\n')
    
    print(f"ImageSets created: train.txt ({len(train_ids)} samples), val.txt ({len(val_ids)} samples)")
    print(f"Token to ID mapping saved to {map_file}")
    
    # Generate ImageSets files
    imageset_dir = os.path.join(target_root, 'ImageSets')
    os.makedirs(imageset_dir, exist_ok=True)
    
    # Write train.txt
    train_ids = sorted([v for k, v in m_train.items()])
    with open(os.path.join(imageset_dir, 'train.txt'), 'w') as f:
        for tid in train_ids:
            f.write(tid + '\n')
    
    # Write val.txt
    val_ids = sorted([v for k, v in m_val.items()])
    with open(os.path.join(imageset_dir, 'val.txt'), 'w') as f:
        for vid in val_ids:
            f.write(vid + '\n')
    
    print(f"ImageSets created: train.txt ({len(train_ids)} samples), val.txt ({len(val_ids)} samples)")
    print(f"Token to ID mapping saved to {map_file}")

if __name__ == "__main__":
    # Limit OpenCV threads to avoid conflicts with multiprocessing
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    main()
