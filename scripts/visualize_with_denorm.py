#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize KITTI 3D boxes with a given ground plane (denorm) and render the ground grid.

- Reads denorm files from training/denorm/*.txt (4 numbers: a b c d)
- Uses P2 for projection
- Uses the plane normal to do vertical correction (same logic as your previous code)
- Draws a grid on the plane to help verify plane correctness

Usage:
  python scripts/visualize_with_denorm.py --kitti-root /home/liming/datasets/Rope3D_mini_kitti --demo-dir ./output/vis_denorm/rope3d_mini_kitti

Optional:
  --denorm-dir /custom/denorm/dir
  --max-vis 50
  --draw-plane 1
  --stride 10                 # draw every N-th image
  --x-range -20 20
  --z-range 2 80
  --grid-step 2  (meters)
"""

import os
import sys
import csv
import math
import argparse
from pathlib import Path

import cv2
import numpy as np

# -----------------------
# Helpers: calib & labels
# -----------------------

def load_calib_P2(calib_file: Path):
    """Load P2 (3x4) and return K (3x3) as well."""
    P2 = None
    with open(calib_file, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if len(row) == 0:
                continue
            if row[0] == 'P2:':
                vals = [float(v) for v in row[1:]]
                P2 = np.array(vals, dtype=np.float32).reshape(3, 4)
                break
    if P2 is None:
        raise ValueError(f"P2 not found in {calib_file}")
    K = P2[:3, :3]
    return K, P2


def read_kitti_labels(label_file: Path):
    """
    Read KITTI label_2. Return a list of dicts:
      {
        'type': str,               # Car, Pedestrian, Cyclist, Bus...
        'dim': np.array([h,w,l]),
        'loc': np.array([x,y,z]),
        'ry': float
      }
    """
    if not label_file.exists():
        return []

    objs = []
    with open(label_file, 'r') as f:
        for line in f.readlines():
            ss = line.strip().split()
            if len(ss) < 15:
                continue
            obj_type = ss[0]
            h, w, l = map(float, ss[8:11])
            x, y, z = map(float, ss[11:14])
            ry = float(ss[14])
            objs.append({
                'type': obj_type,
                'dim': np.array([h, w, l], dtype=np.float32),
                'loc': np.array([x, y, z], dtype=np.float32),
                'ry': float(ry)
            })
    return objs


def load_denorm(denorm_file: Path):
    """Load plane [a b c d] from training/denorm/<id>.txt"""
    with open(denorm_file, 'r') as f:
        vals = [float(v) for v in f.read().strip().split()]
    if len(vals) != 4:
        raise ValueError(f"Bad denorm file {denorm_file}: need 4 values, got {len(vals)}")
    return np.array(vals, dtype=np.float64)  # [a,b,c,d]


# -----------------------
# Geometry & projection
# -----------------------

def project_to_image(pts_3d: np.ndarray, P: np.ndarray):
    """pts_3d: (N,3) in rectified camera coords -> (N,2) pixels."""
    N = pts_3d.shape[0]
    pts_h = np.concatenate([pts_3d, np.ones((N, 1), dtype=np.float32)], axis=1)  # (N,4)
    uvw = (P @ pts_h.T).T  # (N,3)
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
    return uv


def compute_box_3d_camera(dim, location, rotation_y, denorm):
    """
    Same as your original logic:
    - Build 8 corners in object local (y down)
    - Rotate around camera Y by rotation_y
    - Use plane normal to do vertical correction via Rodrigues
    - Translate to camera coords at 'location'
    Return: (8,3) camera coords
    """
    # Rotation (yaw around Y)
    c, s = math.cos(rotation_y), math.sin(rotation_y)
    R_y = np.array([[ c, 0,  s],
                    [ 0, 1,  0],
                    [-s, 0,  c]], dtype=np.float32)

    # dim: [h, w, l]  -> use l,w,h order for corner construction
    h, w, l = float(dim[0]), float(dim[1]), float(dim[2])

    # 8 corners in object local: bottom y=0, top y=-h (KITTI y down)
    x_c = [ +l/2, +l/2, -l/2, -l/2, +l/2, +l/2, -l/2, -l/2 ]
    y_c = [ 0, 0, 0, 0, -h, -h, -h, -h ]
    z_c = [ +w/2, -w/2, -w/2, +w/2, +w/2, -w/2, -w/2, +w/2 ]
    corners = np.array([x_c, y_c, z_c], dtype=np.float32)

    # apply yaw
    corners_3d = (R_y @ corners)  # (3,8)

    # vertical correction using plane normal
    n = np.asarray(denorm[:3], dtype=np.float64)
    n_norm = n / (np.linalg.norm(n) + 1e-9)

    # desired up direction in KITTI rect camera coords is (0, -1, 0)
    up_desired = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    # angle & axis to rotate n_norm -> up_desired
    dotv = np.clip(float(np.dot(n_norm, up_desired)), -1.0, 1.0)
    theta = -math.acos(dotv)  # keep same sign convention as your code
    axis = np.cross(n_norm, up_desired)
    if np.linalg.norm(axis) > 1e-9:
        axis = axis / np.linalg.norm(axis)
        # Rodrigues
        rvec = (axis * theta).astype(np.float32)
        R_corr, _ = cv2.Rodrigues(rvec)
        corners_3d = (R_corr @ corners_3d)
    # else: normal already aligned with up_desired

    # translate to location
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.T  # (8,3)


# -----------------------
# Drawing
# -----------------------

# Unified color map for 4 object types
COLOR_MAP = {
    "car":         (0, 255,   0),  # Green
    "big_vehicle": (0, 255, 255),  # Cyan
    "pedestrian":  (255, 255,  0), # Yellow
    "cyclist":     (0,   0, 255),  # Red
}
DEFAULT_COLOR = (255, 128, 0)

def draw_box_3d(image, corners_2d, color=(0, 255, 0), thickness=2):
    face_idx = [[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    for ind_f in [3,2,1,0]:
        f = face_idx[ind_f]
        for j in range(4):
            p1 = (int(corners_2d[f[j], 0]), int(corners_2d[f[j], 1]))
            p2 = (int(corners_2d[f[(j+1) % 4], 0]), int(corners_2d[f[(j+1) % 4], 1]))
            cv2.line(image, p1, p2, color, thickness, lineType=cv2.LINE_AA)
        if ind_f == 0:
            # draw diagonals for the closest face
            pA = (int(corners_2d[f[0], 0]), int(corners_2d[f[0], 1]))
            pC = (int(corners_2d[f[2], 0]), int(corners_2d[f[2], 1]))
            pB = (int(corners_2d[f[1], 0]), int(corners_2d[f[1], 1]))
            pD = (int(corners_2d[f[3], 0]), int(corners_2d[f[3], 1]))
            cv2.line(image, pA, pC, color, 1, lineType=cv2.LINE_AA)
            cv2.line(image, pB, pD, color, 1, lineType=cv2.LINE_AA)
    return image


def draw_plane_grid(image, denorm, P2,
                    x_min=-20.0, x_max=20.0, z_min=2.0, z_max=80.0,
                    grid_step=2.0, line_thickness=1):
    """
    Draw grid lines lying on plane a x + b y + c z + d = 0 (rect camera coords).
    We sample either x or z and solve y = -(a x + c z + d)/b when |b|>eps.
    """
    a, b, c, d = [float(v) for v in denorm]
    eps = 1e-6
    H, W = image.shape[:2]

    # If b ~ 0, plane is near vertical; skip to avoid exploding y
    if abs(b) < eps:
        return image

    # z-direction grid lines (vary z, sweep x)
    zs = np.arange(z_min, z_max + 1e-6, grid_step)
    xs = np.arange(x_min, x_max + 1e-6, grid_step)

    # draw lines of constant z (sweep x)
    for z in zs:
        pts3 = []
        for x in [x_min, x_max]:
            y = -(a * x + c * z + d) / b
            pts3.append([x, y, z])
        pts3 = np.array(pts3, dtype=np.float32)  # (2,3)
        uv = project_to_image(pts3, P2)          # (2,2)
        p1 = (int(uv[0,0]), int(uv[0,1]))
        p2 = (int(uv[1,0]), int(uv[1,1]))
        if 0 <= p1[0] < W or 0 <= p1[1] < H or 0 <= p2[0] < W or 0 <= p2[1] < H:
            cv2.line(image, p1, p2, (80, 200, 80), line_thickness, cv2.LINE_AA)

    # draw lines of constant x (sweep z)
    for x in xs:
        pts3 = []
        for z in [z_min, z_max]:
            y = -(a * x + c * z + d) / b
            pts3.append([x, y, z])
        pts3 = np.array(pts3, dtype=np.float32)
        uv = project_to_image(pts3, P2)
        p1 = (int(uv[0,0]), int(uv[0,1]))
        p2 = (int(uv[1,0]), int(uv[1,1]))
        if 0 <= p1[0] < W or 0 <= p1[1] < H or 0 <= p2[0] < W or 0 <= p2[1] < H:
            cv2.line(image, p1, p2, (80, 200, 80), line_thickness, cv2.LINE_AA)

    # draw a short normal arrow at z = (z_min+5), x=0 for reference
    z0 = max(z_min + 5.0, z_min)
    x0 = 0.0
    y0 = -(a * x0 + c * z0 + d) / b
    n = np.array([a, b, c], dtype=np.float32)
    n = n / (np.linalg.norm(n) + 1e-9)
    # pick a small step (in meters) along normal
    P_start = np.array([[x0, y0, z0]], dtype=np.float32)
    P_end   = P_start + n * 1.0  # 1m arrow
    uv_s = project_to_image(P_start, P2)[0]
    uv_e = project_to_image(P_end,   P2)[0]
    cv2.arrowedLine(image, (int(uv_s[0]), int(uv_s[1])),
                    (int(uv_e[0]), int(uv_e[1])),
                    (50, 180, 250), 2, tipLength=0.2)

    return image


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser(description="Visualize KITTI with provided ground plane (denorm)")
    ap.add_argument("--kitti-root", type=str, required=True, help="KITTI root")
    ap.add_argument("--demo-dir",  type=str, required=True, help="Output image dir")
    ap.add_argument("--denorm-dir", type=str, default=None,
                    help="Directory of denorm files (default: kitti_root/training/denorm)")
    ap.add_argument("--max-vis", type=int, default=0, help="Max images to visualize (0 for all)")
    ap.add_argument("--stride", type=int, default=1, help="Visualize every N-th image")
    ap.add_argument("--draw-plane", type=int, default=1, help="Also draw plane grid (1/0)")

    # plane grid params
    ap.add_argument("--x-range", nargs=2, type=float, default=[-20.0, 20.0])
    ap.add_argument("--z-range", nargs=2, type=float, default=[2.0, 80.0])
    ap.add_argument("--grid-step", type=float, default=2.0)

    args = ap.parse_args()

    kit = Path(args.kitti_root)
    img_dir   = kit / "training" / "image_2"
    calib_dir = kit / "training" / "calib"
    label_dir = kit / "training" / "label_2"
    if args.denorm_dir is None:
        denorm_dir = kit / "training" / "denorm"
    else:
        denorm_dir = Path(args.denorm_dir)

    out_dir = Path(args.demo_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather image ids
    image_ids = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            image_ids.append(p.stem)

    if len(image_ids) == 0:
        print(f"No images in {img_dir}")
        return

    print(f"Found {len(image_ids)} images.")
    cnt = 0
    for idx, img_id in enumerate(image_ids):
        if idx % args.stride != 0:
            continue
        img_file = (img_dir / f"{img_id}.png")
        if not img_file.exists():
            img_file = (img_dir / f"{img_id}.jpg")
        if not img_file.exists():
            print(f"[Skip] image not found for {img_id}")
            continue

        calib_file  = calib_dir  / f"{img_id}.txt"
        label_file  = label_dir  / f"{img_id}.txt"
        denorm_file = denorm_dir / f"{img_id}.txt"

        if not calib_file.exists() or not denorm_file.exists():
            print(f"[Skip] missing calib/denorm for {img_id}")
            continue

        # load
        image = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        _, P2 = load_calib_P2(calib_file)
        denorm = load_denorm(denorm_file)
        objects = read_kitti_labels(label_file)

        # draw plane grid (optional)
        if args.draw_plane:
            image = draw_plane_grid(
                image, denorm, P2,
                x_min=args.x_range[0], x_max=args.x_range[1],
                z_min=args.z_range[0], z_max=args.z_range[1],
                grid_step=args.grid_step, line_thickness=1
            )

        # draw 3D boxes with vertical correction using denorm
        for obj in objects:
            if obj['type'] not in COLOR_MAP:
                # skip other types to keep view clean; or use DEFAULT_COLOR
                # color = DEFAULT_COLOR
                continue
            corners_3d = compute_box_3d_camera(
                obj['dim'], obj['loc'], obj['ry'], denorm
            )  # (8,3) in rect camera coords
            uv = project_to_image(corners_3d, P2)  # (8,2)
            image = draw_box_3d(image, uv, color=COLOR_MAP[obj['type']], thickness=2)

        # write
        out_path = out_dir / f"{img_id}.jpg"
        cv2.imwrite(str(out_path), image)
        cnt += 1

        if args.max_vis > 0 and cnt >= args.max_vis:
            break

    print(f"Done. Saved {cnt} images to {out_dir}")


if __name__ == "__main__":
    main()
