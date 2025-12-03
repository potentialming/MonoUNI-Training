import os
import numpy as np
from .utils import mkdir_p, read_json, get_files_path


def write_kitti_in_txt(my_json, path_txt):
    wf = open(path_txt, "w")
    for item in my_json:
        i1 = str(item["type"]).title()
        i2 = str(item["truncated_state"])
        i3 = str(item["occluded_state"])
        i4 = str(item["alpha"])
        i5, i6, i7, i8 = (
            str(item["2d_box"]["xmin"]),
            str(item["2d_box"]["ymin"]),
            str(item["2d_box"]["xmax"]),
            str(item["2d_box"]["ymax"]),
        )
        i9, i11, i10 = str(item["3d_dimensions"]["h"]), str(item["3d_dimensions"]["l"]), str(item["3d_dimensions"]["w"])
        i12, i13, i14 = str(item["3d_location"]["x"]), str(item["3d_location"]["y"]), str(item["3d_location"]["z"])
        i15 = str(-0.5 * np.pi  - float(item["rotation"]))
        item_list = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15]
        item_string = " ".join(item_list) + "\n"
        wf.write(item_string)
    wf.close()


def json2kitti(json_root, kitti_label_root):
    mkdir_p(kitti_label_root)
    jsons_path = get_files_path(json_root, ".json")
    for json_path in jsons_path:
        my_json = read_json(json_path)
        name = json_path.split("/")[-1][:-5] + ".txt"
        path_txt = os.path.join(kitti_label_root, name)
        write_kitti_in_txt(my_json, path_txt)


def rewrite_txt(path):
    """
    统一映射规则(与rope3d-kitti保持一致):
    - Car/Van -> car
    - Truck/Bus -> big_vehicle  
    - Pedestrian -> pedestrian
    - Cyclist/Tricyclist/Motorcyclist/Barrowlist -> cyclist
    - Trafficcone -> trafficcone (小写，保留但不训练)
    - 过滤掉: Unknown等
    """
    with open(path, "r+") as f:
        data = f.readlines()
        new_data = ""
        for line in data:
            # 过滤掉不需要的类别
            if line.startswith("Unknown "):
                continue
            # 保持原始类型名,只做标准化处理
            if line.startswith("Car "):
                line = line.replace("Car", "car", 1)
            elif line.startswith("Van "):
                line = line.replace("Van", "car", 1)
            elif line.startswith("Truck "):
                line = line.replace("Truck", "big_vehicle", 1)
            elif line.startswith("Bus "):
                line = line.replace("Bus", "big_vehicle", 1)
            elif line.startswith("Pedestrian "):
                line = line.replace("Pedestrian", "pedestrian", 1)
            elif line.startswith("Cyclist "):
                line = line.replace("Cyclist", "cyclist", 1)
            elif line.startswith("Tricyclist "):
                line = line.replace("Tricyclist", "cyclist", 1)
            elif line.startswith("Motorcyclist "):
                line = line.replace("Motorcyclist", "cyclist", 1)
            elif line.startswith("Barrowlist "):
                line = line.replace("Barrowlist", "cyclist", 1)
            elif line.startswith("Trafficcone "):
                line = line.replace("Trafficcone", "trafficcone", 1)
            new_data = new_data + line
    os.remove(path)
    f_new = open(path, "w")
    f_new.write(new_data)
    f_new.close()


def rewrite_label(path_file):
    path_list = get_files_path(path_file, ".txt")
    for path in path_list:
        rewrite_txt(path)


def label_filter(label_dir):
    label_dir = label_dir
    files = os.listdir(label_dir)

    for file in files:
        path = os.path.join(label_dir, file)

        lines_write = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                wlh = float(line.split(" ")[9])
                if wlh > 0:
                    lines_write.append(line)

        with open(path, "w") as f:
            f.writelines(lines_write)
