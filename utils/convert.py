import os
import cv2
from tqdm import tqdm
import pickle

DATA_PATH = r'C:/Users/yangc/Developer/data'

import os
from PIL import Image

def get_dota_class_map(ann_dir, save_path):
    '''
    Get the class map for DOTA dataset

    Args:
        ann_dir (str): Path to the directory containing DOTA annotations
        save_path (str): Path to the file to save the class
    Returns:
        dict: A dictionary mapping class names to class IDs
    '''
    class_map = {}
    class_id = 0

    if os.path.exists(save_path):
        print('[INFO] Loading class map...')
        with open(save_path, "rb") as f:
            class_map = pickle.load(f)
        print('[INFO] Class map loaded!')
        return class_map

    print('[INFO] Getting class map...')
    for file in tqdm(os.listdir(ann_dir)):
        with open(os.path.join(ann_dir, file), 'r') as f:
            for line in f:
                class_name = line.split(' ')[8]
                if class_name not in class_map:
                    class_map[class_name] = class_id
                    class_id += 1
    # save the class map to a file
    with open(save_path, 'wb') as f:
        pickle.dump(class_map, f)
    print('[INFO] Class map obtained!')
    return class_map
 
def convert_dota_to_yolo(ann_dir: str, out_dir: str, map_dir: str):
    '''
    Convert DOTA annotations to YOLO format

    Args:
        ann_dir (str): Path to the directory containing DOTA annotations
        out_dir (str): Path to the output directory
    '''
    if not os.path.exists(out_dir):
        print(f'[INFO] Creating output directory at {out_dir}')
        os.makedirs(out_dir)
    else:
        print(f'[INFO] Output directory already exists at {out_dir}')
    label_map = get_dota_class_map(ann_dir, map_dir)
    
    print('[INFO] Converting annotations to YOLO format...')
    for filename in tqdm(os.listdir(ann_dir)):
        dota_file = os.path.join(ann_dir, filename)
        yolo_file = os.path.join(ann_dir, filename.replace(".txt", ".txt")).replace("annfiles", "labels")
        img_file = os.path.join(ann_dir, filename.replace(".txt", ".png")).replace("annfiles", "images")
        
        with open(dota_file, "r") as dota_f:
            with open(yolo_file, "w") as yolo_f:
                for line in dota_f:
                    line = line.strip().split()
                    class_name = line[8]
                    x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line[:8])
                    
                    # 如果类别名称不在字典中，则添加新的索引
                    if class_name not in label_map:
                        raise ValueError(f"Unknown class name: {class_name}")
                    
                    width, height = get_image_size(img_file)

                    # 计算yolo格式的中心点坐标和宽高
                    x_center = (x1 + x3) / 2 / width
                    y_center = (y1 + y3) / 2 / height
                    width = (x3 - x1) / width
                    height = (y3 - y1) / height

                    # 将转换后的结果写入yolo格式文件
                    yolo_f.write(f"{label_map[class_name]} {x_center} {y_center} {width} {height}\n")

    print('[INFO] Annotations converted!')

def get_image_size(img_path):
    '''
    Get the size of the image

    Args:
        img_path (str): Path to the image

    Returns:
        tuple: A tuple containing the width and height of the image
    '''
    with Image.open(img_path) as img:
        return img.size

if __name__ == '__main__':
    mode = 'val'
    ann_dir = os.path.join(DATA_PATH, 'dota_small_1024', mode, 'annfiles')
    output_root = os.path.join(DATA_PATH, 'dota_small_1024', mode, 'labels')

    map_dir = r'./cfgs/dota_small/class_map.pkl'

    convert_dota_to_yolo(ann_dir, output_root, map_dir=map_dir)
