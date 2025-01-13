'''
Show the annotation of YOLO format
'''
import cv2
import os
import pickle
import numpy as np
import random

def plot_yolo_annotation(img_file, annot_file, class_dict=None):
    '''
    Plot the YOLO annotation on the image
    
    Args:
        img_path (str): Path to the image
        ann_path (str): Path to the annotation file
    '''
    # Reading the image and annot file
    image = cv2.imread(img_file)
    img_h, img_w, _ = image.shape
    
    with open(annot_file, 'r') as f:
        data = f.readlines()
        data = [i.split(' ') for i in data]
        data = [[float(j) for j in i] for i in data]
    
    # Calculating the bbox in Pascal VOC format
    for bbox in data:
        class_idx, x_center, y_center, width, height = bbox
        xmin = int((x_center - width / 2) * img_w)
        ymin = int((y_center - height / 2) * img_h)
        xmax = int((x_center + width / 2) * img_w)
        ymax = int((y_center + height / 2) * img_h)
        
        # Correcting bbox if out of image size
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > img_w - 1:
            xmax = img_w - 1
        if ymax > img_h - 1:
            ymax = img_h - 1
        
        # Creating the box and label for the image
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
        cv2.putText(image, str(class_idx), (xmin, 0 if ymin-10 < 0 else ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Display the image
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    root = r'C:/Users/yangc/Developer/data/dota_small_1024/train'
    ann_dir = os.path.join(root, 'labels')
    ann_files = [f for f in os.listdir(ann_dir) if os.path.getsize(os.path.join(ann_dir, f)) > 0]

    selected_files = random.sample(ann_files, 8)
    class_dict = pickle.load(open('./cfgs/dota_small/class_map.pkl', 'rb'))

    for ann_file in selected_files:
        ann_path = os.path.join(ann_dir, ann_file)
        img_path = ann_path.replace('labels', 'images').replace('.txt', '.png')
        plot_yolo_annotation(img_path, ann_path, class_dict)
