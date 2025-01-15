# Yolov8-Object-Detection-Satellite-Imagery
This repository contains the code for object detection on satellite imagery using YOLOv8. The code is written in Python and uses PyTorch. The code is based on the YOLOv8 implementation by Ultralytics

## Installation
All the requirements are basically contained in the ultralytics repository. To install the requirements, run the following command:
```bash 
pip install Ultralytics
```

## Usage
1. Prepare the dataset: The dataset should be organized in the following format:
```
datasets
# ├── train
  |   └── images
  |   └── labels  
# ├── val
      └── images
      └── labels
```
2. Use the img_split tool in `utils` to split the images and labels into train and val folders.
- Run the following command to split the images and labels by multi-scale:
```bash
python split_Img.py --base-json /utils/split/split_configs/ms_train.json
```
- or run the following command to split the images and labels by single-scale:
```bash
python split_Img.py --base-json /utils/split/split_configs/ss_train.json
```
3. Run the convert.py script to convert the labels to YOLO format. The script is in the `utils` folder. Remember to change the convert mode to `train` or `val` depending on the dataset you are converting.
```bash
python convert.py --mode train --data_root /path/to/dataset
```
4. Train the model using the following command:
```bash
python train.py 
```
You should change the parameters in the `/cfgs/train.yaml` file to suit your dataset and training requirements.
5. To test the model, run the following command:
```bash
python test.py
```
You should change the parameters in the `/cfgs/test.yaml` file to suit your dataset and testing requirements.

## Improvements
I am currently working on improving the code to make it more precise and faster. 
1. I adopted the Contextual Transformer architecture to improve the performance of the model. The Contextual Transformer is a transformer-based architecture that uses the context of the image to improve the performance of the model. The Contextual Transformer is implemented in the `ultralytics/nn/Attention/` folder.


