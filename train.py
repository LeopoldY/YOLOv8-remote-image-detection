from ultralytics import YOLO
import argparse

def train(args):
    model = YOLO(args.weights)
    results = model.train(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, name=args.name)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--epochs', type=int, default=50)
    args.add_argument('--batch', type=int, default=16)
    args.add_argument('--imgsz', type=int)
    args.add_argument('--data', type=str, default='config.yaml')
    args.add_argument('--weights', type=str, default='yolov8n.yaml', help='path to pretrained weights if training from pretrained')
    args.add_argument('--name', type=str)
    args = args.parse_args()

    train(args)