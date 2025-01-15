from ultralytics import YOLO
import yaml

def train(args):
    model = YOLO(args['weights'])
    results = model.train(
            data=args['data'], 
            epochs=args['epochs'], 
            batch=args['batch'], 
            imgsz=args['imgsz'], 
            name=args['name'],
            resume=args['resume'],
            workers=args['workers'],
    )


if __name__ == '__main__':
    cfg = yaml.load(open(r'./cfgs/train.yaml', 'r'), Loader=yaml.FullLoader)
    train(cfg)