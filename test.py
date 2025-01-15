import os
import shutil
import torch
from tqdm import tqdm
import yaml
from ultralytics import YOLO
import cv2
import time
import matplotlib.pyplot as plt

def test(args):
    # Validating the 
    model = YOLO(args['weights'])
    res = model.val(data=args['data'], imgsz=args['imgsz'], name=args['name'])

def test_on_video(args):
    # Create exp_result directory if it doesn't exist
    os.makedirs('./exp_result', exist_ok=True)
    
    # Initialize metrics storage
    metrics = {
        'frame_times': [],
        'fps_values': [],
        'memory_usage': []
    }

    # Load model
    model = YOLO(args['weights'], verbose=False)
    
    video_list = ['test_data/ships_test.mp4', 'test_data/airplane_test.mp4']
    for file in video_list:
        cap = cv2.VideoCapture(file)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            results = model.predict(source=frame, imgsz=800, conf=0.5)
            end_time = time.time()
            
            # Calculate metrics
            frame_time = (end_time - start_time) * 1000  # ms
            fps = 1 / (end_time - start_time)
            mem_usage = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
            
            # Store metrics
            metrics['frame_times'].append(frame_time)
            metrics['fps_values'].append(fps)
            metrics['memory_usage'].append(mem_usage)
            
            annotated_frame = results[0].plot()
            cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Frame Time: {frame_time:.2f} ms', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f'Memory: {mem_usage:.2f} MB', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Inference', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        
        # Calculate averages for current video
        avg_frame_time = sum(metrics['frame_times']) / len(metrics['frame_times'])
        avg_fps = sum(metrics['fps_values']) / len(metrics['fps_values'])
        avg_memory = sum(metrics['memory_usage']) / len(metrics['memory_usage'])
        
        # Save metrics to file
        with open(f'./exp_result/{os.path.basename(file)}_metrics.txt', 'w') as f:
            f.write(f"Average Frame Time: {avg_frame_time:.2f} ms\n")
            f.write(f"Average FPS: {avg_fps:.2f}\n")
            f.write(f"Average Memory Usage: {avg_memory:.2f} MB\n")
        
        # Clear metrics for next video
        metrics = {key: [] for key in metrics}
    
    cv2.destroyAllWindows()

def test_on_images(args):
    # Create exp_result directory if it doesn't exist
    os.makedirs('./exp_result', exist_ok=True)
    
    # Load model
    model = YOLO(args['weights'], verbose=False)
    
    image_list = ['test_data/ship.jpg', 'test_data/airplane.jpg']
    for file in image_list:
        image = cv2.imread(file)
        results = model.predict(source=image, imgsz=800, conf=0.5)
        annotated_image = results[0].plot()
        
        cv2.imshow('Inference', annotated_image)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    args = yaml.load(open(r'./cfgs/test.yaml', 'r'), Loader=yaml.FullLoader)

    assert args['task'] in ['video', 'val', 'images'], "Invalid metrics argument. Choose from 'video' , 'val' or images"

    args['name'] = args['weights'].split('/')[2] + '_on_test'

    if args['task'] == 'video':
        test_on_video(args)
    elif args['task'] == 'val':
        test(args)
    elif args['task'] == 'images':
        test_on_images(args)     
