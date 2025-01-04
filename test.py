import os
import shutil
import torch
from tqdm import tqdm
from ultralytics import YOLO
import cv2
import time

def test(model):
    # Validating the 
    res = model.val(data='test_config.yaml', imgsz=800, name='yolov8n_val_on_test')

def test_on_video(model):
    # Predicting on video files that the model has not seen.
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
            fps = 1 / (end_time - start_time)
            annotated_frame = results[0].plot()
            # Display FPS on the frame
            cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Display frame time on the frame
            frame_time = (end_time - start_time) * 1000
            cv2.putText(annotated_frame, f'Frame Time: {frame_time:.2f} ms', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Display memory usage on the frame
            mem_usage = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
            cv2.putText(annotated_frame, f'Memory: {mem_usage:.2f} MB', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Inference', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model = YOLO('runs/detect/my_yolov8n_epochs50_batch16/weights/best.pt')
    test(model)