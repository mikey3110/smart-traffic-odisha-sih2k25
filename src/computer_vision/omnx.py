import cv2, time
from ultralytics import YOLO

model = YOLO("models/yolov8n.onnx")
cap = cv2.VideoCapture("../../data/videos/traffic.mp4")
frame_count, start = 0, time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    frame_count += 1
    if frame_count % 50 == 0:
        fps = frame_count / (time.time() - start)
        print(f"FPS: {fps:.2f}")
