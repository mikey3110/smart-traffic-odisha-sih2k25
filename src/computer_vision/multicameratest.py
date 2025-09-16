import cv2
from ultralytics import YOLO
import threading

model = YOLO("models/yolov8n.onnx")

def run_camera(rtsp_url, cam_id):
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        cv2.imshow(f"Camera {cam_id}", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

threads = [
    threading.Thread(target=run_camera, args=("rtsp://camera1", 1)),
    threading.Thread(target=run_camera, args=("rtsp://camera2", 2))
]

for t in threads: t.start()
for t in threads: t.join()
