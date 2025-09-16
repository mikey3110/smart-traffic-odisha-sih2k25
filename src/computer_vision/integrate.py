from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8n.onnx")
cap = cv2.VideoCapture(0)  # replace with RTSP for real feed

while True:
    ret, frame = cap.read()
    if not ret: break
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow("Demo", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"): break
cap.release()
cv2.destroyAllWindows()
