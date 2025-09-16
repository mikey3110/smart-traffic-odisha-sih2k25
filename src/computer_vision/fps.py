import cv2
import time
import argparse
import onnxruntime as ort
import numpy as np

def preprocess(frame, size=640):
    img = cv2.resize(frame, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return img

def benchmark(model_path, video_path):
    cap = cv2.VideoCapture(video_path)
    session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    fps_list = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        inp = preprocess(frame)
        start = time.time()
        session.run(None, {session.get_inputs()[0].name: inp})
        end = time.time()
        fps_list.append(1 / (end - start))
        frame_count += 1
        if frame_count >= 200:  # benchmark for 200 frames
            break

    avg_fps = sum(fps_list) / len(fps_list)
    print(f"✅ Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    args = parser.parse_args()
    benchmark(args.model, args.video)
