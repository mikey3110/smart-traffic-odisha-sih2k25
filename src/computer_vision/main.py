import cv2
import time
import requests
import numpy as np
import threading
from ultralytics import YOLO
from src.cv.utils import (
    load_optimized_model, 
    process_stream, 
    validate_model_accuracy
)
from config.api_config import BACKEND_API_URL

def main():
    """
    Main execution entry point for the computer vision module.
    Orchestrates model loading, multi-camera stream processing,
    and real-time data transmission to the backend.
    """
    
    # --- Part 1: Model Optimization & Loading ---
    print("Initializing CV module...")
    try:
        # Load the ONNX/TensorRT optimized model
        model = load_optimized_model("yolov8n-optimized.onnx")
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Optimized model not found. Please run the optimization script first.")
        return
    
    # --- Part 2: Multi-Camera Stream Handling ---
    rtsp_streams = [
        "rtsp://user:pass@camera1_ip/stream", 
        "rtsp://user:pass@camera2_ip/stream"
    ]
    camera_threads = []
    
    for i, stream_url in enumerate(rtsp_streams):
        thread = threading.Thread(
            target=process_stream, 
            args=(model, stream_url, i, BACKEND_API_URL)
        )
        camera_threads.append(thread)
        thread.start()

    print(f"Started processing {len(rtsp_streams)} camera feeds.")

    # --- Part 3: Accuracy Validation & Reporting (Executed on demand) ---
    # This section would be a separate, non-real-time script.
    # To run validation: python src/cv/validate.py
    
    # --- Part 4: Main Loop and API Integration ---
    try:
        while True:
            # Main thread can monitor the status of sub-threads or perform
            # other system-level checks.
            time.sleep(10)
    except KeyboardInterrupt:
        print("Shutting down CV module.")

if __name__ == "__main__":
    main()
