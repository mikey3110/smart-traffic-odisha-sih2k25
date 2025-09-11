from ultralytics import YOLO
import cv2
import requests
import time
import json
from datetime import datetime

# Load YOLO model
model = YOLO('yolov8n.pt')

# API configuration
API_URL = "http://localhost:8000"

def send_vehicle_data(intersection_id, lane_counts):
    """Send vehicle count data to backend API"""
    try:
        data = {
            "intersection_id": intersection_id,
            "timestamp": int(time.time()),
            "lane_counts": lane_counts,
            "avg_speed": 25.5,  # Mock speed data
            "weather_condition": "clear"
        }
        
        response = requests.post(f"{API_URL}/traffic/ingest", json=data)
        if response.status_code == 200:
            print(f"✅ Data sent: {lane_counts}")
        else:
            print(f"❌ Failed to send data: {response.status_code}")
    except Exception as e:
        print(f"❌ Error sending data: {e}")

def detect_vehicles_in_video(video_path, intersection_id="junction-1"):
    """Detect vehicles in video and send data to API"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return
    
    print(f"🚗 Starting vehicle detection for {intersection_id}")
    print(f"📹 Video: {video_path}")
    print("Press 'q' to quit, 's' to send data manually")
    
    last_sent_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("📹 End of video reached")
            break
        
        frame_count += 1
        
        # Run YOLO detection
        results = model(frame, classes=[2, 3, 5, 7])  # car, motorcycle, bus, truck
        
        # Count vehicles in different regions (simulating lanes)
        height, width = frame.shape[:2]
        vehicle_count = len(results[0].boxes)
        
        # Simulate lane-based counting
        lane_counts = {
            "north_lane": max(0, vehicle_count - 2),
            "south_lane": max(0, vehicle_count - 1), 
            "east_lane": max(0, vehicle_count + 1),
            "west_lane": max(0, vehicle_count - 3)
        }
        
        # Draw detection results
        annotated_frame = results[0].plot()
        
        # Add text overlay
        cv2.putText(annotated_frame, f"Total Vehicles: {vehicle_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"North: {lane_counts['north_lane']}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"South: {lane_counts['south_lane']}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"East: {lane_counts['east_lane']}", (10, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"West: {lane_counts['west_lane']}", (10, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Send data every 5 seconds
        if time.time() - last_sent_time >= 5:
            send_vehicle_data(intersection_id, lane_counts)
            last_sent_time = time.time()
        
        # Display frame
        cv2.imshow('Smart Traffic - Vehicle Detection', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            send_vehicle_data(intersection_id, lane_counts)
            print("📤 Data sent manually")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("🏁 Vehicle detection completed")

def main():
    """Main function to run vehicle detection"""
    print("🚦 Smart Traffic - Computer Vision Module")
    print("=" * 50)
    
    # Try different video files
    video_files = [
        "data/videos/vehicles.mp4",
        "data/videos/traffic.mp4", 
        "data/videos/test video_1.mp4",
        "data/videos/Incredible Indian Traffic - isn't it crazy？!.mp4"
    ]
    
    video_path = None
    for vf in video_files:
        try:
            cap = cv2.VideoCapture(vf)
            if cap.isOpened():
                video_path = vf
                cap.release()
                break
        except:
            continue
    
    if not video_path:
        print("❌ No video files found. Please add videos to data/videos/ folder")
        print("📁 Available video files:")
        for vf in video_files:
            print(f"   - {vf}")
        return
    
    print(f"📹 Using video: {video_path}")
    
    # Start detection
    detect_vehicles_in_video(video_path, "junction-1")

if __name__ == "__main__":
    main()
