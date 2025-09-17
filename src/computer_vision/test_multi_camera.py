#!/usr/bin/env python3
"""
Multi-Camera Performance Testing Script
Tests YOLO model performance with multiple camera streams
"""

import os
import cv2
import time
import threading
import argparse
import psutil
import json
from datetime import datetime
from ultralytics import YOLO
import numpy as np

class MultiCameraTester:
    def __init__(self, model_path, duration=900, num_cameras=2):
        self.model_path = model_path
        self.duration = duration  # Test duration in seconds
        self.num_cameras = num_cameras
        self.model = None
        self.camera_threads = []
        self.results = []
        self.start_time = None
        self.running = False
        
    def load_model(self):
        """Load the YOLO model"""
        try:
            print(f"Loading model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def create_test_video(self, camera_id):
        """Create a test video stream for camera simulation"""
        # Create a simple test video with moving rectangles
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = f"test_camera_{camera_id}.avi"
        
        # Create video writer
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
        
        # Generate 100 frames of test video
        for i in range(100):
            # Create frame with moving rectangles (simulating vehicles)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add moving rectangles
            x = (i * 5) % 600
            y = 200 + int(50 * np.sin(i * 0.1))
            cv2.rectangle(frame, (x, y), (x + 40, y + 20), (0, 255, 0), -1)
            
            # Add some noise
            noise = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
            frame = cv2.add(frame, noise)
            
            out.write(frame)
        
        out.release()
        return video_path
    
    def process_camera_stream(self, camera_id, video_path):
        """Process a single camera stream"""
        print(f"🎥 Starting camera {camera_id} processing...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video for camera {camera_id}")
            return
        
        frame_count = 0
        total_vehicles = 0
        processing_times = []
        memory_usage = []
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Loop the video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
            
            # Run detection
            start_time = time.time()
            results = self.model(frame, classes=[2, 3, 5, 7])  # car, motorcycle, bus, truck
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Count vehicles
            vehicle_count = len(results[0].boxes) if results[0].boxes is not None else 0
            total_vehicles += vehicle_count
            
            # Monitor memory usage
            memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
            
            # Log every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - self.start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Camera {camera_id}: {frame_count} frames, {fps:.2f} FPS, {vehicle_count} vehicles")
            
            # Check if test duration is over
            if time.time() - self.start_time >= self.duration:
                break
        
        cap.release()
        
        # Calculate metrics
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        avg_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        max_memory = max(memory_usage) if memory_usage else 0
        avg_memory = np.mean(memory_usage) if memory_usage else 0
        
        result = {
            'camera_id': camera_id,
            'frames_processed': frame_count,
            'total_vehicles_detected': total_vehicles,
            'average_processing_time': avg_processing_time,
            'average_fps': avg_fps,
            'max_memory_mb': max_memory,
            'average_memory_mb': avg_memory,
            'test_duration': time.time() - self.start_time
        }
        
        self.results.append(result)
        print(f"✅ Camera {camera_id} completed: {frame_count} frames, {avg_fps:.2f} FPS")
    
    def run_test(self):
        """Run multi-camera performance test"""
        print(f"🚀 Starting Multi-Camera Performance Test")
        print(f"  Duration: {self.duration} seconds")
        print(f"  Cameras: {self.num_cameras}")
        print("=" * 50)
        
        if not self.load_model():
            return False
        
        # Create test videos
        video_paths = []
        for i in range(self.num_cameras):
            video_path = self.create_test_video(i)
            video_paths.append(video_path)
            print(f"✅ Created test video for camera {i}: {video_path}")
        
        # Start camera threads
        self.running = True
        self.start_time = time.time()
        
        for i, video_path in enumerate(video_paths):
            thread = threading.Thread(
                target=self.process_camera_stream,
                args=(i, video_path)
            )
            self.camera_threads.append(thread)
            thread.start()
        
        # Monitor system resources
        self.monitor_system_resources()
        
        # Wait for test completion
        for thread in self.camera_threads:
            thread.join()
        
        self.running = False
        
        # Clean up test videos
        for video_path in video_paths:
            if os.path.exists(video_path):
                os.remove(video_path)
        
        # Generate report
        self.generate_report()
        
        return True
    
    def monitor_system_resources(self):
        """Monitor system resources during test"""
        def monitor():
            while self.running:
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                if cpu_percent > 80 or memory_percent > 85:
                    print(f"⚠️  High resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%")
                
                time.sleep(5)
        
        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def generate_report(self):
        """Generate performance test report"""
        report_file = "docs/cv_multi_camera_report.md"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Multi-Camera Performance Test Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Test Configuration\n")
            f.write(f"- **Model:** {self.model_path}\n")
            f.write(f"- **Test Duration:** {self.duration} seconds\n")
            f.write(f"- **Number of Cameras:** {self.num_cameras}\n")
            f.write(f"- **Total Test Time:** {time.time() - self.start_time:.2f} seconds\n\n")
            
            f.write("## Performance Results\n")
            f.write("| Camera | Frames | Vehicles | Avg FPS | Avg Time (s) | Max Memory (MB) |\n")
            f.write("|--------|--------|----------|---------|--------------|-----------------|\n")
            
            total_frames = 0
            total_vehicles = 0
            total_fps = 0
            
            for result in self.results:
                f.write(f"| {result['camera_id']} | {result['frames_processed']} | {result['total_vehicles_detected']} | {result['average_fps']:.2f} | {result['average_processing_time']:.3f} | {result['max_memory_mb']:.1f} |\n")
                total_frames += result['frames_processed']
                total_vehicles += result['total_vehicles_detected']
                total_fps += result['average_fps']
            
            f.write(f"| **Total** | **{total_frames}** | **{total_vehicles}** | **{total_fps/len(self.results):.2f}** | - | - |\n\n")
            
            f.write("## Performance Analysis\n")
            avg_fps = total_fps / len(self.results) if self.results else 0
            if avg_fps >= 15:
                f.write("✅ **Overall Performance:** Excellent (≥15 FPS per camera)\n")
            elif avg_fps >= 10:
                f.write("⚠️  **Overall Performance:** Good (10-15 FPS per camera)\n")
            else:
                f.write("❌ **Overall Performance:** Needs improvement (<10 FPS per camera)\n")
            
            f.write(f"- **Average FPS per Camera:** {avg_fps:.2f}\n")
            f.write(f"- **Total Frames Processed:** {total_frames}\n")
            f.write(f"- **Total Vehicles Detected:** {total_vehicles}\n")
            f.write(f"- **Average Vehicles per Frame:** {total_vehicles/total_frames:.2f}\n\n")
            
            f.write("## Recommendations\n")
            if avg_fps < 15:
                f.write("- Consider model optimization (ONNX/TensorRT conversion)\n")
                f.write("- Use GPU acceleration if available\n")
                f.write("- Reduce input resolution if needed\n")
                f.write("- Consider reducing number of simultaneous cameras\n")
            
            f.write("- Monitor memory usage during extended operation\n")
            f.write("- Test with real camera feeds for validation\n")
            f.write("- Implement frame dropping if performance is critical\n\n")
        
        print(f"📄 Multi-camera report generated: {report_file}")
        
        # Save JSON results
        json_file = report_file.replace('.md', '_results.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_config': {
                    'model_path': self.model_path,
                    'duration': self.duration,
                    'num_cameras': self.num_cameras
                },
                'results': self.results,
                'summary': {
                    'total_frames': total_frames,
                    'total_vehicles': total_vehicles,
                    'average_fps': avg_fps
                }
            }, f, indent=2)
        
        print(f"📄 JSON results saved: {json_file}")

def main():
    parser = argparse.ArgumentParser(description='Test Multi-Camera Performance')
    parser.add_argument('--model', default='yolov8n.pt', help='Path to YOLO model file')
    parser.add_argument('--duration', type=int, default=900, help='Test duration in seconds (default: 900)')
    parser.add_argument('--cameras', type=int, default=2, help='Number of cameras to test (default: 2)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    # Run test
    tester = MultiCameraTester(args.model, args.duration, args.cameras)
    success = tester.run_test()
    
    if success:
        print(f"\n✅ Multi-camera test completed successfully!")
        print(f"  Duration: {args.duration}s")
        print(f"  Cameras: {args.cameras}")
        print(f"  Report: docs/cv_multi_camera_report.md")
    else:
        print(f"\n❌ Multi-camera test failed!")

if __name__ == "__main__":
    main()
