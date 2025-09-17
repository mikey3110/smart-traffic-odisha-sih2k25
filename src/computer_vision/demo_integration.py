"""
Computer Vision Demo Integration
Integrates CV detection with frontend dashboard and HLS streaming
"""

import cv2
import time
import json
import threading
import requests
from datetime import datetime
from typing import Dict, List, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CVDemoIntegration:
    def __init__(self, backend_url: str = "http://localhost:8000", 
                 hls_url: str = "http://localhost:5001"):
        self.backend_url = backend_url
        self.hls_url = hls_url
        self.cameras = {}
        self.detection_threads = {}
        self.running = False
        
    def add_camera(self, camera_id: str, rtsp_url: str, intersection_id: str,
                   coordinates: Dict[str, float], model_path: str = "yolov8n.pt"):
        """Add a camera for detection and streaming"""
        self.cameras[camera_id] = {
            'camera_id': camera_id,
            'rtsp_url': rtsp_url,
            'intersection_id': intersection_id,
            'coordinates': coordinates,
            'model_path': model_path,
            'status': 'inactive'
        }
        logger.info(f"Added camera {camera_id} for intersection {intersection_id}")
    
    def start_detection(self, camera_id: str):
        """Start vehicle detection for a camera"""
        if camera_id not in self.cameras:
            logger.error(f"Camera {camera_id} not found")
            return False
        
        if camera_id in self.detection_threads:
            logger.warning(f"Detection already running for camera {camera_id}")
            return False
        
        # Start detection thread
        thread = threading.Thread(
            target=self._detection_loop,
            args=(camera_id,)
        )
        thread.daemon = True
        thread.start()
        
        self.detection_threads[camera_id] = thread
        self.cameras[camera_id]['status'] = 'active'
        
        logger.info(f"Started detection for camera {camera_id}")
        return True
    
    def stop_detection(self, camera_id: str):
        """Stop vehicle detection for a camera"""
        if camera_id in self.detection_threads:
            self.cameras[camera_id]['status'] = 'stopping'
            # Thread will stop on next iteration
            logger.info(f"Stopping detection for camera {camera_id}")
            return True
        return False
    
    def _detection_loop(self, camera_id: str):
        """Main detection loop for a camera"""
        camera_info = self.cameras[camera_id]
        
        # Load YOLO model
        try:
            from ultralytics import YOLO
            model = YOLO(camera_info['model_path'])
            logger.info(f"Loaded YOLO model for camera {camera_id}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model for camera {camera_id}: {e}")
            return
        
        # Open video stream
        cap = cv2.VideoCapture(camera_info['rtsp_url'])
        if not cap.isOpened():
            logger.error(f"Failed to open video stream for camera {camera_id}")
            return
        
        frame_count = 0
        last_count_time = time.time()
        vehicle_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
        
        while self.cameras[camera_id]['status'] == 'active':
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame from camera {camera_id}, retrying...")
                time.sleep(1)
                continue
            
            frame_count += 1
            
            # Run detection every 10 frames (reduce processing load)
            if frame_count % 10 == 0:
                try:
                    # Run YOLO detection
                    results = model(frame, classes=[2, 3, 5, 7], verbose=False)  # car, motorcycle, bus, truck
                    
                    # Count vehicles
                    current_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
                    if results[0].boxes is not None:
                        for box in results[0].boxes:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            if class_name in current_counts:
                                current_counts[class_name] += 1
                    
                    # Update running totals
                    for vehicle_type in current_counts:
                        vehicle_counts[vehicle_type] = current_counts[vehicle_type]
                    
                    # Send counts to backend every 30 seconds
                    if time.time() - last_count_time >= 30:
                        self._send_vehicle_counts(camera_id, vehicle_counts)
                        last_count_time = time.time()
                        logger.info(f"Camera {camera_id}: {sum(vehicle_counts.values())} vehicles detected")
                
                except Exception as e:
                    logger.error(f"Detection error for camera {camera_id}: {e}")
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.1)
        
        cap.release()
        logger.info(f"Detection stopped for camera {camera_id}")
    
    def _send_vehicle_counts(self, camera_id: str, counts: Dict[str, int]):
        """Send vehicle counts to backend API"""
        try:
            payload = {
                "camera_id": camera_id,
                "intersection_id": self.cameras[camera_id]['intersection_id'],
                "timestamp": int(time.time()),
                "total_vehicles": sum(counts.values()),
                "counts_by_class": counts,
                "coordinates": self.cameras[camera_id]['coordinates']
            }
            
            response = requests.post(
                f"{self.backend_url}/cv/counts",
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                logger.debug(f"Successfully sent counts for camera {camera_id}")
            else:
                logger.warning(f"Failed to send counts for camera {camera_id}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending counts for camera {camera_id}: {e}")
    
    def start_hls_streaming(self, camera_id: str):
        """Start HLS streaming for a camera"""
        try:
            camera_info = self.cameras[camera_id]
            payload = {
                'camera_id': camera_id,
                'rtsp_url': camera_info['rtsp_url'],
                'intersection_id': camera_info['intersection_id'],
                'coordinates': camera_info['coordinates']
            }
            
            response = requests.post(
                f"{self.hls_url}/cv/streams/{camera_id}/start",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Started HLS streaming for camera {camera_id}")
                return True
            else:
                logger.error(f"Failed to start HLS streaming for camera {camera_id}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting HLS streaming for camera {camera_id}: {e}")
            return False
    
    def stop_hls_streaming(self, camera_id: str):
        """Stop HLS streaming for a camera"""
        try:
            response = requests.post(
                f"{self.hls_url}/cv/streams/{camera_id}/stop",
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Stopped HLS streaming for camera {camera_id}")
                return True
            else:
                logger.error(f"Failed to stop HLS streaming for camera {camera_id}: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping HLS streaming for camera {camera_id}: {e}")
            return False
    
    def get_stream_urls(self) -> List[Dict]:
        """Get all available stream URLs for frontend"""
        try:
            response = requests.get(f"{self.hls_url}/cv/streams", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get('streams', [])
            else:
                logger.error(f"Failed to get stream URLs: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting stream URLs: {e}")
            return []
    
    def start_all_cameras(self):
        """Start detection and streaming for all cameras"""
        for camera_id in self.cameras:
            self.start_detection(camera_id)
            self.start_hls_streaming(camera_id)
            time.sleep(2)  # Stagger starts
        
        self.running = True
        logger.info("Started all cameras")
    
    def stop_all_cameras(self):
        """Stop detection and streaming for all cameras"""
        for camera_id in self.cameras:
            self.stop_detection(camera_id)
            self.stop_hls_streaming(camera_id)
        
        self.running = False
        logger.info("Stopped all cameras")
    
    def get_status(self) -> Dict:
        """Get status of all cameras"""
        status = {
            'running': self.running,
            'cameras': {},
            'stream_urls': self.get_stream_urls()
        }
        
        for camera_id, camera_info in self.cameras.items():
            status['cameras'][camera_id] = {
                'status': camera_info['status'],
                'intersection_id': camera_info['intersection_id'],
                'coordinates': camera_info['coordinates']
            }
        
        return status

def create_demo_setup():
    """Create demo camera setup for testing"""
    integration = CVDemoIntegration()
    
    # Add demo cameras
    integration.add_camera(
        camera_id="cam_001",
        rtsp_url="rtsp://demo:demo@ipvmdemo.com/axis-media/media.amp",
        intersection_id="intersection_1",
        coordinates={"lat": 20.2961, "lng": 85.8245}
    )
    
    integration.add_camera(
        camera_id="cam_002",
        rtsp_url="rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov",
        intersection_id="intersection_2", 
        coordinates={"lat": 20.2971, "lng": 85.8255}
    )
    
    return integration

if __name__ == '__main__':
    # Create demo setup
    demo = create_demo_setup()
    
    try:
        # Start all cameras
        demo.start_all_cameras()
        
        # Keep running
        while True:
            time.sleep(10)
            status = demo.get_status()
            print(f"Status: {json.dumps(status, indent=2)}")
            
    except KeyboardInterrupt:
        print("Shutting down...")
        demo.stop_all_cameras()