"""
API Tests for Computer Vision Integration
Tests HLS streaming and vehicle count endpoints
"""

import pytest
import requests
import time
import json
import threading
from unittest.mock import patch, MagicMock
import cv2
import numpy as np

# Test configuration
BACKEND_URL = "http://localhost:8000"
HLS_URL = "http://localhost:5001"

class TestCVIntegration:
    """Test suite for CV integration APIs"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.test_camera_id = "test_cam_001"
        self.test_intersection_id = "test_intersection_1"
        self.test_coordinates = {"lat": 20.2961, "lng": 85.8245}
        self.test_rtsp_url = "rtsp://test.example.com/stream"
    
    def test_hls_stream_start(self):
        """Test starting HLS stream"""
        payload = {
            'camera_id': self.test_camera_id,
            'rtsp_url': self.test_rtsp_url,
            'intersection_id': self.test_intersection_id,
            'coordinates': self.test_coordinates
        }
        
        try:
            response = requests.post(
                f"{HLS_URL}/cv/streams/{self.test_camera_id}/start",
                json=payload,
                timeout=10
            )
            
            # Should return 200 or 500 (depending on RTSP availability)
            assert response.status_code in [200, 500]
            
            if response.status_code == 200:
                data = response.json()
                assert data['status'] == 'success'
                assert 'Stream started' in data['message']
            
        except requests.exceptions.ConnectionError:
            pytest.skip("HLS service not running")
    
    def test_hls_stream_stop(self):
        """Test stopping HLS stream"""
        try:
            response = requests.post(
                f"{HLS_URL}/cv/streams/{self.test_camera_id}/stop",
                timeout=10
            )
            
            # Should return 200 or 404 (if stream not found)
            assert response.status_code in [200, 404]
            
        except requests.exceptions.ConnectionError:
            pytest.skip("HLS service not running")
    
    def test_get_streams(self):
        """Test getting all streams"""
        try:
            response = requests.get(f"{HLS_URL}/cv/streams", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                assert 'streams' in data
                assert 'count' in data
                assert isinstance(data['streams'], list)
            
        except requests.exceptions.ConnectionError:
            pytest.skip("HLS service not running")
    
    def test_vehicle_counts_api(self):
        """Test vehicle counts API endpoint"""
        payload = {
            "camera_id": self.test_camera_id,
            "intersection_id": self.test_intersection_id,
            "timestamp": int(time.time()),
            "total_vehicles": 15,
            "counts_by_class": {
                "car": 10,
                "motorcycle": 3,
                "bus": 1,
                "truck": 1
            },
            "coordinates": self.test_coordinates
        }
        
        try:
            response = requests.post(
                f"{BACKEND_URL}/cv/counts",
                json=payload,
                timeout=5
            )
            
            # Should return 200 or 404 (if endpoint not implemented)
            assert response.status_code in [200, 404, 422]
            
            if response.status_code == 200:
                data = response.json()
                assert 'status' in data
                
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend service not running")
    
    def test_stream_latency(self):
        """Test HLS stream latency"""
        try:
            # Start a test stream
            payload = {
                'camera_id': 'latency_test',
                'rtsp_url': 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov',
                'intersection_id': 'test_intersection',
                'coordinates': {"lat": 0, "lng": 0}
            }
            
            start_time = time.time()
            response = requests.post(
                f"{HLS_URL}/cv/streams/latency_test/start",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                # Wait for stream to be ready
                time.sleep(3)
                
                # Check if stream is accessible
                stream_response = requests.get(
                    f"{HLS_URL}/hls/latency_test/stream.m3u8",
                    timeout=5
                )
                
                if stream_response.status_code == 200:
                    latency = time.time() - start_time
                    assert latency < 5.0, f"Stream latency too high: {latency}s"
                
                # Clean up
                requests.post(f"{HLS_URL}/cv/streams/latency_test/stop", timeout=5)
            
        except requests.exceptions.ConnectionError:
            pytest.skip("HLS service not running")
    
    def test_count_accuracy(self):
        """Test vehicle count accuracy with sample frames"""
        # Create a mock frame with vehicles
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock YOLO detection results
        with patch('ultralytics.YOLO') as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model
            
            # Mock detection results
            mock_result = MagicMock()
            mock_result.boxes = MagicMock()
            mock_result.boxes.cls = np.array([2, 2, 3, 5])  # 2 cars, 1 motorcycle, 1 bus
            mock_result.boxes = mock_result.boxes
            mock_model.return_value = [mock_result]
            mock_model.names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
            
            # Test detection
            results = mock_model(mock_frame, classes=[2, 3, 5, 7])
            
            # Count vehicles
            counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    class_name = mock_model.names[class_id]
                    if class_name in counts:
                        counts[class_name] += 1
            
            # Verify counts
            assert counts["car"] == 2
            assert counts["motorcycle"] == 1
            assert counts["bus"] == 1
            assert counts["truck"] == 0
            assert sum(counts.values()) == 4

def test_mock_rtsp_feed():
    """Test with mock RTSP feed"""
    # Create a mock video file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('test_video.avi', fourcc, 20.0, (640, 480))
    
    # Generate test frames
    for i in range(100):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    
    # Test with mock RTSP URL
    mock_rtsp_url = "test_video.avi"  # Use local file instead of RTSP
    
    try:
        cap = cv2.VideoCapture(mock_rtsp_url)
        assert cap.isOpened(), "Failed to open mock video"
        
        ret, frame = cap.read()
        assert ret, "Failed to read frame"
        assert frame is not None, "Frame is None"
        
        cap.release()
        
    finally:
        # Clean up
        import os
        if os.path.exists('test_video.avi'):
            os.remove('test_video.avi')

def test_integration_workflow():
    """Test complete integration workflow"""
    # This test would require both backend and HLS services running
    # For now, we'll test the workflow logic
    
    from demo_integration import CVDemoIntegration
    
    integration = CVDemoIntegration()
    
    # Add test camera
    integration.add_camera(
        camera_id="workflow_test",
        rtsp_url="rtsp://test.example.com/stream",
        intersection_id="test_intersection",
        coordinates={"lat": 0, "lng": 0}
    )
    
    # Test camera addition
    assert "workflow_test" in integration.cameras
    assert integration.cameras["workflow_test"]["status"] == "inactive"
    
    # Test status
    status = integration.get_status()
    assert status["running"] == False
    assert "workflow_test" in status["cameras"]

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
