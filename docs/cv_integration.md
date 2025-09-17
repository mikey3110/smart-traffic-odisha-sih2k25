# Computer Vision Integration Guide

## Overview

This document describes the integration of Computer Vision (CV) components with the Smart Traffic Management System, including HLS streaming, vehicle detection, and frontend dashboard integration.

## Architecture

### Components

1. **HLS Streaming Pipeline** (`hls_streaming.py`)
   - Converts RTSP feeds to HLS streams
   - Serves video content to frontend
   - Manages stream lifecycle

2. **CV Demo Integration** (`demo_integration.py`)
   - Coordinates vehicle detection and streaming
   - Manages multiple camera feeds
   - Integrates with backend APIs

3. **Backend API Integration**
   - `/cv/counts` - Vehicle count data
   - `/cv/streams` - Stream information
   - `/cv/frames` - Frame data (if needed)

## Setup

### Prerequisites

```bash
# Install required packages
pip install opencv-python ultralytics flask requests

# Install FFmpeg for HLS streaming
# Windows: Download from https://ffmpeg.org/download.html
# Linux: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
```

### Configuration

1. **Start HLS Streaming Service**:
   ```bash
   cd src/computer_vision
   python hls_streaming.py
   ```

2. **Start CV Demo Integration**:
   ```bash
   python demo_integration.py
   ```

3. **Start Backend Service**:
   ```bash
   cd src/backend
   python main.py
   ```

## API Endpoints

### HLS Streaming Service (Port 5001)

#### Start Stream
```http
POST /cv/streams/{camera_id}/start
Content-Type: application/json

{
    "camera_id": "cam_001",
    "rtsp_url": "rtsp://example.com/stream",
    "intersection_id": "intersection_1",
    "coordinates": {"lat": 20.2961, "lng": 85.8245}
}
```

#### Stop Stream
```http
POST /cv/streams/{camera_id}/stop
```

#### Get All Streams
```http
GET /cv/streams
```

Response:
```json
{
    "status": "success",
    "streams": [
        {
            "camera_id": "cam_001",
            "intersection_id": "intersection_1",
            "hls_url": "/hls/cam_001/stream.m3u8",
            "coordinates": {"lat": 20.2961, "lng": 85.8245},
            "status": "active"
        }
    ],
    "count": 1
}
```

#### Serve HLS Files
```http
GET /hls/{camera_id}/{filename}
```

### Backend API (Port 8000)

#### Vehicle Counts
```http
POST /cv/counts
Content-Type: application/json

{
    "camera_id": "cam_001",
    "intersection_id": "intersection_1",
    "timestamp": 1640995200,
    "total_vehicles": 15,
    "counts_by_class": {
        "car": 10,
        "motorcycle": 3,
        "bus": 1,
        "truck": 1
    },
    "coordinates": {"lat": 20.2961, "lng": 85.8245}
}
```

## Frontend Integration

### Camera Feed Overlay

The frontend can display camera feeds using the HLS streams:

```typescript
// Example React component
const CameraFeed = ({ cameraId, intersectionId }) => {
  const [streamUrl, setStreamUrl] = useState(null);
  
  useEffect(() => {
    // Get stream URL from backend
    fetch(`/api/cv/streams/${cameraId}`)
      .then(res => res.json())
      .then(data => {
        if (data.stream) {
          setStreamUrl(`http://localhost:5001${data.stream.hls_url}`);
        }
      });
  }, [cameraId]);
  
  return (
    <div className="camera-feed">
      {streamUrl && (
        <video
          src={streamUrl}
          autoPlay
          muted
          controls
          style={{ width: '100%', height: '200px' }}
        />
      )}
    </div>
  );
};
```

### Vehicle Count Display

```typescript
// Example vehicle count component
const VehicleCount = ({ intersectionId }) => {
  const [counts, setCounts] = useState(null);
  
  useEffect(() => {
    const interval = setInterval(() => {
      fetch(`/api/cv/counts/${intersectionId}`)
        .then(res => res.json())
        .then(data => setCounts(data));
    }, 30000); // Update every 30 seconds
    
    return () => clearInterval(interval);
  }, [intersectionId]);
  
  return (
    <div className="vehicle-count">
      <h3>Vehicle Count</h3>
      {counts && (
        <div>
          <p>Total: {counts.total_vehicles}</p>
          <p>Cars: {counts.counts_by_class.car}</p>
          <p>Motorcycles: {counts.counts_by_class.motorcycle}</p>
          <p>Buses: {counts.counts_by_class.bus}</p>
          <p>Trucks: {counts.counts_by_class.truck}</p>
        </div>
      )}
    </div>
  );
};
```

## Testing

### Run API Tests

```bash
cd src/computer_vision
python -m pytest tests/test_cv_integration.py -v
```

### Test HLS Streaming

```bash
# Start HLS service
python hls_streaming.py

# Test stream creation
curl -X POST http://localhost:5001/cv/streams/test_cam/start \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "test_cam",
    "rtsp_url": "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov",
    "intersection_id": "test_intersection",
    "coordinates": {"lat": 0, "lng": 0}
  }'

# Check stream status
curl http://localhost:5001/cv/streams
```

### Test Vehicle Detection

```bash
# Start demo integration
python demo_integration.py

# Check logs for detection output
# Should see: "Camera cam_001: X vehicles detected"
```

## Performance

### Latency Requirements

- **HLS Stream Latency**: < 2 seconds
- **Vehicle Count Update**: Every 30 seconds
- **Detection Processing**: < 100ms per frame

### Optimization

1. **Frame Skipping**: Process every 10th frame to reduce CPU load
2. **ONNX Model**: Use optimized ONNX model for 2x speedup
3. **HLS Segments**: 2-second segments for low latency
4. **Caching**: Cache model and stream metadata

## Troubleshooting

### Common Issues

1. **FFmpeg Not Found**
   ```
   Error: FFmpeg executable not found
   Solution: Install FFmpeg and add to PATH
   ```

2. **RTSP Connection Failed**
   ```
   Error: Failed to open video stream
   Solution: Check RTSP URL and network connectivity
   ```

3. **HLS Stream Not Playing**
   ```
   Error: Video not loading in browser
   Solution: Check CORS settings and stream URL
   ```

4. **YOLO Model Loading Failed**
   ```
   Error: Failed to load YOLO model
   Solution: Download model file: yolov8n.pt
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

```bash
# Check HLS service
curl http://localhost:5001/cv/streams

# Check backend service
curl http://localhost:8000/health

# Check vehicle counts
curl http://localhost:8000/cv/counts
```

## Production Deployment

### Docker Configuration

```dockerfile
# HLS Streaming Service
FROM python:3.9-slim
RUN apt-get update && apt-get install -y ffmpeg
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5001
CMD ["python", "hls_streaming.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cv-hls-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: cv-hls
  template:
    metadata:
      labels:
        app: cv-hls
    spec:
      containers:
      - name: hls-service
        image: cv-hls:latest
        ports:
        - containerPort: 5001
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## Security Considerations

1. **RTSP Authentication**: Use secure RTSP URLs with authentication
2. **API Rate Limiting**: Implement rate limiting for API endpoints
3. **CORS Configuration**: Restrict CORS to trusted domains
4. **Input Validation**: Validate all input parameters
5. **Error Handling**: Don't expose sensitive information in errors

## Monitoring

### Metrics to Track

- Stream uptime and latency
- Vehicle detection accuracy
- API response times
- Resource usage (CPU, memory)
- Error rates

### Logging

```python
# Structured logging
import structlog
logger = structlog.get_logger()

logger.info("stream_started", 
           camera_id=camera_id, 
           intersection_id=intersection_id,
           latency=latency)
```

---

**Computer Vision Integration Guide v1.0 - Smart India Hackathon 2025**
