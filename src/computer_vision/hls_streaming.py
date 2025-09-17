"""
HLS Streaming Pipeline for Computer Vision
Converts RTSP feeds to HLS streams for frontend integration
"""

import subprocess
import os
import time
import threading
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import requests
from flask import Flask, jsonify, send_file
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HLSStreamingPipeline:
    def __init__(self, output_dir: str = "hls_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.streams = {}  # {camera_id: stream_info}
        self.processes = {}  # {camera_id: ffmpeg_process}
        self.running = False
        
    def start_stream(self, camera_id: str, rtsp_url: str, intersection_id: str, 
                    coordinates: Dict[str, float]) -> bool:
        """Start HLS streaming for a camera"""
        try:
            # Create camera-specific output directory
            camera_dir = self.output_dir / camera_id
            camera_dir.mkdir(exist_ok=True)
            
            # HLS configuration
            hls_segment_time = 2  # 2-second segments
            hls_playlist_size = 5  # Keep 5 segments in playlist
            hls_path = camera_dir / "stream.m3u8"
            
            # FFmpeg command for RTSP to HLS conversion
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', rtsp_url,
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-c:a', 'aac',
                '-f', 'hls',
                '-hls_time', str(hls_segment_time),
                '-hls_list_size', str(hls_playlist_size),
                '-hls_flags', 'delete_segments',
                '-hls_segment_filename', str(camera_dir / 'segment_%03d.ts'),
                str(hls_path)
            ]
            
            # Start FFmpeg process
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(camera_dir)
            )
            
            # Store stream information
            self.streams[camera_id] = {
                'camera_id': camera_id,
                'intersection_id': intersection_id,
                'rtsp_url': rtsp_url,
                'hls_url': f"/hls/{camera_id}/stream.m3u8",
                'coordinates': coordinates,
                'status': 'starting',
                'start_time': time.time(),
                'process': process
            }
            
            self.processes[camera_id] = process
            
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_stream,
                args=(camera_id,)
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            
            logger.info(f"Started HLS stream for camera {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start HLS stream for camera {camera_id}: {e}")
            return False
    
    def stop_stream(self, camera_id: str) -> bool:
        """Stop HLS streaming for a camera"""
        try:
            if camera_id in self.processes:
                process = self.processes[camera_id]
                process.terminate()
                process.wait(timeout=5)
                del self.processes[camera_id]
                
            if camera_id in self.streams:
                self.streams[camera_id]['status'] = 'stopped'
                
            logger.info(f"Stopped HLS stream for camera {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop HLS stream for camera {camera_id}: {e}")
            return False
    
    def _monitor_stream(self, camera_id: str):
        """Monitor stream health and restart if needed"""
        while camera_id in self.processes:
            process = self.processes[camera_id]
            
            # Check if process is still running
            if process.poll() is not None:
                logger.warning(f"FFmpeg process for camera {camera_id} died, restarting...")
                
                # Restart stream
                if camera_id in self.streams:
                    stream_info = self.streams[camera_id]
                    self.stop_stream(camera_id)
                    time.sleep(2)
                    self.start_stream(
                        camera_id,
                        stream_info['rtsp_url'],
                        stream_info['intersection_id'],
                        stream_info['coordinates']
                    )
                break
            
            time.sleep(5)  # Check every 5 seconds
    
    def get_stream_info(self, camera_id: str) -> Optional[Dict]:
        """Get stream information for a camera"""
        return self.streams.get(camera_id)
    
    def get_all_streams(self) -> List[Dict]:
        """Get information for all active streams"""
        return list(self.streams.values())
    
    def cleanup_old_segments(self, max_age_hours: int = 1):
        """Clean up old HLS segments to save disk space"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for camera_id, stream_info in self.streams.items():
            camera_dir = self.output_dir / camera_id
            if camera_dir.exists():
                for file_path in camera_dir.glob("*.ts"):
                    if current_time - file_path.stat().st_mtime > max_age_seconds:
                        file_path.unlink()

# Flask app for serving HLS streams and API
app = Flask(__name__)
hls_pipeline = HLSStreamingPipeline()

@app.route('/hls/<camera_id>/<filename>')
def serve_hls_file(camera_id: str, filename: str):
    """Serve HLS files (playlist and segments)"""
    file_path = hls_pipeline.output_dir / camera_id / filename
    if file_path.exists():
        return send_file(str(file_path))
    else:
        return "File not found", 404

@app.route('/cv/streams')
def get_streams():
    """Get all available HLS streams"""
    streams = hls_pipeline.get_all_streams()
    return jsonify({
        'status': 'success',
        'streams': streams,
        'count': len(streams)
    })

@app.route('/cv/streams/<camera_id>')
def get_stream(camera_id: str):
    """Get specific stream information"""
    stream_info = hls_pipeline.get_stream_info(camera_id)
    if stream_info:
        return jsonify({
            'status': 'success',
            'stream': stream_info
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'Stream {camera_id} not found'
        }), 404

@app.route('/cv/streams/<camera_id>/start', methods=['POST'])
def start_stream_endpoint():
    """Start HLS streaming for a camera"""
    from flask import request
    data = request.get_json()
    
    camera_id = data.get('camera_id')
    rtsp_url = data.get('rtsp_url')
    intersection_id = data.get('intersection_id')
    coordinates = data.get('coordinates', {})
    
    if not all([camera_id, rtsp_url, intersection_id]):
        return jsonify({
            'status': 'error',
            'message': 'Missing required fields: camera_id, rtsp_url, intersection_id'
        }), 400
    
    success = hls_pipeline.start_stream(camera_id, rtsp_url, intersection_id, coordinates)
    
    if success:
        return jsonify({
            'status': 'success',
            'message': f'Stream started for camera {camera_id}'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'Failed to start stream for camera {camera_id}'
        }), 500

@app.route('/cv/streams/<camera_id>/stop', methods=['POST'])
def stop_stream_endpoint(camera_id: str):
    """Stop HLS streaming for a camera"""
    success = hls_pipeline.stop_stream(camera_id)
    
    if success:
        return jsonify({
            'status': 'success',
            'message': f'Stream stopped for camera {camera_id}'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'Failed to stop stream for camera {camera_id}'
        }), 500

def create_demo_streams():
    """Create demo streams for testing"""
    demo_cameras = [
        {
            'camera_id': 'cam_001',
            'rtsp_url': 'rtsp://demo:demo@ipvmdemo.com/axis-media/media.amp',
            'intersection_id': 'intersection_1',
            'coordinates': {'lat': 20.2961, 'lng': 85.8245}
        },
        {
            'camera_id': 'cam_002', 
            'rtsp_url': 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov',
            'intersection_id': 'intersection_2',
            'coordinates': {'lat': 20.2971, 'lng': 85.8255}
        }
    ]
    
    for camera in demo_cameras:
        hls_pipeline.start_stream(
            camera['camera_id'],
            camera['rtsp_url'],
            camera['intersection_id'],
            camera['coordinates']
        )
        time.sleep(1)  # Stagger starts

if __name__ == '__main__':
    # Create demo streams for testing
    create_demo_streams()
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(
        target=lambda: [time.sleep(3600), hls_pipeline.cleanup_old_segments()],
        daemon=True
    )
    cleanup_thread.start()
    
    # Start Flask app
    app.run(host='0.0.0.0', port=5001, debug=True)
