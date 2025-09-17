"""
Real-Time Data Streaming for ML Traffic Optimization
Phase 4: API Integration & Demo Preparation

Features:
- WebSocket endpoints for live ML metrics streaming
- Server-Sent Events for dashboard updates
- Data transformation pipelines for frontend consumption
- Caching mechanisms for frequently accessed metrics
- Data compression for high-frequency updates
"""

import asyncio
import json
import logging
import time
import zlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import redis
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from sse_starlette.sse import EventSourceResponse
import yaml
import pickle
import base64
from collections import deque
import threading
import queue


class StreamType(Enum):
    """Stream type enumeration"""
    WEBSOCKET = "websocket"
    SERVER_SENT_EVENTS = "sse"
    POLLING = "polling"


class DataFormat(Enum):
    """Data format enumeration"""
    JSON = "json"
    BINARY = "binary"
    COMPRESSED = "compressed"


@dataclass
class StreamMessage:
    """Stream message structure"""
    message_id: str
    stream_type: StreamType
    data_type: str
    payload: Any
    timestamp: datetime
    compression: bool = False
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class StreamConfig:
    """Stream configuration"""
    max_connections: int = 1000
    message_buffer_size: int = 10000
    compression_threshold: int = 1024  # bytes
    heartbeat_interval: int = 30  # seconds
    cleanup_interval: int = 300  # seconds
    max_message_age: int = 3600  # seconds


class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Active connections
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Message queues for each connection
        self.message_queues: Dict[str, asyncio.Queue] = {}
        
        # Statistics
        self.total_connections = 0
        self.active_connections_count = 0
        self.messages_sent = 0
        self.messages_failed = 0
        
        self.logger.info("Connection Manager initialized")
    
    async def connect(self, websocket: WebSocket, client_id: str, 
                     subscription_filters: Dict[str, Any] = None) -> bool:
        """Accept new WebSocket connection"""
        try:
            await websocket.accept()
            
            # Check connection limit
            if len(self.active_connections) >= self.config.max_connections:
                await websocket.close(code=1013, reason="Server overloaded")
                return False
            
            # Store connection
            self.active_connections[client_id] = websocket
            self.connection_metadata[client_id] = {
                'connected_at': datetime.now(),
                'subscription_filters': subscription_filters or {},
                'last_heartbeat': datetime.now(),
                'message_count': 0
            }
            
            # Create message queue
            self.message_queues[client_id] = asyncio.Queue(maxsize=self.config.message_buffer_size)
            
            # Update statistics
            self.total_connections += 1
            self.active_connections_count += 1
            
            self.logger.info(f"Client {client_id} connected. Total connections: {self.active_connections_count}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error connecting client {client_id}: {e}")
            return False
    
    def disconnect(self, client_id: str):
        """Disconnect client"""
        try:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
                del self.connection_metadata[client_id]
                del self.message_queues[client_id]
                
                self.active_connections_count -= 1
                self.logger.info(f"Client {client_id} disconnected. Active connections: {self.active_connections_count}")
                
        except Exception as e:
            self.logger.error(f"Error disconnecting client {client_id}: {e}")
    
    async def send_message(self, client_id: str, message: StreamMessage) -> bool:
        """Send message to specific client"""
        try:
            if client_id not in self.active_connections:
                return False
            
            websocket = self.active_connections[client_id]
            
            # Prepare message
            message_data = {
                'message_id': message.message_id,
                'data_type': message.data_type,
                'payload': message.payload,
                'timestamp': message.timestamp.isoformat(),
                'compression': message.compression
            }
            
            # Compress if needed
            if message.compression and len(str(message_data)) > self.config.compression_threshold:
                message_data = self._compress_message(message_data)
            
            # Send message
            await websocket.send_text(json.dumps(message_data))
            
            # Update statistics
            self.connection_metadata[client_id]['message_count'] += 1
            self.connection_metadata[client_id]['last_heartbeat'] = datetime.now()
            self.messages_sent += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending message to {client_id}: {e}")
            self.messages_failed += 1
            return False
    
    async def broadcast_message(self, message: StreamMessage, 
                              filter_func: Callable[[str], bool] = None) -> int:
        """Broadcast message to all connected clients"""
        sent_count = 0
        
        for client_id in list(self.active_connections.keys()):
            try:
                # Apply filter if provided
                if filter_func and not filter_func(client_id):
                    continue
                
                # Check subscription filters
                metadata = self.connection_metadata.get(client_id, {})
                filters = metadata.get('subscription_filters', {})
                
                if self._matches_filters(message, filters):
                    success = await self.send_message(client_id, message)
                    if success:
                        sent_count += 1
                        
            except Exception as e:
                self.logger.error(f"Error broadcasting to {client_id}: {e}")
        
        return sent_count
    
    def _matches_filters(self, message: StreamMessage, filters: Dict[str, Any]) -> bool:
        """Check if message matches subscription filters"""
        if not filters:
            return True
        
        # Check data type filter
        if 'data_types' in filters:
            if message.data_type not in filters['data_types']:
                return False
        
        # Check priority filter
        if 'min_priority' in filters:
            if message.priority > filters['min_priority']:
                return False
        
        return True
    
    def _compress_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress message data"""
        try:
            json_str = json.dumps(message_data)
            compressed = zlib.compress(json_str.encode())
            encoded = base64.b64encode(compressed).decode()
            
            return {
                'compressed': True,
                'data': encoded
            }
        except Exception as e:
            self.logger.error(f"Error compressing message: {e}")
            return message_data
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'total_connections': self.total_connections,
            'active_connections': self.active_connections_count,
            'messages_sent': self.messages_sent,
            'messages_failed': self.messages_failed,
            'success_rate': self.messages_sent / max(self.messages_sent + self.messages_failed, 1)
        }


class DataTransformer:
    """Data transformation pipeline for frontend consumption"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Transformation rules
        self.transform_rules = config.get('transform_rules', {})
        
        # Data aggregators
        self.aggregators = {
            'minute': self._aggregate_by_minute,
            'hour': self._aggregate_by_hour,
            'day': self._aggregate_by_day
        }
        
        self.logger.info("Data Transformer initialized")
    
    def transform_metrics(self, raw_metrics: Dict[str, Any], 
                         target_format: str = 'dashboard') -> Dict[str, Any]:
        """Transform raw metrics for frontend consumption"""
        try:
            if target_format == 'dashboard':
                return self._transform_for_dashboard(raw_metrics)
            elif target_format == 'chart':
                return self._transform_for_chart(raw_metrics)
            elif target_format == 'table':
                return self._transform_for_table(raw_metrics)
            else:
                return raw_metrics
                
        except Exception as e:
            self.logger.error(f"Error transforming metrics: {e}")
            return raw_metrics
    
    def _transform_for_dashboard(self, raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Transform metrics for dashboard display"""
        return {
            'intersection_id': raw_metrics.get('intersection_id'),
            'timestamp': raw_metrics.get('timestamp'),
            'performance': {
                'wait_time_reduction': {
                    'value': raw_metrics.get('wait_time_reduction', 0),
                    'trend': self._calculate_trend(raw_metrics, 'wait_time_reduction'),
                    'status': self._get_status(raw_metrics.get('wait_time_reduction', 0), 'wait_time')
                },
                'throughput_increase': {
                    'value': raw_metrics.get('throughput_increase', 0),
                    'trend': self._calculate_trend(raw_metrics, 'throughput_increase'),
                    'status': self._get_status(raw_metrics.get('throughput_increase', 0), 'throughput')
                },
                'fuel_consumption_reduction': {
                    'value': raw_metrics.get('fuel_consumption_reduction', 0),
                    'trend': self._calculate_trend(raw_metrics, 'fuel_consumption_reduction'),
                    'status': self._get_status(raw_metrics.get('fuel_consumption_reduction', 0), 'fuel')
                }
            },
            'traffic_conditions': {
                'queue_length': raw_metrics.get('queue_length', 0),
                'delay_time': raw_metrics.get('delay_time', 0),
                'signal_efficiency': raw_metrics.get('signal_efficiency', 0),
                'safety_score': raw_metrics.get('safety_score', 0)
            },
            'alerts': self._generate_alerts(raw_metrics)
        }
    
    def _transform_for_chart(self, raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Transform metrics for chart visualization"""
        return {
            'x': raw_metrics.get('timestamp'),
            'y': {
                'wait_time_reduction': raw_metrics.get('wait_time_reduction', 0),
                'throughput_increase': raw_metrics.get('throughput_increase', 0),
                'fuel_consumption_reduction': raw_metrics.get('fuel_consumption_reduction', 0)
            },
            'metadata': {
                'intersection_id': raw_metrics.get('intersection_id'),
                'data_quality': self._assess_data_quality(raw_metrics)
            }
        }
    
    def _transform_for_table(self, raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Transform metrics for table display"""
        return {
            'intersection_id': raw_metrics.get('intersection_id'),
            'timestamp': raw_metrics.get('timestamp'),
            'wait_time_reduction': f"{raw_metrics.get('wait_time_reduction', 0):.1f}%",
            'throughput_increase': f"{raw_metrics.get('throughput_increase', 0):.1f}%",
            'fuel_consumption_reduction': f"{raw_metrics.get('fuel_consumption_reduction', 0):.1f}%",
            'emission_reduction': f"{raw_metrics.get('emission_reduction', 0):.1f}%",
            'queue_length': raw_metrics.get('queue_length', 0),
            'delay_time': f"{raw_metrics.get('delay_time', 0):.1f}s",
            'status': self._get_overall_status(raw_metrics)
        }
    
    def _calculate_trend(self, metrics: Dict[str, Any], metric_name: str) -> str:
        """Calculate trend for metric"""
        # This would typically compare with historical data
        # For now, return a simple trend based on value
        value = metrics.get(metric_name, 0)
        if value > 30:
            return 'up'
        elif value > 15:
            return 'stable'
        else:
            return 'down'
    
    def _get_status(self, value: float, metric_type: str) -> str:
        """Get status for metric value"""
        thresholds = {
            'wait_time': {'excellent': 40, 'good': 25, 'fair': 15},
            'throughput': {'excellent': 30, 'good': 20, 'fair': 10},
            'fuel': {'excellent': 20, 'good': 15, 'fair': 10}
        }
        
        if metric_type not in thresholds:
            return 'unknown'
        
        thresh = thresholds[metric_type]
        if value >= thresh['excellent']:
            return 'excellent'
        elif value >= thresh['good']:
            return 'good'
        elif value >= thresh['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _generate_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on metrics"""
        alerts = []
        
        # Performance alerts
        if metrics.get('wait_time_reduction', 0) < 10:
            alerts.append({
                'type': 'warning',
                'message': 'Low wait time reduction detected',
                'metric': 'wait_time_reduction',
                'value': metrics.get('wait_time_reduction', 0)
            })
        
        if metrics.get('throughput_increase', 0) < 5:
            alerts.append({
                'type': 'warning',
                'message': 'Low throughput increase detected',
                'metric': 'throughput_increase',
                'value': metrics.get('throughput_increase', 0)
            })
        
        # Safety alerts
        if metrics.get('safety_score', 100) < 80:
            alerts.append({
                'type': 'critical',
                'message': 'Safety score below threshold',
                'metric': 'safety_score',
                'value': metrics.get('safety_score', 100)
            })
        
        return alerts
    
    def _assess_data_quality(self, metrics: Dict[str, Any]) -> str:
        """Assess data quality"""
        required_fields = ['wait_time_reduction', 'throughput_increase', 'fuel_consumption_reduction']
        present_fields = sum(1 for field in required_fields if field in metrics and metrics[field] is not None)
        
        quality_score = present_fields / len(required_fields)
        
        if quality_score >= 0.9:
            return 'excellent'
        elif quality_score >= 0.7:
            return 'good'
        elif quality_score >= 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _get_overall_status(self, metrics: Dict[str, Any]) -> str:
        """Get overall status"""
        wait_time = metrics.get('wait_time_reduction', 0)
        throughput = metrics.get('throughput_increase', 0)
        fuel = metrics.get('fuel_consumption_reduction', 0)
        
        avg_performance = (wait_time + throughput + fuel) / 3
        
        if avg_performance >= 25:
            return 'excellent'
        elif avg_performance >= 15:
            return 'good'
        elif avg_performance >= 10:
            return 'fair'
        else:
            return 'poor'
    
    def _aggregate_by_minute(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate data by minute"""
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample to minute frequency
        minute_data = df.resample('1T').mean()
        
        return {
            'data': minute_data.to_dict('records'),
            'aggregation': 'minute',
            'count': len(minute_data)
        }
    
    def _aggregate_by_hour(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate data by hour"""
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample to hour frequency
        hour_data = df.resample('1H').mean()
        
        return {
            'data': hour_data.to_dict('records'),
            'aggregation': 'hour',
            'count': len(hour_data)
        }
    
    def _aggregate_by_day(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate data by day"""
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Resample to day frequency
        day_data = df.resample('1D').mean()
        
        return {
            'data': day_data.to_dict('records'),
            'aggregation': 'day',
            'count': len(day_data)
        }


class MetricsCache:
    """Caching system for frequently accessed metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Redis client for caching
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            db=config.get('redis_db', 1)
        )
        
        # Cache configuration
        self.default_ttl = config.get('default_ttl', 3600)  # 1 hour
        self.max_cache_size = config.get('max_cache_size', 10000)
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        
        self.logger.info("Metrics Cache initialized")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                self.hits += 1
                return json.loads(cached_data)
            else:
                self.misses += 1
                return None
        except Exception as e:
            self.logger.error(f"Error getting from cache: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or self.default_ttl
            serialized_value = json.dumps(value, default=str)
            self.redis_client.setex(key, ttl, serialized_value)
            return True
        except Exception as e:
            self.logger.error(f"Error setting cache: {e}")
            return False
    
    async def get_or_set(self, key: str, factory_func: Callable, ttl: int = None) -> Any:
        """Get from cache or set using factory function"""
        value = await self.get(key)
        if value is None:
            value = await factory_func()
            await self.set(key, value, ttl)
        return value
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / max(total_requests, 1)
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }


class RealTimeStreamingAPI:
    """Main real-time streaming API"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Real-Time ML Metrics Streaming API",
            description="WebSocket and SSE endpoints for live ML metrics",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize components
        self.connection_manager = ConnectionManager(StreamConfig(**config.get('stream_config', {})))
        self.data_transformer = DataTransformer(config.get('data_transformer', {}))
        self.metrics_cache = MetricsCache(config.get('cache', {}))
        
        # Message queue for broadcasting
        self.message_queue = asyncio.Queue()
        
        # Background tasks
        self.background_tasks = set()
        
        # Setup routes
        self._setup_routes()
        
        # Start background tasks
        asyncio.create_task(self._start_background_tasks())
        
        self.logger.info("Real-Time Streaming API initialized")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: str):
            """WebSocket endpoint for real-time metrics"""
            await self.connection_manager.connect(websocket, client_id)
            
            try:
                while True:
                    # Send heartbeat
                    await asyncio.sleep(30)
                    heartbeat = StreamMessage(
                        message_id=str(uuid.uuid4()),
                        stream_type=StreamType.WEBSOCKET,
                        data_type='heartbeat',
                        payload={'status': 'alive'},
                        timestamp=datetime.now()
                    )
                    await self.connection_manager.send_message(client_id, heartbeat)
                    
            except WebSocketDisconnect:
                self.connection_manager.disconnect(client_id)
            except Exception as e:
                self.logger.error(f"WebSocket error for {client_id}: {e}")
                self.connection_manager.disconnect(client_id)
        
        @self.app.get("/sse/{client_id}")
        async def sse_endpoint(client_id: str):
            """Server-Sent Events endpoint"""
            async def event_generator():
                while True:
                    try:
                        # Get message from queue
                        message = await self.message_queue.get()
                        
                        # Transform message for SSE
                        sse_data = {
                            'id': message.message_id,
                            'event': message.data_type,
                            'data': json.dumps(message.payload)
                        }
                        
                        yield f"id: {sse_data['id']}\n"
                        yield f"event: {sse_data['event']}\n"
                        yield f"data: {sse_data['data']}\n\n"
                        
                        await asyncio.sleep(1)
                        
                    except Exception as e:
                        self.logger.error(f"SSE error for {client_id}: {e}")
                        break
            
            return EventSourceResponse(event_generator())
        
        @self.app.post("/stream/metrics")
        async def submit_metrics(metrics: Dict[str, Any]):
            """Submit metrics for streaming"""
            try:
                # Transform metrics
                transformed_metrics = self.data_transformer.transform_metrics(metrics)
                
                # Create stream message
                message = StreamMessage(
                    message_id=str(uuid.uuid4()),
                    stream_type=StreamType.WEBSOCKET,
                    data_type='metrics',
                    payload=transformed_metrics,
                    timestamp=datetime.now(),
                    compression=True
                )
                
                # Broadcast to all connected clients
                sent_count = await self.connection_manager.broadcast_message(message)
                
                # Add to SSE queue
                await self.message_queue.put(message)
                
                return {
                    "status": "metrics_submitted",
                    "sent_to_clients": sent_count,
                    "message_id": message.message_id
                }
                
            except Exception as e:
                self.logger.error(f"Error submitting metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/stream/stats")
        async def get_stream_stats():
            """Get streaming statistics"""
            connection_stats = self.connection_manager.get_connection_stats()
            cache_stats = self.metrics_cache.get_cache_stats()
            
            return {
                "connections": connection_stats,
                "cache": cache_stats,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/stream/metrics/{intersection_id}")
        async def get_cached_metrics(intersection_id: str, format: str = 'dashboard'):
            """Get cached metrics for intersection"""
            try:
                cache_key = f"metrics:{intersection_id}:latest"
                cached_metrics = await self.metrics_cache.get(cache_key)
                
                if not cached_metrics:
                    raise HTTPException(status_code=404, detail="No metrics found")
                
                # Transform based on requested format
                transformed = self.data_transformer.transform_metrics(cached_metrics, format)
                
                return transformed
                
            except Exception as e:
                self.logger.error(f"Error getting cached metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _start_background_tasks(self):
        """Start background tasks"""
        # Cleanup task
        asyncio.create_task(self._cleanup_connections())
        
        # Cache warming task
        asyncio.create_task(self._warm_cache())
    
    async def _cleanup_connections(self):
        """Cleanup inactive connections"""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                current_time = datetime.now()
                inactive_connections = []
                
                for client_id, metadata in self.connection_manager.connection_metadata.items():
                    last_heartbeat = metadata.get('last_heartbeat', current_time)
                    if (current_time - last_heartbeat).total_seconds() > 300:  # 5 minutes
                        inactive_connections.append(client_id)
                
                for client_id in inactive_connections:
                    self.connection_manager.disconnect(client_id)
                    self.logger.info(f"Cleaned up inactive connection: {client_id}")
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
    
    async def _warm_cache(self):
        """Warm up cache with recent metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # 1 minute
                
                # This would typically fetch recent metrics and cache them
                # For now, just log the task
                self.logger.debug("Cache warming task executed")
                
            except Exception as e:
                self.logger.error(f"Error in cache warming task: {e}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8002):
        """Run the streaming API server"""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )


def create_streaming_api(config_path: str = "config/streaming_config.yaml") -> RealTimeStreamingAPI:
    """Create streaming API instance with configuration"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # Default configuration
        config = {
            'stream_config': {
                'max_connections': 1000,
                'message_buffer_size': 10000,
                'compression_threshold': 1024,
                'heartbeat_interval': 30,
                'cleanup_interval': 300,
                'max_message_age': 3600
            },
            'data_transformer': {
                'transform_rules': {}
            },
            'cache': {
                'redis_host': 'localhost',
                'redis_port': 6379,
                'redis_db': 1,
                'default_ttl': 3600,
                'max_cache_size': 10000
            }
        }
    
    return RealTimeStreamingAPI(config)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run streaming API
    streaming_api = create_streaming_api()
    streaming_api.run()
