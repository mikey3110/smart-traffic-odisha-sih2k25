#!/usr/bin/env python3
"""
Data Flow Manager for Smart Traffic Management System
Handles real-time data pipeline and event-driven architecture
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import redis
import pika
from kafka import KafkaProducer, KafkaConsumer
import websockets
import requests
from pathlib import Path

class DataSource(Enum):
    SUMO_SIMULATION = "sumo_simulation"
    ML_OPTIMIZER = "ml_optimizer"
    BACKEND_API = "backend_api"
    FRONTEND = "frontend"
    EXTERNAL_API = "external_api"

class DataType(Enum):
    TRAFFIC_LIGHT_DATA = "traffic_light_data"
    VEHICLE_DATA = "vehicle_data"
    INTERSECTION_DATA = "intersection_data"
    PERFORMANCE_METRICS = "performance_metrics"
    OPTIMIZATION_RESULTS = "optimization_results"
    ALERTS = "alerts"
    COMMANDS = "commands"

@dataclass
class DataMessage:
    id: str
    source: DataSource
    data_type: DataType
    payload: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    ttl: Optional[int] = None  # Time to live in seconds
    correlation_id: Optional[str] = None

@dataclass
class DataFlowConfig:
    source: DataSource
    target: DataSource
    data_types: List[DataType]
    enabled: bool = True
    batch_size: int = 100
    batch_timeout: int = 5  # seconds
    retry_attempts: int = 3
    retry_delay: int = 1  # seconds
    filters: Dict[str, Any] = field(default_factory=dict)
    transformers: List[str] = field(default_factory=list)

class DataFlowManager:
    """Manages data flow between system components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # Message queues
        self.redis_client = None
        self.rabbitmq_connection = None
        self.kafka_producer = None
        self.kafka_consumer = None
        
        # WebSocket connections
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        
        # Data flow configurations
        self.data_flows: List[DataFlowConfig] = []
        self.message_handlers: Dict[DataType, List[Callable]] = {}
        
        # Metrics
        self.metrics = {
            "messages_processed": 0,
            "messages_failed": 0,
            "messages_dropped": 0,
            "processing_time": 0,
            "queue_sizes": {}
        }
        
        self._initialize_connections()
        self._setup_data_flows()
    
    def _initialize_connections(self):
        """Initialize message queue connections"""
        try:
            # Redis connection
            redis_config = self.config.get("message_queue", {})
            if redis_config.get("type") == "redis":
                self.redis_client = redis.Redis(
                    host=redis_config.get("host", "localhost"),
                    port=redis_config.get("port", 6379),
                    password=redis_config.get("password", ""),
                    decode_responses=True
                )
                self.logger.info("Redis connection established")
            
            # RabbitMQ connection
            elif redis_config.get("type") == "rabbitmq":
                credentials = pika.PlainCredentials(
                    redis_config.get("username", "guest"),
                    redis_config.get("password", "guest")
                )
                self.rabbitmq_connection = pika.BlockingConnection(
                    pika.ConnectionParameters(
                        host=redis_config.get("host", "localhost"),
                        port=redis_config.get("port", 5672),
                        credentials=credentials
                    )
                )
                self.logger.info("RabbitMQ connection established")
            
            # Kafka connection
            elif redis_config.get("type") == "kafka":
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=[f"{redis_config.get('host', 'localhost')}:{redis_config.get('port', 9092)}"],
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
                self.kafka_consumer = KafkaConsumer(
                    'traffic_data',
                    bootstrap_servers=[f"{redis_config.get('host', 'localhost')}:{redis_config.get('port', 9092)}"],
                    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
                )
                self.logger.info("Kafka connection established")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize message queue connections: {e}")
    
    def _setup_data_flows(self):
        """Setup data flow configurations"""
        # SUMO to Backend data flow
        self.data_flows.append(DataFlowConfig(
            source=DataSource.SUMO_SIMULATION,
            target=DataSource.BACKEND_API,
            data_types=[DataType.TRAFFIC_LIGHT_DATA, DataType.VEHICLE_DATA, DataType.INTERSECTION_DATA],
            batch_size=50,
            batch_timeout=2
        ))
        
        # Backend to Frontend data flow
        self.data_flows.append(DataFlowConfig(
            source=DataSource.BACKEND_API,
            target=DataSource.FRONTEND,
            data_types=[DataType.TRAFFIC_LIGHT_DATA, DataType.PERFORMANCE_METRICS, DataType.ALERTS],
            batch_size=20,
            batch_timeout=1
        ))
        
        # ML Optimizer to Backend data flow
        self.data_flows.append(DataFlowConfig(
            source=DataSource.ML_OPTIMIZER,
            target=DataSource.BACKEND_API,
            data_types=[DataType.OPTIMIZATION_RESULTS],
            batch_size=10,
            batch_timeout=5
        ))
        
        # Backend to ML Optimizer data flow
        self.data_flows.append(DataFlowConfig(
            source=DataSource.BACKEND_API,
            target=DataSource.ML_OPTIMIZER,
            data_types=[DataType.TRAFFIC_LIGHT_DATA, DataType.VEHICLE_DATA],
            batch_size=100,
            batch_timeout=10
        ))
    
    async def start(self):
        """Start the data flow manager"""
        self.logger.info("Starting Data Flow Manager...")
        self.running = True
        
        # Start data flow tasks
        tasks = [
            asyncio.create_task(self._process_message_queue()),
            asyncio.create_task(self._handle_websocket_connections()),
            asyncio.create_task(self._monitor_queue_sizes()),
            asyncio.create_task(self._cleanup_expired_messages())
        ]
        
        # Start data flow coordination for each flow
        for flow in self.data_flows:
            if flow.enabled:
                tasks.append(asyncio.create_task(self._coordinate_data_flow(flow)))
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Data flow manager error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the data flow manager"""
        self.logger.info("Stopping Data Flow Manager...")
        self.running = False
        
        # Close connections
        if self.redis_client:
            self.redis_client.close()
        if self.rabbitmq_connection:
            self.rabbitmq_connection.close()
        if self.kafka_producer:
            self.kafka_producer.close()
        if self.kafka_consumer:
            self.kafka_consumer.close()
    
    async def _coordinate_data_flow(self, flow: DataFlowConfig):
        """Coordinate data flow between source and target"""
        self.logger.info(f"Starting data flow: {flow.source.value} -> {flow.target.value}")
        
        while self.running:
            try:
                # Collect messages from source
                messages = await self._collect_messages(flow)
                
                if messages:
                    # Process messages in batches
                    await self._process_message_batch(flow, messages)
                
                await asyncio.sleep(flow.batch_timeout)
                
            except Exception as e:
                self.logger.error(f"Data flow error {flow.source.value} -> {flow.target.value}: {e}")
                await asyncio.sleep(5)
    
    async def _collect_messages(self, flow: DataFlowConfig) -> List[DataMessage]:
        """Collect messages from source component"""
        messages = []
        
        try:
            if self.redis_client:
                # Collect from Redis queues
                for data_type in flow.data_types:
                    queue_name = f"{flow.source.value}:{data_type.value}"
                    while len(messages) < flow.batch_size:
                        message_data = self.redis_client.lpop(queue_name)
                        if not message_data:
                            break
                        
                        message = json.loads(message_data)
                        messages.append(DataMessage(**message))
            
            elif self.kafka_consumer:
                # Collect from Kafka topics
                messages = self.kafka_consumer.poll(timeout_ms=1000)
                # Process Kafka messages...
            
        except Exception as e:
            self.logger.error(f"Failed to collect messages from {flow.source.value}: {e}")
        
        return messages
    
    async def _process_message_batch(self, flow: DataFlowConfig, messages: List[DataMessage]):
        """Process a batch of messages"""
        if not messages:
            return
        
        start_time = time.time()
        
        try:
            # Apply filters
            filtered_messages = self._apply_filters(messages, flow.filters)
            
            # Apply transformers
            transformed_messages = await self._apply_transformers(filtered_messages, flow.transformers)
            
            # Send to target
            await self._send_messages(flow.target, transformed_messages)
            
            # Update metrics
            self.metrics["messages_processed"] += len(transformed_messages)
            self.metrics["processing_time"] += time.time() - start_time
            
        except Exception as e:
            self.logger.error(f"Failed to process message batch: {e}")
            self.metrics["messages_failed"] += len(messages)
    
    def _apply_filters(self, messages: List[DataMessage], filters: Dict[str, Any]) -> List[DataMessage]:
        """Apply filters to messages"""
        filtered = messages
        
        for filter_key, filter_value in filters.items():
            if filter_key == "priority_min":
                filtered = [m for m in filtered if m.priority >= filter_value]
            elif filter_key == "data_type":
                filtered = [m for m in filtered if m.data_type.value == filter_value]
            elif filter_key == "source":
                filtered = [m for m in filtered if m.source.value == filter_value]
        
        return filtered
    
    async def _apply_transformers(self, messages: List[DataMessage], transformers: List[str]) -> List[DataMessage]:
        """Apply transformers to messages"""
        transformed = messages
        
        for transformer_name in transformers:
            if transformer_name == "aggregate_metrics":
                transformed = self._aggregate_metrics(transformed)
            elif transformer_name == "compress_data":
                transformed = self._compress_data(transformed)
            elif transformer_name == "add_timestamp":
                transformed = self._add_timestamp(transformed)
        
        return transformed
    
    def _aggregate_metrics(self, messages: List[DataMessage]) -> List[DataMessage]:
        """Aggregate performance metrics"""
        if not messages:
            return messages
        
        # Group by data type and time window
        grouped = {}
        for message in messages:
            if message.data_type == DataType.PERFORMANCE_METRICS:
                key = f"{message.data_type.value}:{message.timestamp.strftime('%Y-%m-%d %H:%M')}"
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(message)
        
        # Aggregate metrics
        aggregated = []
        for key, group in grouped.items():
            if len(group) > 1:
                # Calculate aggregated values
                payload = group[0].payload.copy()
                for metric in ['total_vehicles', 'running_vehicles', 'waiting_vehicles']:
                    if metric in payload:
                        payload[metric] = sum(m.payload.get(metric, 0) for m in group)
                
                aggregated.append(DataMessage(
                    id=f"agg_{group[0].id}",
                    source=group[0].source,
                    data_type=group[0].data_type,
                    payload=payload,
                    timestamp=group[0].timestamp,
                    priority=group[0].priority
                ))
            else:
                aggregated.extend(group)
        
        return aggregated
    
    def _compress_data(self, messages: List[DataMessage]) -> List[DataMessage]:
        """Compress message data"""
        # Simple compression by removing redundant fields
        for message in messages:
            if 'redundant_field' in message.payload:
                del message.payload['redundant_field']
        return messages
    
    def _add_timestamp(self, messages: List[DataMessage]) -> List[DataMessage]:
        """Add processing timestamp to messages"""
        for message in messages:
            message.payload['processing_timestamp'] = datetime.now().isoformat()
        return messages
    
    async def _send_messages(self, target: DataSource, messages: List[DataMessage]):
        """Send messages to target component"""
        if not messages:
            return
        
        try:
            if target == DataSource.BACKEND_API:
                await self._send_to_backend(messages)
            elif target == DataSource.FRONTEND:
                await self._send_to_frontend(messages)
            elif target == DataSource.ML_OPTIMIZER:
                await self._send_to_ml_optimizer(messages)
            elif target == DataSource.SUMO_SIMULATION:
                await self._send_to_sumo(messages)
        
        except Exception as e:
            self.logger.error(f"Failed to send messages to {target.value}: {e}")
            self.metrics["messages_failed"] += len(messages)
    
    async def _send_to_backend(self, messages: List[DataMessage]):
        """Send messages to backend API"""
        for message in messages:
            try:
                url = f"http://localhost:8000/api/data/{message.data_type.value}"
                response = requests.post(url, json=message.payload, timeout=5)
                if response.status_code != 200:
                    self.logger.warning(f"Backend API error: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Failed to send to backend: {e}")
    
    async def _send_to_frontend(self, messages: List[DataMessage]):
        """Send messages to frontend via WebSocket"""
        for connection in self.websocket_connections.values():
            try:
                for message in messages:
                    await connection.send(json.dumps({
                        "type": message.data_type.value,
                        "data": message.payload,
                        "timestamp": message.timestamp.isoformat()
                    }))
            except Exception as e:
                self.logger.error(f"Failed to send to frontend: {e}")
    
    async def _send_to_ml_optimizer(self, messages: List[DataMessage]):
        """Send messages to ML optimizer"""
        for message in messages:
            try:
                url = f"http://localhost:8001/api/data/{message.data_type.value}"
                response = requests.post(url, json=message.payload, timeout=5)
                if response.status_code != 200:
                    self.logger.warning(f"ML Optimizer API error: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Failed to send to ML optimizer: {e}")
    
    async def _send_to_sumo(self, messages: List[DataMessage]):
        """Send messages to SUMO simulation"""
        for message in messages:
            try:
                url = f"http://localhost:8002/api/control/{message.data_type.value}"
                response = requests.post(url, json=message.payload, timeout=5)
                if response.status_code != 200:
                    self.logger.warning(f"SUMO API error: {response.status_code}")
            except Exception as e:
                self.logger.error(f"Failed to send to SUMO: {e}")
    
    async def _process_message_queue(self):
        """Process incoming messages from message queue"""
        while self.running:
            try:
                if self.redis_client:
                    # Process Redis messages
                    for data_type in DataType:
                        queue_name = f"incoming:{data_type.value}"
                        message_data = self.redis_client.lpop(queue_name)
                        if message_data:
                            message = json.loads(message_data)
                            await self._handle_message(DataMessage(**message))
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Message queue processing error: {e}")
                await asyncio.sleep(1)
    
    async def _handle_message(self, message: DataMessage):
        """Handle incoming message"""
        try:
            # Check TTL
            if message.ttl and (datetime.now() - message.timestamp).seconds > message.ttl:
                self.metrics["messages_dropped"] += 1
                return
            
            # Call registered handlers
            handlers = self.message_handlers.get(message.data_type, [])
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    self.logger.error(f"Message handler error: {e}")
            
            self.metrics["messages_processed"] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to handle message: {e}")
            self.metrics["messages_failed"] += 1
    
    async def _handle_websocket_connections(self):
        """Handle WebSocket connections for real-time data"""
        async def websocket_handler(websocket, path):
            self.websocket_connections[str(websocket)] = websocket
            try:
                async for message in websocket:
                    # Handle incoming WebSocket messages
                    data = json.loads(message)
                    await self._handle_websocket_message(data, websocket)
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                if str(websocket) in self.websocket_connections:
                    del self.websocket_connections[str(websocket)]
        
        # Start WebSocket server
        start_server = websockets.serve(websocket_handler, "localhost", 8765)
        await start_server
    
    async def _handle_websocket_message(self, data: Dict[str, Any], websocket):
        """Handle incoming WebSocket message"""
        try:
            message_type = data.get("type")
            if message_type == "subscribe":
                # Handle subscription to data types
                data_types = data.get("data_types", [])
                # Implementation for subscription handling
            elif message_type == "command":
                # Handle commands from frontend
                command = data.get("command")
                await self._handle_command(command, data.get("payload", {}))
        
        except Exception as e:
            self.logger.error(f"WebSocket message handling error: {e}")
    
    async def _handle_command(self, command: str, payload: Dict[str, Any]):
        """Handle commands from frontend"""
        try:
            if command == "control_traffic_light":
                # Send traffic light control command
                message = DataMessage(
                    id=f"cmd_{int(time.time())}",
                    source=DataSource.FRONTEND,
                    data_type=DataType.COMMANDS,
                    payload={"command": command, **payload},
                    timestamp=datetime.now(),
                    priority=3
                )
                await self._send_to_backend([message])
            
            elif command == "trigger_optimization":
                # Trigger ML optimization
                message = DataMessage(
                    id=f"opt_{int(time.time())}",
                    source=DataSource.FRONTEND,
                    data_type=DataType.COMMANDS,
                    payload={"command": command, **payload},
                    timestamp=datetime.now(),
                    priority=2
                )
                await self._send_to_ml_optimizer([message])
        
        except Exception as e:
            self.logger.error(f"Command handling error: {e}")
    
    async def _monitor_queue_sizes(self):
        """Monitor queue sizes for alerting"""
        while self.running:
            try:
                if self.redis_client:
                    for data_type in DataType:
                        queue_name = f"incoming:{data_type.value}"
                        size = self.redis_client.llen(queue_name)
                        self.metrics["queue_sizes"][data_type.value] = size
                        
                        # Alert if queue size is too large
                        if size > 1000:
                            self.logger.warning(f"Queue {data_type.value} size is large: {size}")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Queue monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_messages(self):
        """Clean up expired messages"""
        while self.running:
            try:
                # Implementation for cleaning up expired messages
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60)
    
    def register_message_handler(self, data_type: DataType, handler: Callable):
        """Register a message handler for a specific data type"""
        if data_type not in self.message_handlers:
            self.message_handlers[data_type] = []
        self.message_handlers[data_type].append(handler)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get data flow metrics"""
        return {
            "metrics": self.metrics,
            "active_flows": len([f for f in self.data_flows if f.enabled]),
            "websocket_connections": len(self.websocket_connections),
            "message_handlers": {dt.value: len(handlers) for dt, handlers in self.message_handlers.items()}
        }

# Example usage
async def main():
    config = {
        "message_queue": {
            "type": "redis",
            "host": "localhost",
            "port": 6379
        }
    }
    
    manager = DataFlowManager(config)
    await manager.start()

if __name__ == "__main__":
    asyncio.run(main())
