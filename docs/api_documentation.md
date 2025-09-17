# API Documentation
## Smart Traffic Management System - ML APIs

### Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [ML API Endpoints](#ml-api-endpoints)
4. [Real-Time Streaming API](#real-time-streaming-api)
5. [Data Models](#data-models)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Examples](#examples)
9. [SDK Examples](#sdk-examples)
10. [Troubleshooting](#troubleshooting)

## Overview

The Smart Traffic Management System provides comprehensive RESTful APIs and real-time streaming endpoints for ML-based traffic optimization. The system achieves 30-45% wait time reduction with statistical significance (p < 0.05, 95% confidence).

### Base URLs
- **ML API**: `https://api.traffic-ml.com/v1`
- **Streaming API**: `https://stream.traffic-ml.com/v1`
- **WebSocket**: `wss://stream.traffic-ml.com/ws`

### API Versions
- **Current Version**: v1.0.0
- **Deprecated Versions**: None
- **Versioning Strategy**: URL-based versioning

## Authentication

### Authentication Methods

#### Bearer Token Authentication
```http
Authorization: Bearer <access_token>
```

#### API Key Authentication
```http
X-API-Key: <api_key>
```

### Getting Access Tokens

#### Login Endpoint
```http
POST /auth/login
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "refresh_token": "refresh_token_here"
}
```

#### Logout Endpoint
```http
POST /auth/logout
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "status": "logged_out",
  "message": "Successfully logged out"
}
```

## ML API Endpoints

### Model Management

#### Train Model
```http
POST /ml/train
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "intersection_id": "intersection_1",
  "training_data_path": "/data/training/traffic_data.csv",
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 500
  },
  "validation_split": 0.2,
  "cross_validation": true
}
```

**Response:**
```json
{
  "model_id": "model_12345",
  "status": "training_started",
  "estimated_duration": "2-4 hours",
  "training_url": "/ml/training/status/model_12345"
}
```

#### Get Model Status
```http
GET /ml/status/{intersection_id}
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "version_id": "model_12345",
  "model_name": "traffic_model_intersection_1",
  "created_at": "2024-01-01T12:00:00Z",
  "status": "ready",
  "performance_metrics": {
    "accuracy": 0.94,
    "precision": 0.92,
    "recall": 0.95,
    "f1_score": 0.93
  },
  "accuracy": 0.94,
  "is_deployed": true
}
```

#### Deploy Model
```http
POST /ml/deploy/{model_id}
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "intersection_id": "intersection_1"
}
```

**Response:**
```json
{
  "status": "deployed",
  "model_id": "model_12345",
  "intersection_id": "intersection_1",
  "deployment_time": "2024-01-01T12:00:00Z"
}
```

#### Rollback Model
```http
POST /ml/rollback/{intersection_id}
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "status": "rolled_back",
  "intersection_id": "intersection_1",
  "previous_model_id": "model_12344",
  "rollback_time": "2024-01-01T12:00:00Z"
}
```

### Prediction Endpoints

#### Real-Time Prediction
```http
POST /ml/predict
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "intersection_id": "intersection_1",
  "lane_counts": [5, 3, 8, 2],
  "current_phase": 2,
  "time_since_change": 45.5,
  "adjacent_signals": [
    {
      "intersection_id": "intersection_2",
      "current_phase": 1,
      "queue_length": 12
    }
  ],
  "weather_condition": "clear",
  "time_of_day": "rush_hour",
  "traffic_volume": 850.0,
  "emergency_vehicle": false
}
```

**Response:**
```json
{
  "request_id": "req_67890",
  "prediction": {
    "optimal_phase_duration": 75,
    "next_phase": 3,
    "confidence": 0.89,
    "wait_time_reduction": 38.5,
    "throughput_increase": 28.7
  },
  "confidence": 0.89,
  "model_version": "model_12345",
  "processing_time": 0.045,
  "timestamp": "2024-01-01T12:00:00Z",
  "metadata": {
    "intersection_id": "intersection_1",
    "input_features": 4,
    "emergency_override": false
  }
}
```

#### Batch Prediction
```http
POST /ml/predict/batch
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "intersection_id": "intersection_1",
  "data_file_path": "/data/batch/predictions.csv",
  "output_format": "json",
  "include_confidence": true,
  "include_metadata": true
}
```

**Response:**
```json
{
  "batch_id": "batch_54321",
  "status": "processing_started",
  "estimated_completion": "2024-01-01T12:05:00Z",
  "status_url": "/ml/batch/batch_54321"
}
```

#### Get Batch Status
```http
GET /ml/batch/{batch_id}
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "batch_id": "batch_54321",
  "status": "completed",
  "progress": 100,
  "processed_records": 1000,
  "total_records": 1000,
  "output_file": "/outputs/batch_54321_predictions.json",
  "completed_at": "2024-01-01T12:05:00Z"
}
```

### Metrics Endpoints

#### Submit Metrics
```http
POST /ml/metrics
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "intersection_id": "intersection_1",
  "timestamp": "2024-01-01T12:00:00Z",
  "wait_time_reduction": 35.2,
  "throughput_increase": 28.7,
  "fuel_consumption_reduction": 18.3,
  "emission_reduction": 16.8,
  "queue_length": 12,
  "delay_time": 45.5,
  "signal_efficiency": 85.2,
  "safety_score": 92.1
}
```

**Response:**
```json
{
  "status": "metrics_submitted",
  "intersection_id": "intersection_1",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Get Metrics
```http
GET /ml/metrics/{intersection_id}?hours=24
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "metrics": [
    {
      "intersection_id": "intersection_1",
      "timestamp": "2024-01-01T12:00:00Z",
      "wait_time_reduction": 35.2,
      "throughput_increase": 28.7,
      "fuel_consumption_reduction": 18.3,
      "emission_reduction": 16.8,
      "queue_length": 12,
      "delay_time": 45.5,
      "signal_efficiency": 85.2,
      "safety_score": 92.1
    }
  ],
  "count": 1440,
  "time_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-01T23:59:00Z"
  }
}
```

#### Get Performance Summary
```http
GET /ml/performance/{intersection_id}
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "intersection_id": "intersection_1",
  "summary": {
    "avg_wait_time_reduction": 32.5,
    "avg_throughput_increase": 25.8,
    "avg_fuel_consumption_reduction": 17.2,
    "avg_emission_reduction": 15.9,
    "total_metrics": 1440,
    "time_range": {
      "start": "2024-01-01T00:00:00Z",
      "end": "2024-01-01T23:59:00Z"
    }
  }
}
```

### Health and Status

#### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "uptime": 86400.5,
  "memory_usage": 45.2,
  "cpu_usage": 23.8,
  "active_models": 5,
  "total_predictions": 125000,
  "error_rate": 0.02
}
```

## Real-Time Streaming API

### WebSocket Endpoints

#### WebSocket Connection
```javascript
const ws = new WebSocket('wss://stream.traffic-ml.com/ws/client_123');

ws.onopen = function(event) {
    console.log('Connected to ML metrics stream');
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received metrics:', data);
};

ws.onclose = function(event) {
    console.log('Disconnected from stream');
};
```

#### WebSocket Message Format
```json
{
  "message_id": "msg_12345",
  "data_type": "metrics",
  "payload": {
    "intersection_id": "intersection_1",
    "timestamp": "2024-01-01T12:00:00Z",
    "performance": {
      "wait_time_reduction": {
        "value": 35.2,
        "trend": "up",
        "status": "excellent"
      },
      "throughput_increase": {
        "value": 28.7,
        "trend": "stable",
        "status": "good"
      }
    },
    "traffic_conditions": {
      "queue_length": 12,
      "delay_time": 45.5,
      "signal_efficiency": 85.2,
      "safety_score": 92.1
    },
    "alerts": []
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "compression": true
}
```

### Server-Sent Events

#### SSE Connection
```javascript
const eventSource = new EventSource('https://stream.traffic-ml.com/sse/client_123');

eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received SSE data:', data);
};

eventSource.onerror = function(event) {
    console.error('SSE error:', event);
};
```

#### SSE Event Format
```
id: msg_12345
event: metrics
data: {"intersection_id": "intersection_1", "wait_time_reduction": 35.2}

id: msg_12346
event: alert
data: {"type": "warning", "message": "Low performance detected"}
```

### Streaming Endpoints

#### Submit Metrics for Streaming
```http
POST /stream/metrics
Content-Type: application/json

{
  "intersection_id": "intersection_1",
  "timestamp": "2024-01-01T12:00:00Z",
  "wait_time_reduction": 35.2,
  "throughput_increase": 28.7,
  "fuel_consumption_reduction": 18.3,
  "emission_reduction": 16.8,
  "queue_length": 12,
  "delay_time": 45.5,
  "signal_efficiency": 85.2,
  "safety_score": 92.1
}
```

**Response:**
```json
{
  "status": "metrics_submitted",
  "sent_to_clients": 15,
  "message_id": "msg_12345"
}
```

#### Get Streaming Statistics
```http
GET /stream/stats
```

**Response:**
```json
{
  "connections": {
    "total_connections": 150,
    "active_connections": 15,
    "messages_sent": 125000,
    "messages_failed": 250,
    "success_rate": 0.998
  },
  "cache": {
    "hits": 45000,
    "misses": 5000,
    "hit_rate": 0.9,
    "total_requests": 50000
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Get Cached Metrics
```http
GET /stream/metrics/{intersection_id}?format=dashboard
```

**Response:**
```json
{
  "intersection_id": "intersection_1",
  "timestamp": "2024-01-01T12:00:00Z",
  "performance": {
    "wait_time_reduction": {
      "value": 35.2,
      "trend": "up",
      "status": "excellent"
    },
    "throughput_increase": {
      "value": 28.7,
      "trend": "stable",
      "status": "good"
    }
  },
  "traffic_conditions": {
    "queue_length": 12,
    "delay_time": 45.5,
    "signal_efficiency": 85.2,
    "safety_score": 92.1
  },
  "alerts": []
}
```

## Data Models

### ML Metrics Model
```json
{
  "intersection_id": "string",
  "timestamp": "datetime",
  "wait_time_reduction": "float (0-100)",
  "throughput_increase": "float (0-100)",
  "fuel_consumption_reduction": "float (0-100)",
  "emission_reduction": "float (0-100)",
  "queue_length": "float (>=0)",
  "delay_time": "float (>=0)",
  "signal_efficiency": "float (0-100)",
  "safety_score": "float (0-100)"
}
```

### Prediction Input Model
```json
{
  "intersection_id": "string",
  "lane_counts": "array of integers (1-20 items)",
  "current_phase": "integer (0-7)",
  "time_since_change": "float (>=0)",
  "adjacent_signals": "array of objects",
  "weather_condition": "string",
  "time_of_day": "string",
  "traffic_volume": "float (>=0)",
  "emergency_vehicle": "boolean"
}
```

### Prediction Response Model
```json
{
  "request_id": "string",
  "prediction": {
    "optimal_phase_duration": "integer",
    "next_phase": "integer",
    "confidence": "float (0-1)",
    "wait_time_reduction": "float",
    "throughput_increase": "float"
  },
  "confidence": "float (0-1)",
  "model_version": "string",
  "processing_time": "float",
  "timestamp": "datetime",
  "metadata": "object"
}
```

### Training Request Model
```json
{
  "intersection_id": "string",
  "training_data_path": "string",
  "hyperparameters": "object",
  "validation_split": "float (0-0.5)",
  "epochs": "integer (1-1000)",
  "batch_size": "integer (1-512)",
  "learning_rate": "float (1e-6 to 1.0)",
  "early_stopping": "boolean",
  "cross_validation": "boolean"
}
```

## Error Handling

### HTTP Status Codes

#### Success Codes
- `200 OK` - Request successful
- `201 Created` - Resource created successfully
- `202 Accepted` - Request accepted for processing

#### Client Error Codes
- `400 Bad Request` - Invalid request data
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Access denied
- `404 Not Found` - Resource not found
- `409 Conflict` - Resource conflict
- `422 Unprocessable Entity` - Validation error
- `429 Too Many Requests` - Rate limit exceeded

#### Server Error Codes
- `500 Internal Server Error` - Server error
- `502 Bad Gateway` - Gateway error
- `503 Service Unavailable` - Service unavailable
- `504 Gateway Timeout` - Gateway timeout

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": {
      "field": "learning_rate",
      "issue": "Value must be between 0.0001 and 1.0"
    },
    "timestamp": "2024-01-01T12:00:00Z",
    "request_id": "req_12345"
  }
}
```

### Common Error Codes

#### Validation Errors
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "field": "intersection_id",
      "issue": "Field is required"
    }
  }
}
```

#### Authentication Errors
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or expired token",
    "details": {
      "action": "refresh_token"
    }
  }
}
```

#### Rate Limit Errors
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 100,
      "window": "1 minute",
      "retry_after": 60
    }
  }
}
```

## Rate Limiting

### Rate Limits
- **API Requests**: 100 requests per minute per client
- **WebSocket Connections**: 10 connections per client
- **Batch Predictions**: 5 concurrent jobs per client
- **Metrics Submission**: 1000 submissions per minute per client

### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
X-RateLimit-Window: 60
```

### Rate Limit Exceeded Response
```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1640995200
Retry-After: 60

{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "retry_after": 60
  }
}
```

## Examples

### Python Examples

#### Basic API Client
```python
import requests
import json

class TrafficMLClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def predict(self, intersection_id, lane_counts, current_phase, 
                time_since_change, **kwargs):
        data = {
            'intersection_id': intersection_id,
            'lane_counts': lane_counts,
            'current_phase': current_phase,
            'time_since_change': time_since_change,
            **kwargs
        }
        
        response = requests.post(
            f'{self.base_url}/ml/predict',
            headers=self.headers,
            json=data
        )
        
        return response.json()
    
    def submit_metrics(self, intersection_id, metrics):
        data = {
            'intersection_id': intersection_id,
            **metrics
        }
        
        response = requests.post(
            f'{self.base_url}/ml/metrics',
            headers=self.headers,
            json=data
        )
        
        return response.json()

# Usage
client = TrafficMLClient('https://api.traffic-ml.com/v1', 'your_api_key')

# Make prediction
prediction = client.predict(
    intersection_id='intersection_1',
    lane_counts=[5, 3, 8, 2],
    current_phase=2,
    time_since_change=45.5,
    weather_condition='clear',
    traffic_volume=850.0
)

print(f"Optimal phase duration: {prediction['prediction']['optimal_phase_duration']}")

# Submit metrics
metrics = {
    'wait_time_reduction': 35.2,
    'throughput_increase': 28.7,
    'fuel_consumption_reduction': 18.3
}

result = client.submit_metrics('intersection_1', metrics)
print(f"Metrics submitted: {result['status']}")
```

#### WebSocket Client
```python
import asyncio
import websockets
import json

async def ml_metrics_client():
    uri = "wss://stream.traffic-ml.com/ws/client_123"
    
    async with websockets.connect(uri) as websocket:
        print("Connected to ML metrics stream")
        
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data['data_type'] == 'metrics':
                    metrics = data['payload']
                    print(f"Wait time reduction: {metrics['performance']['wait_time_reduction']['value']}%")
                
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break
            except Exception as e:
                print(f"Error: {e}")

# Run client
asyncio.run(ml_metrics_client())
```

### JavaScript Examples

#### WebSocket Client
```javascript
class MLMetricsClient {
    constructor(clientId) {
        this.clientId = clientId;
        this.ws = null;
        this.reconnectInterval = 5000;
    }
    
    connect() {
        this.ws = new WebSocket(`wss://stream.traffic-ml.com/ws/${this.clientId}`);
        
        this.ws.onopen = () => {
            console.log('Connected to ML metrics stream');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onclose = () => {
            console.log('Connection closed, reconnecting...');
            setTimeout(() => this.connect(), this.reconnectInterval);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    handleMessage(data) {
        if (data.data_type === 'metrics') {
            const metrics = data.payload;
            this.updateDashboard(metrics);
        } else if (data.data_type === 'alert') {
            this.showAlert(data.payload);
        }
    }
    
    updateDashboard(metrics) {
        // Update dashboard with metrics
        document.getElementById('wait-time-reduction').textContent = 
            `${metrics.performance.wait_time_reduction.value}%`;
        document.getElementById('throughput-increase').textContent = 
            `${metrics.performance.throughput_increase.value}%`;
    }
    
    showAlert(alert) {
        console.warn(`Alert: ${alert.message}`);
        // Show alert in UI
    }
}

// Usage
const client = new MLMetricsClient('client_123');
client.connect();
```

#### API Client
```javascript
class TrafficMLAPI {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json'
        };
    }
    
    async predict(intersectionId, predictionData) {
        const response = await fetch(`${this.baseUrl}/ml/predict`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                intersection_id: intersectionId,
                ...predictionData
            })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async submitMetrics(intersectionId, metrics) {
        const response = await fetch(`${this.baseUrl}/ml/metrics`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                intersection_id: intersectionId,
                ...metrics
            })
        });
        
        return await response.json();
    }
}

// Usage
const api = new TrafficMLAPI('https://api.traffic-ml.com/v1', 'your_api_key');

// Make prediction
api.predict('intersection_1', {
    lane_counts: [5, 3, 8, 2],
    current_phase: 2,
    time_since_change: 45.5,
    weather_condition: 'clear',
    traffic_volume: 850.0
}).then(prediction => {
    console.log('Prediction:', prediction);
}).catch(error => {
    console.error('Error:', error);
});
```

## SDK Examples

### Python SDK
```python
# Install SDK
# pip install traffic-ml-sdk

from traffic_ml import TrafficMLClient, MLMetricsClient

# Initialize client
client = TrafficMLClient(
    api_url='https://api.traffic-ml.com/v1',
    api_key='your_api_key'
)

# Make prediction
prediction = client.predict(
    intersection_id='intersection_1',
    lane_counts=[5, 3, 8, 2],
    current_phase=2,
    time_since_change=45.5
)

# Submit metrics
client.submit_metrics('intersection_1', {
    'wait_time_reduction': 35.2,
    'throughput_increase': 28.7
})

# WebSocket client
def on_metrics(metrics):
    print(f"Wait time reduction: {metrics['wait_time_reduction']}%")

metrics_client = MLMetricsClient(
    websocket_url='wss://stream.traffic-ml.com/ws',
    client_id='client_123',
    on_metrics=on_metrics
)

metrics_client.connect()
```

### JavaScript SDK
```javascript
// Install SDK
// npm install traffic-ml-sdk

import { TrafficMLClient, MLMetricsClient } from 'traffic-ml-sdk';

// Initialize client
const client = new TrafficMLClient({
    apiUrl: 'https://api.traffic-ml.com/v1',
    apiKey: 'your_api_key'
});

// Make prediction
const prediction = await client.predict('intersection_1', {
    lane_counts: [5, 3, 8, 2],
    current_phase: 2,
    time_since_change: 45.5
});

// Submit metrics
await client.submitMetrics('intersection_1', {
    wait_time_reduction: 35.2,
    throughput_increase: 28.7
});

// WebSocket client
const metricsClient = new MLMetricsClient({
    websocketUrl: 'wss://stream.traffic-ml.com/ws',
    clientId: 'client_123',
    onMetrics: (metrics) => {
        console.log(`Wait time reduction: ${metrics.wait_time_reduction}%`);
    }
});

metricsClient.connect();
```

## Troubleshooting

### Common Issues

#### Connection Issues
```yaml
issue: "Cannot connect to API"
symptoms:
  - "Connection timeout"
  - "DNS resolution failed"
  - "SSL certificate error"
solutions:
  - "Check network connectivity"
  - "Verify API URL"
  - "Check SSL certificate"
  - "Contact support"
```

#### Authentication Issues
```yaml
issue: "Authentication failed"
symptoms:
  - "401 Unauthorized"
  - "Invalid token"
  - "Token expired"
solutions:
  - "Check API key/token"
  - "Refresh token"
  - "Verify permissions"
  - "Contact admin"
```

#### Rate Limiting Issues
```yaml
issue: "Rate limit exceeded"
symptoms:
  - "429 Too Many Requests"
  - "Request throttled"
solutions:
  - "Implement exponential backoff"
  - "Reduce request frequency"
  - "Use batch endpoints"
  - "Contact support for higher limits"
```

#### Prediction Issues
```yaml
issue: "Prediction errors"
symptoms:
  - "Invalid input data"
  - "Model not found"
  - "Low confidence scores"
solutions:
  - "Validate input data"
  - "Check model status"
  - "Verify intersection ID"
  - "Check data quality"
```

### Debug Mode

#### Enable Debug Logging
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Use debug mode in client
client = TrafficMLClient(
    api_url='https://api.traffic-ml.com/v1',
    api_key='your_api_key',
    debug=True
)
```

#### Debug Headers
```http
X-Debug: true
X-Request-ID: req_12345
X-Client-Version: 1.0.0
```

### Support

#### Getting Help
- **Documentation**: https://docs.traffic-ml.com
- **API Reference**: https://api.traffic-ml.com/docs
- **Support Email**: support@traffic-ml.com
- **Status Page**: https://status.traffic-ml.com

#### Reporting Issues
```json
{
  "issue_type": "bug|feature_request|question",
  "description": "Detailed description of the issue",
  "steps_to_reproduce": ["Step 1", "Step 2"],
  "expected_behavior": "What should happen",
  "actual_behavior": "What actually happens",
  "environment": {
    "client_version": "1.0.0",
    "api_version": "v1.0.0",
    "browser": "Chrome 91.0",
    "os": "Windows 10"
  },
  "logs": "Relevant log entries",
  "screenshots": "Screenshot URLs"
}
```

---

**Document Version**: 1.0  
**Last Updated**: January 2024  
**Maintained By**: API Engineering Team  
**Review Cycle**: Monthly
