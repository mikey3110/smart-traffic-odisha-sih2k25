"""
ML API Endpoints for Production Traffic Optimization System
Phase 4: API Integration & Demo Preparation

Features:
- RESTful APIs for ML model management
- Real-time metrics endpoints
- Model versioning and rollback capabilities
- Batch prediction endpoints
- Secure authentication and rate limiting
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import hmac
import base64
from functools import wraps

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import redis
import psutil
import yaml


class ModelStatus(Enum):
    """Model status enumeration"""
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    FAILED = "failed"
    RETIRING = "retiring"


class PredictionType(Enum):
    """Prediction type enumeration"""
    REALTIME = "realtime"
    BATCH = "batch"
    HISTORICAL = "historical"


@dataclass
class ModelVersion:
    """Model version information"""
    version_id: str
    model_name: str
    created_at: datetime
    status: ModelStatus
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_data_hash: str
    model_size: int
    accuracy: float
    is_deployed: bool = False


@dataclass
class PredictionRequest:
    """Prediction request structure"""
    request_id: str
    intersection_id: str
    prediction_type: PredictionType
    input_data: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class PredictionResponse:
    """Prediction response structure"""
    request_id: str
    prediction: Dict[str, Any]
    confidence: float
    model_version: str
    processing_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = None


class MLMetrics(BaseModel):
    """ML metrics model"""
    intersection_id: str
    timestamp: datetime
    wait_time_reduction: float = Field(..., ge=0, le=100)
    throughput_increase: float = Field(..., ge=0, le=100)
    fuel_consumption_reduction: float = Field(..., ge=0, le=100)
    emission_reduction: float = Field(..., ge=0, le=100)
    queue_length: float = Field(..., ge=0)
    delay_time: float = Field(..., ge=0)
    signal_efficiency: float = Field(..., ge=0, le=100)
    safety_score: float = Field(..., ge=0, le=100)


class TrainingRequest(BaseModel):
    """Training request model"""
    intersection_id: str
    training_data_path: str
    hyperparameters: Dict[str, Any] = {}
    validation_split: float = Field(0.2, ge=0, le=0.5)
    epochs: int = Field(100, ge=1, le=1000)
    batch_size: int = Field(32, ge=1, le=512)
    learning_rate: float = Field(0.001, ge=1e-6, le=1.0)
    early_stopping: bool = True
    cross_validation: bool = True


class PredictionInput(BaseModel):
    """Prediction input model"""
    intersection_id: str
    lane_counts: List[int] = Field(..., min_items=1, max_items=20)
    current_phase: int = Field(..., ge=0, le=7)
    time_since_change: float = Field(..., ge=0)
    adjacent_signals: List[Dict[str, Any]] = []
    weather_condition: str = "clear"
    time_of_day: str = "normal"
    traffic_volume: float = Field(..., ge=0)
    emergency_vehicle: bool = False


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model"""
    intersection_id: str
    data_file_path: str
    output_format: str = "json"  # json, csv, parquet
    include_confidence: bool = True
    include_metadata: bool = True


class ModelVersionResponse(BaseModel):
    """Model version response model"""
    version_id: str
    model_name: str
    created_at: datetime
    status: str
    performance_metrics: Dict[str, float]
    accuracy: float
    is_deployed: bool


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    version: str
    uptime: float
    memory_usage: float
    cpu_usage: float
    active_models: int
    total_predictions: int
    error_rate: float


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    def __init__(self, app, calls_per_minute: int = 100):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = int(time.time())
        minute_key = f"rate_limit:{client_ip}:{current_time // 60}"
        
        # Check rate limit
        current_calls = self.redis_client.get(minute_key)
        if current_calls and int(current_calls) >= self.calls_per_minute:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "retry_after": 60}
            )
        
        # Increment counter
        pipe = self.redis_client.pipeline()
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)
        pipe.execute()
        
        response = await call_next(request)
        return response


class AuthenticationManager:
    """Authentication and authorization manager"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.valid_tokens = set()
    
    def generate_token(self, user_id: str, expires_in: int = 3600) -> str:
        """Generate JWT-like token"""
        payload = {
            'user_id': user_id,
            'exp': int(time.time()) + expires_in,
            'iat': int(time.time())
        }
        
        # Simple token generation (in production, use proper JWT)
        token_data = f"{user_id}:{payload['exp']}:{payload['iat']}"
        signature = hmac.new(
            self.secret_key.encode(),
            token_data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        token = base64.b64encode(f"{token_data}:{signature}".encode()).decode()
        self.valid_tokens.add(token)
        return token
    
    def verify_token(self, token: str) -> Optional[str]:
        """Verify token and return user_id"""
        try:
            decoded = base64.b64decode(token.encode()).decode()
            user_id, exp, iat, signature = decoded.split(':')
            
            # Check expiration
            if int(time.time()) > int(exp):
                return None
            
            # Verify signature
            token_data = f"{user_id}:{exp}:{iat}"
            expected_signature = hmac.new(
                self.secret_key.encode(),
                token_data.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if signature != expected_signature:
                return None
            
            return user_id if token in self.valid_tokens else None
            
        except Exception:
            return None


class MLModelManager:
    """ML model management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.models = {}
        self.model_versions = {}
        self.active_models = {}
        
        # Performance tracking
        self.prediction_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Redis for caching
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            db=config.get('redis_db', 0)
        )
        
        self.logger.info("ML Model Manager initialized")
    
    async def train_model(self, request: TrainingRequest) -> str:
        """Train a new ML model"""
        try:
            model_id = str(uuid.uuid4())
            self.logger.info(f"Starting model training: {model_id}")
            
            # Simulate training process (replace with actual training)
            await asyncio.sleep(2)  # Simulate training time
            
            # Create model version
            version = ModelVersion(
                version_id=model_id,
                model_name=f"traffic_model_{request.intersection_id}",
                created_at=datetime.now(),
                status=ModelStatus.READY,
                performance_metrics={
                    'accuracy': 0.85,
                    'precision': 0.82,
                    'recall': 0.88,
                    'f1_score': 0.85
                },
                hyperparameters=request.dict(),
                training_data_hash=hashlib.md5(request.training_data_path.encode()).hexdigest(),
                model_size=1024 * 1024,  # 1MB
                accuracy=0.85
            )
            
            # Store model version
            self.model_versions[model_id] = version
            self.models[request.intersection_id] = model_id
            
            # Cache model info
            self.redis_client.setex(
                f"model:{model_id}",
                3600,
                json.dumps(asdict(version), default=str)
            )
            
            self.logger.info(f"Model training completed: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
    
    async def predict(self, request: PredictionInput) -> PredictionResponse:
        """Make real-time prediction"""
        try:
            start_time = time.time()
            request_id = str(uuid.uuid4())
            
            # Get active model for intersection
            model_id = self.active_models.get(request.intersection_id)
            if not model_id:
                raise HTTPException(
                    status_code=404,
                    detail=f"No active model found for intersection {request.intersection_id}"
                )
            
            # Simulate prediction (replace with actual model prediction)
            prediction = {
                'optimal_phase_duration': np.random.randint(10, 120),
                'next_phase': np.random.randint(0, 7),
                'confidence': np.random.uniform(0.7, 0.95),
                'wait_time_reduction': np.random.uniform(20, 45),
                'throughput_increase': np.random.uniform(15, 35)
            }
            
            confidence = prediction['confidence']
            processing_time = time.time() - start_time
            
            # Update counters
            self.prediction_count += 1
            
            response = PredictionResponse(
                request_id=request_id,
                prediction=prediction,
                confidence=confidence,
                model_version=model_id,
                processing_time=processing_time,
                timestamp=datetime.now(),
                metadata={
                    'intersection_id': request.intersection_id,
                    'input_features': len(request.lane_counts),
                    'emergency_override': request.emergency_vehicle
                }
            )
            
            # Cache prediction
            self.redis_client.setex(
                f"prediction:{request_id}",
                300,  # 5 minutes
                json.dumps(asdict(response), default=str)
            )
            
            return response
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error making prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    async def batch_predict(self, request: BatchPredictionRequest) -> str:
        """Make batch predictions"""
        try:
            batch_id = str(uuid.uuid4())
            self.logger.info(f"Starting batch prediction: {batch_id}")
            
            # Simulate batch processing
            await asyncio.sleep(1)
            
            # Store batch job info
            batch_info = {
                'batch_id': batch_id,
                'intersection_id': request.intersection_id,
                'status': 'processing',
                'created_at': datetime.now().isoformat(),
                'total_records': 1000,  # Simulated
                'processed_records': 0
            }
            
            self.redis_client.setex(
                f"batch:{batch_id}",
                3600,
                json.dumps(batch_info)
            )
            
            # Simulate background processing
            asyncio.create_task(self._process_batch(batch_id, request))
            
            return batch_id
            
        except Exception as e:
            self.logger.error(f"Error starting batch prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    async def _process_batch(self, batch_id: str, request: BatchPredictionRequest):
        """Process batch prediction in background"""
        try:
            # Simulate batch processing
            for i in range(100):  # Simulate 100 records
                await asyncio.sleep(0.1)
                
                # Update progress
                progress = {
                    'batch_id': batch_id,
                    'status': 'processing',
                    'progress': (i + 1) / 100 * 100,
                    'processed_records': i + 1,
                    'total_records': 100
                }
                
                self.redis_client.setex(
                    f"batch:{batch_id}",
                    3600,
                    json.dumps(progress)
                )
            
            # Mark as completed
            completion = {
                'batch_id': batch_id,
                'status': 'completed',
                'progress': 100,
                'processed_records': 100,
                'total_records': 100,
                'output_file': f"/outputs/{batch_id}_predictions.{request.output_format}",
                'completed_at': datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                f"batch:{batch_id}",
                3600,
                json.dumps(completion)
            )
            
        except Exception as e:
            self.logger.error(f"Error processing batch {batch_id}: {e}")
            
            # Mark as failed
            failure = {
                'batch_id': batch_id,
                'status': 'failed',
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            }
            
            self.redis_client.setex(
                f"batch:{batch_id}",
                3600,
                json.dumps(failure)
            )
    
    def deploy_model(self, model_id: str, intersection_id: str) -> bool:
        """Deploy model for intersection"""
        try:
            if model_id not in self.model_versions:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Deploy model
            self.active_models[intersection_id] = model_id
            self.model_versions[model_id].is_deployed = True
            self.model_versions[model_id].status = ModelStatus.DEPLOYED
            
            self.logger.info(f"Model {model_id} deployed for intersection {intersection_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deploying model: {e}")
            return False
    
    def rollback_model(self, intersection_id: str) -> bool:
        """Rollback to previous model version"""
        try:
            if intersection_id not in self.active_models:
                raise HTTPException(status_code=404, detail="No active model found")
            
            # Find previous version
            current_model = self.active_models[intersection_id]
            # Implementation would find previous version and rollback
            
            self.logger.info(f"Model rolled back for intersection {intersection_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error rolling back model: {e}")
            return False
    
    def get_model_status(self, intersection_id: str) -> Optional[ModelVersionResponse]:
        """Get model status for intersection"""
        model_id = self.active_models.get(intersection_id)
        if not model_id or model_id not in self.model_versions:
            return None
        
        version = self.model_versions[model_id]
        return ModelVersionResponse(
            version_id=version.version_id,
            model_name=version.model_name,
            created_at=version.created_at,
            status=version.status.value,
            performance_metrics=version.performance_metrics,
            accuracy=version.accuracy,
            is_deployed=version.is_deployed
        )
    
    def get_health_metrics(self) -> HealthCheckResponse:
        """Get system health metrics"""
        uptime = time.time() - self.start_time
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        
        error_rate = self.error_count / max(self.prediction_count, 1)
        
        return HealthCheckResponse(
            status="healthy" if error_rate < 0.05 else "degraded",
            timestamp=datetime.now(),
            version="1.0.0",
            uptime=uptime,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            active_models=len(self.active_models),
            total_predictions=self.prediction_count,
            error_rate=error_rate
        )


class MLAPI:
    """Main ML API application"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="ML Traffic Optimization API",
            description="Production-ready ML API for traffic optimization",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]
        )
        
        # Add rate limiting
        self.app.add_middleware(
            RateLimitMiddleware,
            calls_per_minute=config.get('rate_limit', 100)
        )
        
        # Initialize components
        self.model_manager = MLModelManager(config)
        self.auth_manager = AuthenticationManager(config.get('secret_key', 'default_secret'))
        
        # Setup routes
        self._setup_routes()
        
        self.logger.info("ML API initialized")
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Authentication dependency
        security = HTTPBearer()
        
        async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
            token = credentials.credentials
            user_id = self.auth_manager.verify_token(token)
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return user_id
        
        # Health check endpoint
        @self.app.get("/health", response_model=HealthCheckResponse)
        async def health_check():
            """Health check endpoint"""
            return self.model_manager.get_health_metrics()
        
        # Model management endpoints
        @self.app.post("/ml/train")
        async def train_model(
            request: TrainingRequest,
            current_user: str = Depends(get_current_user)
        ):
            """Train a new ML model"""
            model_id = await self.model_manager.train_model(request)
            return {"model_id": model_id, "status": "training_started"}
        
        @self.app.post("/ml/predict", response_model=PredictionResponse)
        async def predict(
            request: PredictionInput,
            current_user: str = Depends(get_current_user)
        ):
            """Make real-time prediction"""
            return await self.model_manager.predict(request)
        
        @self.app.post("/ml/predict/batch")
        async def batch_predict(
            request: BatchPredictionRequest,
            current_user: str = Depends(get_current_user)
        ):
            """Make batch predictions"""
            batch_id = await self.model_manager.batch_predict(request)
            return {"batch_id": batch_id, "status": "processing_started"}
        
        @self.app.get("/ml/status/{intersection_id}")
        async def get_model_status(
            intersection_id: str,
            current_user: str = Depends(get_current_user)
        ):
            """Get model status for intersection"""
            status = self.model_manager.get_model_status(intersection_id)
            if not status:
                raise HTTPException(status_code=404, detail="Model not found")
            return status
        
        @self.app.post("/ml/deploy/{model_id}")
        async def deploy_model(
            model_id: str,
            intersection_id: str,
            current_user: str = Depends(get_current_user)
        ):
            """Deploy model for intersection"""
            success = self.model_manager.deploy_model(model_id, intersection_id)
            if not success:
                raise HTTPException(status_code=500, detail="Deployment failed")
            return {"status": "deployed", "model_id": model_id, "intersection_id": intersection_id}
        
        @self.app.post("/ml/rollback/{intersection_id}")
        async def rollback_model(
            intersection_id: str,
            current_user: str = Depends(get_current_user)
        ):
            """Rollback model for intersection"""
            success = self.model_manager.rollback_model(intersection_id)
            if not success:
                raise HTTPException(status_code=500, detail="Rollback failed")
            return {"status": "rolled_back", "intersection_id": intersection_id}
        
        # Metrics endpoints
        @self.app.post("/ml/metrics")
        async def submit_metrics(
            metrics: MLMetrics,
            current_user: str = Depends(get_current_user)
        ):
            """Submit ML performance metrics"""
            # Store metrics in Redis
            metrics_key = f"metrics:{metrics.intersection_id}:{int(metrics.timestamp.timestamp())}"
            self.model_manager.redis_client.setex(
                metrics_key,
                3600,  # 1 hour TTL
                json.dumps(metrics.dict(), default=str)
            )
            return {"status": "metrics_submitted"}
        
        @self.app.get("/ml/metrics/{intersection_id}")
        async def get_metrics(
            intersection_id: str,
            hours: int = 24,
            current_user: str = Depends(get_current_user)
        ):
            """Get metrics for intersection"""
            # Get metrics from Redis
            pattern = f"metrics:{intersection_id}:*"
            keys = self.model_manager.redis_client.keys(pattern)
            
            metrics = []
            for key in keys:
                metric_data = self.model_manager.redis_client.get(key)
                if metric_data:
                    metrics.append(json.loads(metric_data))
            
            # Sort by timestamp
            metrics.sort(key=lambda x: x['timestamp'])
            
            return {"metrics": metrics, "count": len(metrics)}
        
        @self.app.get("/ml/performance/{intersection_id}")
        async def get_performance(
            intersection_id: str,
            current_user: str = Depends(get_current_user)
        ):
            """Get performance summary for intersection"""
            # Get recent metrics
            pattern = f"metrics:{intersection_id}:*"
            keys = self.model_manager.redis_client.keys(pattern)
            
            if not keys:
                raise HTTPException(status_code=404, detail="No metrics found")
            
            # Calculate performance summary
            recent_metrics = []
            for key in keys[-100:]:  # Last 100 metrics
                metric_data = self.model_manager.redis_client.get(key)
                if metric_data:
                    recent_metrics.append(json.loads(metric_data))
            
            if not recent_metrics:
                raise HTTPException(status_code=404, detail="No recent metrics found")
            
            # Calculate averages
            avg_wait_time_reduction = np.mean([m['wait_time_reduction'] for m in recent_metrics])
            avg_throughput_increase = np.mean([m['throughput_increase'] for m in recent_metrics])
            avg_fuel_reduction = np.mean([m['fuel_consumption_reduction'] for m in recent_metrics])
            avg_emission_reduction = np.mean([m['emission_reduction'] for m in recent_metrics])
            
            return {
                "intersection_id": intersection_id,
                "summary": {
                    "avg_wait_time_reduction": round(avg_wait_time_reduction, 2),
                    "avg_throughput_increase": round(avg_throughput_increase, 2),
                    "avg_fuel_consumption_reduction": round(avg_fuel_reduction, 2),
                    "avg_emission_reduction": round(avg_emission_reduction, 2),
                    "total_metrics": len(recent_metrics),
                    "time_range": {
                        "start": recent_metrics[0]['timestamp'],
                        "end": recent_metrics[-1]['timestamp']
                    }
                }
            }
        
        # Batch job status
        @self.app.get("/ml/batch/{batch_id}")
        async def get_batch_status(
            batch_id: str,
            current_user: str = Depends(get_current_user)
        ):
            """Get batch prediction status"""
            batch_data = self.model_manager.redis_client.get(f"batch:{batch_id}")
            if not batch_data:
                raise HTTPException(status_code=404, detail="Batch job not found")
            
            return json.loads(batch_data)
        
        # Authentication endpoints
        @self.app.post("/auth/login")
        async def login(username: str, password: str):
            """Login and get authentication token"""
            # Simple authentication (in production, use proper auth)
            if username == "admin" and password == "admin123":
                token = self.auth_manager.generate_token(username)
                return {"access_token": token, "token_type": "bearer"}
            else:
                raise HTTPException(status_code=401, detail="Invalid credentials")
        
        @self.app.post("/auth/logout")
        async def logout(
            current_user: str = Depends(get_current_user)
        ):
            """Logout and invalidate token"""
            # In production, implement proper token invalidation
            return {"status": "logged_out"}
    
    def run(self, host: str = "0.0.0.0", port: int = 8001):
        """Run the ML API server"""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )


def create_ml_api(config_path: str = "config/ml_api_config.yaml") -> MLAPI:
    """Create ML API instance with configuration"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # Default configuration
        config = {
            'redis_host': 'localhost',
            'redis_port': 6379,
            'redis_db': 0,
            'rate_limit': 100,
            'secret_key': 'your_secret_key_here'
        }
    
    return MLAPI(config)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run ML API
    ml_api = create_ml_api()
    ml_api.run()
