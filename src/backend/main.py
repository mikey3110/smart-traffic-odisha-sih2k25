from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import redis
import json
from datetime import datetime
import uvicorn

app = FastAPI(
    title="Smart Traffic Management API",
    description="AI-based traffic signal optimization system for SIH 2025",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection
try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    print("✅ Redis connected successfully")
except:
    print("⚠️  Redis not available, using in-memory storage")
    redis_client = None

# In-memory storage fallback
traffic_data_store = {}
signal_data_store = {}

# Pydantic models
class TrafficData(BaseModel):
    intersection_id: str
    timestamp: int
    lane_counts: Dict[str, int]
    avg_speed: Optional[float] = None
    weather_condition: Optional[str] = None

class SignalOptimization(BaseModel):
    intersection_id: str
    optimized_timings: Dict[str, int]
    confidence_score: Optional[float] = None
    expected_improvement: Optional[float] = None

class TrafficStatus(BaseModel):
    intersection_id: str
    current_counts: Dict[str, int]
    current_timings: Dict[str, int]
    last_updated: str
    optimization_status: str

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Smart Traffic Management System API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": ["/traffic/ingest", "/traffic/status/{id}", "/signal/optimize/{id}", "/signal/status/{id}"]
    }

# Traffic data ingestion
@app.post("/traffic/ingest")
async def ingest_traffic_data(data: TrafficData):
    try:
        traffic_info = {
            "intersection_id": data.intersection_id,
            "timestamp": data.timestamp,
            "lane_counts": data.lane_counts,
            "avg_speed": data.avg_speed,
            "weather_condition": data.weather_condition,
            "ingested_at": datetime.now().isoformat()
        }
        
        # Store in Redis or memory
        if redis_client:
            redis_client.setex(f"traffic:{data.intersection_id}", 3600, json.dumps(traffic_info))
        else:
            traffic_data_store[data.intersection_id] = traffic_info
        
        print(f"📊 Traffic data ingested for {data.intersection_id}")
        return {
            "status": "success",
            "message": f"Traffic data stored for intersection {data.intersection_id}",
            "data": traffic_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest traffic data: {str(e)}")

# Get traffic status
@app.get("/traffic/status/{intersection_id}")
async def get_traffic_status(intersection_id: str):
    try:
        # Retrieve from Redis or memory
        if redis_client:
            data = redis_client.get(f"traffic:{intersection_id}")
            if data:
                traffic_info = json.loads(data)
            else:
                traffic_info = None
        else:
            traffic_info = traffic_data_store.get(intersection_id)
        
        if not traffic_info:
            # Return mock data if no real data available
            traffic_info = {
                "intersection_id": intersection_id,
                "timestamp": int(datetime.now().timestamp()),
                "lane_counts": {"north_lane": 12, "south_lane": 8, "east_lane": 15, "west_lane": 10},
                "avg_speed": 25.5,
                "weather_condition": "clear",
                "ingested_at": datetime.now().isoformat()
            }
        
        return {
            "status": "success",
            "data": traffic_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve traffic status: {str(e)}")

# Signal optimization
@app.put("/signal/optimize/{intersection_id}")
async def optimize_signal(intersection_id: str, optimization: SignalOptimization):
    try:
        signal_info = {
            "intersection_id": intersection_id,
            "optimized_timings": optimization.optimized_timings,
            "confidence_score": optimization.confidence_score,
            "expected_improvement": optimization.expected_improvement,
            "optimized_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Store in Redis or memory
        if redis_client:
            redis_client.setex(f"signal:{intersection_id}", 3600, json.dumps(signal_info))
        else:
            signal_data_store[intersection_id] = signal_info
        
        print(f"🚦 Signal optimized for {intersection_id}")
        return {
            "status": "success",
            "message": f"Signal optimization applied to intersection {intersection_id}",
            "data": signal_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize signal: {str(e)}")

# Get signal status
@app.get("/signal/status/{intersection_id}")
async def get_signal_status(intersection_id: str):
    try:
        # Retrieve from Redis or memory
        if redis_client:
            data = redis_client.get(f"signal:{intersection_id}")
            if data:
                signal_info = json.loads(data)
            else:
                signal_info = None
        else:
            signal_info = signal_data_store.get(intersection_id)
        
        if not signal_info:
            # Return default signal timings
            signal_info = {
                "intersection_id": intersection_id,
                "optimized_timings": {"north_lane": 30, "south_lane": 30, "east_lane": 30, "west_lane": 30},
                "confidence_score": 0.8,
                "expected_improvement": 15.0,
                "optimized_at": datetime.now().isoformat(),
                "status": "default"
            }
        
        return {
            "status": "success",
            "data": signal_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve signal status: {str(e)}")

# Health check
@app.get("/health")
async def health_check():
    redis_status = "connected" if redis_client else "disconnected"
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis_status": redis_status,
        "version": "1.0.0"
    }

# Get all intersections
@app.get("/intersections")
async def get_all_intersections():
    try:
        intersections = []
        
        if redis_client:
            # Get all traffic keys from Redis
            keys = redis_client.keys("traffic:*")
            for key in keys:
                intersection_id = key.split(":")[1]
                intersections.append(intersection_id)
        else:
            # Get from memory store
            intersections = list(traffic_data_store.keys())
        
        if not intersections:
            # Return default intersections
            intersections = ["junction-1", "junction-2", "junction-3"]
        
        return {
            "status": "success",
            "intersections": intersections,
            "total_count": len(intersections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve intersections: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
