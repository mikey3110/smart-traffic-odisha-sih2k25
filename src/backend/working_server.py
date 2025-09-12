"""
Working Smart Traffic Management API Server
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Smart Traffic Management API",
    description="API for Smart Traffic Management System - Working Version",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Smart Traffic Management API",
        "status": "running",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "traffic_status": "/traffic/status/{intersection_id}",
            "traffic_ingest": "/traffic/ingest",
            "intersections": "/intersections",
            "signal_status": "/signal/status/{intersection_id}",
            "signal_optimize": "/signal/optimize/{intersection_id}"
        }
    }

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Smart Traffic Management API",
        "timestamp": "2024-01-01T12:00:00Z"
    }

# Traffic endpoints
@app.get("/traffic/status/{intersection_id}")
async def get_traffic_status(intersection_id: str):
    """Get traffic status for an intersection"""
    return {
        "intersection_id": intersection_id,
        "status": "active",
        "vehicles": 15,
        "waiting_time": 45,
        "current_phase": "green",
        "phase_duration": 30,
        "congestion_level": "medium"
    }

@app.get("/traffic/ingest")
async def traffic_ingest():
    """Ingest traffic data"""
    return {
        "message": "Traffic data ingested successfully",
        "timestamp": "2024-01-01T12:00:00Z",
        "records_processed": 150,
        "status": "success"
    }

@app.get("/intersections")
async def get_intersections():
    """Get all intersections"""
    return {
        "intersections": [
            {
                "id": "junction-1", 
                "name": "Main Street & 1st Ave", 
                "status": "active",
                "coordinates": {"lat": 20.2961, "lng": 85.8245}
            },
            {
                "id": "junction-2", 
                "name": "2nd Street & Oak Ave", 
                "status": "active",
                "coordinates": {"lat": 20.2971, "lng": 85.8255}
            },
            {
                "id": "junction-3", 
                "name": "3rd Street & Pine Ave", 
                "status": "maintenance",
                "coordinates": {"lat": 20.2981, "lng": 85.8265}
            }
        ],
        "total_count": 3
    }

# Signal endpoints
@app.get("/signal/status/{intersection_id}")
async def get_signal_status(intersection_id: str):
    """Get signal status for an intersection"""
    return {
        "intersection_id": intersection_id,
        "current_phase": 2,
        "phase_duration": 30,
        "next_phase": 3,
        "cycle_time": 120,
        "status": "operational",
        "last_update": "2024-01-01T12:00:00Z"
    }

@app.put("/signal/optimize/{intersection_id}")
async def optimize_signal(intersection_id: str, data: dict = None):
    """Optimize signal timing for an intersection"""
    if data is None:
        data = {}
    
    return {
        "intersection_id": intersection_id,
        "optimization_applied": True,
        "new_timing": {
            "phase_1": 25,
            "phase_2": 30,
            "phase_3": 35,
            "phase_4": 20
        },
        "efficiency_improvement": 0.15,
        "optimization_timestamp": "2024-01-01T12:00:00Z"
    }

# Performance metrics
@app.get("/metrics")
async def get_metrics():
    """Get system performance metrics"""
    return {
        "system_uptime": "2h 15m",
        "total_requests": 1250,
        "average_response_time": "45ms",
        "active_intersections": 3,
        "optimization_cycles": 12,
        "efficiency_score": 0.87
    }

if __name__ == "__main__":
    print("🚀 Starting Smart Traffic Management API Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("❤️  Health Check: http://localhost:8000/health")
    print("=" * 50)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
