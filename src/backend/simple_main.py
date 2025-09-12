"""
Simple FastAPI application for Smart Traffic Management System
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="Smart Traffic Management API",
    description="API for Smart Traffic Management System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Smart Traffic Management API"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

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
        "phase_duration": 30
    }

@app.get("/traffic/ingest")
async def traffic_ingest():
    """Ingest traffic data"""
    return {
        "message": "Traffic data ingested successfully",
        "timestamp": "2024-01-01T12:00:00Z",
        "records_processed": 150
    }

@app.get("/intersections")
async def get_intersections():
    """Get all intersections"""
    return {
        "intersections": [
            {"id": "junction-1", "name": "Main Street & 1st Ave", "status": "active"},
            {"id": "junction-2", "name": "2nd Street & Oak Ave", "status": "active"},
            {"id": "junction-3", "name": "3rd Street & Pine Ave", "status": "maintenance"}
        ]
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
        "status": "operational"
    }

@app.put("/signal/optimize/{intersection_id}")
async def optimize_signal(intersection_id: str, data: dict):
    """Optimize signal timing for an intersection"""
    return {
        "intersection_id": intersection_id,
        "optimization_applied": True,
        "new_timing": {
            "phase_1": 25,
            "phase_2": 30,
            "phase_3": 35,
            "phase_4": 20
        },
        "efficiency_improvement": 0.15
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
