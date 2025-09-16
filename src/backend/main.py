"""
Main FastAPI application for Smart Traffic Management System
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.traffic import router as traffic_router
from api.signals import router as signals_router
from api.v1.ml_metrics import router as ml_metrics_router
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

# Register API routers
app.include_router(traffic_router, prefix="/traffic", tags=["traffic"])
app.include_router(signals_router, prefix="/signal", tags=["signal"])
app.include_router(ml_metrics_router, prefix="/api/v1", tags=["ml-metrics"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Smart Traffic Management API"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)