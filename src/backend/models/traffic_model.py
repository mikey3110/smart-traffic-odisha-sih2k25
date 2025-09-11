from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime

class TrafficData(BaseModel):
    intersection_id: str
    timestamp: int
    lane_counts: Dict[str, int]
    avg_speed: Optional[float] = None
    weather_condition: Optional[str] = None
    vehicle_types: Optional[Dict[str, int]] = None
    congestion_level: Optional[str] = None

class TrafficResponse(BaseModel):
    status: str
    data: Dict
    message: Optional[str] = None

class IntersectionStats(BaseModel):
    intersection_id: str
    total_vehicles: int
    avg_wait_time: float
    throughput: float
    efficiency_score: float
    last_updated: str
