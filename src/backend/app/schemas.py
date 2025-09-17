# app/schemas.py
from pydantic import BaseModel
from datetime import datetime

class TrafficData(BaseModel):
    id: int
    location: str
    vehicles: int
    timestamp: datetime
