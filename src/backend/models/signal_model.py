from pydantic import BaseModel
from typing import Dict, Optional

class SignalOptimization(BaseModel):
    intersection_id: str
    optimized_timings: Dict[str, int]
    confidence_score: Optional[float] = None
    expected_improvement: Optional[float] = None
    algorithm_used: Optional[str] = "ai_optimizer"

class SignalStatus(BaseModel):
    intersection_id: str
    current_timings: Dict[str, int]
    status: str
    last_optimized: str
    performance_metrics: Optional[Dict[str, float]] = None

class SignalResponse(BaseModel):
    status: str
    data: Dict
    message: Optional[str] = None
