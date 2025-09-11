"""
API v1 module for Smart Traffic Management System
"""

from .traffic import router as traffic_router
from .signals import router as signals_router
from .health import router as health_router

__all__ = [
    "traffic_router",
    "signals_router", 
    "health_router"
]