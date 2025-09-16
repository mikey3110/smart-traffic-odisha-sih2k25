"""
API module for Smart Traffic Management System
"""

from .v1 import traffic_router, signals_router, health_router

__all__ = [
    "traffic_router",
    "signals_router",
    "health_router"
]



