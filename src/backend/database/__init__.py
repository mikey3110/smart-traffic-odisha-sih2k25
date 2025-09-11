"""
Database module for Smart Traffic Management System
"""

from .connection import (
    Base,
    create_database_engine,
    create_session_factory,
    get_database_session,
    get_database_session_context,
    init_database,
    health_check,
    close_database,
    get_db
)

from .models import (
    Intersection,
    TrafficData,
    SignalTiming,
    OptimizationResult,
    SystemEvent,
    APILog,
    HealthCheck,
    Configuration
)

__all__ = [
    # Connection management
    "Base",
    "create_database_engine",
    "create_session_factory", 
    "get_database_session",
    "get_database_session_context",
    "init_database",
    "health_check",
    "close_database",
    "get_db",
    
    # Models
    "Intersection",
    "TrafficData", 
    "SignalTiming",
    "OptimizationResult",
    "SystemEvent",
    "APILog",
    "HealthCheck",
    "Configuration"
]