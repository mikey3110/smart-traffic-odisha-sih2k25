"""
Configuration package for Smart Traffic Management System
"""

from .settings import settings, get_settings
from .logging_config import setup_logging, get_logger, app_logger

__all__ = [
    "settings",
    "get_settings", 
    "setup_logging",
    "get_logger",
    "app_logger"
]

