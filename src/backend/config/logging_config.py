"""
Comprehensive logging configuration for the Smart Traffic Management System
Provides structured logging with different handlers for different log levels
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from config.settings import settings


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class TrafficLogFilter(logging.Filter):
    """Custom filter to add context information to log records"""
    
    def filter(self, record):
        # Add request ID if available
        if hasattr(record, 'request_id'):
            record.request_id = getattr(record, 'request_id', 'N/A')
        else:
            record.request_id = 'N/A'
        
        # Add intersection ID if available
        if hasattr(record, 'intersection_id'):
            record.intersection_id = getattr(record, 'intersection_id', 'N/A')
        else:
            record.intersection_id = 'N/A'
        
        return True


def setup_logging() -> logging.Logger:
    """
    Set up comprehensive logging configuration
    Returns the main application logger
    """
    # Create main logger
    logger = logging.getLogger("traffic_management")
    logger.setLevel(getattr(logging, settings.logging.level))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(request_id)s | %(intersection_id)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    simple_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    colored_formatter = ColoredFormatter(
        fmt="%(asctime)s | %(levelname)s | %(request_id)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Console handler
    if settings.logging.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(colored_formatter)
        console_handler.addFilter(TrafficLogFilter())
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if settings.logging.enable_file and settings.logging.file_path:
        # Ensure log directory exists
        log_path = Path(settings.logging.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=settings.logging.file_path,
            maxBytes=settings.logging.max_file_size,
            backupCount=settings.logging.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        file_handler.addFilter(TrafficLogFilter())
        logger.addHandler(file_handler)
    
    # Error file handler (separate file for errors)
    if settings.logging.enable_file and settings.logging.file_path:
        error_log_path = Path(settings.logging.file_path).parent / "traffic_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            filename=str(error_log_path),
            maxBytes=settings.logging.max_file_size,
            backupCount=settings.logging.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        error_handler.addFilter(TrafficLogFilter())
        logger.addHandler(error_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the traffic management configuration
    """
    return logging.getLogger(f"traffic_management.{name}")


def log_api_request(logger: logging.Logger, method: str, path: str, 
                   request_id: str, user_id: Optional[str] = None):
    """Log API request details"""
    logger.info(
        f"API Request: {method} {path}",
        extra={
            'request_id': request_id,
            'user_id': user_id or 'anonymous',
            'event_type': 'api_request'
        }
    )


def log_api_response(logger: logging.Logger, method: str, path: str, 
                    status_code: int, response_time: float, request_id: str):
    """Log API response details"""
    level = logging.INFO if status_code < 400 else logging.WARNING
    logger.log(
        level,
        f"API Response: {method} {path} | Status: {status_code} | Time: {response_time:.3f}s",
        extra={
            'request_id': request_id,
            'status_code': status_code,
            'response_time': response_time,
            'event_type': 'api_response'
        }
    )


def log_traffic_event(logger: logging.Logger, event_type: str, intersection_id: str, 
                     details: dict, request_id: Optional[str] = None):
    """Log traffic-related events"""
    logger.info(
        f"Traffic Event: {event_type}",
        extra={
            'request_id': request_id or 'system',
            'intersection_id': intersection_id,
            'event_type': event_type,
            'details': details
        }
    )


def log_system_event(logger: logging.Logger, event_type: str, message: str, 
                    level: int = logging.INFO, **kwargs):
    """Log system events"""
    logger.log(
        level,
        f"System Event: {message}",
        extra={
            'event_type': event_type,
            'system_component': kwargs.get('component', 'unknown'),
            **kwargs
        }
    )


# Initialize logging
main_logger = setup_logging()

# Log startup
main_logger.info("Logging system initialized", extra={'event_type': 'system_startup'})