"""
Rate limiting middleware using Redis for distributed rate limiting
"""

import time
import hashlib
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Optional
import logging

from config.settings import settings
from config.logging_config import get_logger
from services.redis_service import redis_service
from exceptions.custom_exceptions import RateLimitExceededError, create_rate_limit_error

logger = get_logger("rate_limiting")


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm
    """
    
    def __init__(
        self, 
        app, 
        requests_per_minute: int = None,
        window_size: int = None,
        skip_paths: list = None,
        key_func: Optional[Callable] = None
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute or settings.api.rate_limit_requests
        self.window_size = window_size or settings.api.rate_limit_window
        self.skip_paths = skip_paths or ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
        self.key_func = key_func or self._default_key_func
        
        # Rate limit configuration
        self.max_requests = self.requests_per_minute
        self.window_seconds = self.window_size
        
        logger.info(
            f"Rate limiting configured: {self.max_requests} requests per {self.window_seconds} seconds"
        )
    
    def _default_key_func(self, request: Request) -> str:
        """
        Default function to generate rate limit key
        Uses client IP address as the key
        """
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Create a hash of IP + path for more granular limiting
        key_data = f"{client_ip}:{request.url.path}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_rate_limit_key(self, request: Request) -> str:
        """
        Get rate limit key for the request
        """
        return f"rate_limit:{self.key_func(request)}"
    
    def _is_rate_limited(self, key: str) -> tuple[bool, int, int]:
        """
        Check if request is rate limited using sliding window
        Returns: (is_limited, current_count, reset_time)
        """
        try:
            current_time = int(time.time())
            window_start = current_time - self.window_seconds
            
            # Get current window data
            window_data = redis_service.get(key) or {}
            
            # Clean old entries
            cleaned_requests = {
                timestamp: count for timestamp, count in window_data.items()
                if int(timestamp) > window_start
            }
            
            # Count requests in current window
            current_count = sum(cleaned_requests.values())
            
            # Check if limit exceeded
            is_limited = current_count >= self.max_requests
            
            if not is_limited:
                # Add current request
                cleaned_requests[str(current_time)] = cleaned_requests.get(str(current_time), 0) + 1
                
                # Store updated data with TTL
                redis_service.set(key, cleaned_requests, ttl=self.window_seconds)
                current_count += 1
            
            # Calculate reset time (when the oldest request in window expires)
            reset_time = window_start + self.window_seconds if cleaned_requests else current_time + self.window_seconds
            
            return is_limited, current_count, reset_time
            
        except Exception as e:
            logger.error(f"Rate limiting check failed: {e}")
            # Allow request if rate limiting fails
            return False, 0, 0
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        Process request with rate limiting
        """
        # Skip rate limiting for certain paths
        if request.url.path in self.skip_paths:
            return await call_next(request)
        
        # Skip if Redis is not available
        if not redis_service.connected:
            logger.warning("Redis not available, skipping rate limiting")
            return await call_next(request)
        
        # Get rate limit key
        rate_limit_key = self._get_rate_limit_key(request)
        
        # Check rate limit
        is_limited, current_count, reset_time = self._is_rate_limited(rate_limit_key)
        
        if is_limited:
            # Calculate retry after seconds
            retry_after = max(1, reset_time - int(time.time()))
            
            logger.warning(
                f"Rate limit exceeded for {request.client.host}: {current_count}/{self.max_requests} requests",
                extra={
                    'request_id': getattr(request.state, 'request_id', 'unknown'),
                    'client_ip': request.client.host if request.client else 'unknown',
                    'path': str(request.url.path),
                    'current_count': current_count,
                    'max_requests': self.max_requests,
                    'retry_after': retry_after,
                    'event_type': 'rate_limit_exceeded'
                }
            )
            
            # Raise rate limit exception
            raise create_rate_limit_error(retry_after)
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.max_requests - current_count))
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        
        return response


class AdvancedRateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Advanced rate limiting with different limits for different endpoints
    """
    
    def __init__(self, app, rate_limits: dict = None, skip_paths: list = None):
        super().__init__(app)
        self.rate_limits = rate_limits or {
            "/traffic/ingest": {"requests": 200, "window": 60},  # High frequency data
            "/signal/optimize": {"requests": 10, "window": 60},  # Low frequency optimization
            "/traffic/status": {"requests": 100, "window": 60},  # Medium frequency status
            "default": {"requests": 50, "window": 60}  # Default limit
        }
        self.skip_paths = skip_paths or ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
        
        logger.info(f"Advanced rate limiting configured with {len(self.rate_limits)} rules")
    
    def _get_rate_limit_config(self, path: str) -> dict:
        """
        Get rate limit configuration for a specific path
        """
        # Check for exact path match
        if path in self.rate_limits:
            return self.rate_limits[path]
        
        # Check for pattern matching
        for pattern, config in self.rate_limits.items():
            if pattern != "default" and path.startswith(pattern):
                return config
        
        # Return default configuration
        return self.rate_limits.get("default", {"requests": 50, "window": 60})
    
    def _get_rate_limit_key(self, request: Request, config: dict) -> str:
        """
        Generate rate limit key with configuration
        """
        client_ip = request.client.host if request.client else "unknown"
        key_data = f"{client_ip}:{request.url.path}:{config['requests']}:{config['window']}"
        return f"rate_limit_advanced:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def _is_rate_limited(self, key: str, max_requests: int, window_seconds: int) -> tuple[bool, int, int]:
        """
        Check rate limit with specific configuration
        """
        try:
            current_time = int(time.time())
            window_start = current_time - window_seconds
            
            # Get current window data
            window_data = redis_service.get(key) or {}
            
            # Clean old entries
            cleaned_requests = {
                timestamp: count for timestamp, count in window_data.items()
                if int(timestamp) > window_start
            }
            
            # Count requests in current window
            current_count = sum(cleaned_requests.values())
            
            # Check if limit exceeded
            is_limited = current_count >= max_requests
            
            if not is_limited:
                # Add current request
                cleaned_requests[str(current_time)] = cleaned_requests.get(str(current_time), 0) + 1
                
                # Store updated data with TTL
                redis_service.set(key, cleaned_requests, ttl=window_seconds)
                current_count += 1
            
            # Calculate reset time
            reset_time = window_start + window_seconds if cleaned_requests else current_time + window_seconds
            
            return is_limited, current_count, reset_time
            
        except Exception as e:
            logger.error(f"Advanced rate limiting check failed: {e}")
            return False, 0, 0
    
    async def dispatch(self, request: Request, call_next: Callable):
        """
        Process request with advanced rate limiting
        """
        # Skip rate limiting for certain paths
        if request.url.path in self.skip_paths:
            return await call_next(request)
        
        # Skip if Redis is not available
        if not redis_service.connected:
            logger.warning("Redis not available, skipping advanced rate limiting")
            return await call_next(request)
        
        # Get rate limit configuration for this path
        config = self._get_rate_limit_config(request.url.path)
        max_requests = config["requests"]
        window_seconds = config["window"]
        
        # Get rate limit key
        rate_limit_key = self._get_rate_limit_key(request, config)
        
        # Check rate limit
        is_limited, current_count, reset_time = self._is_rate_limited(
            rate_limit_key, max_requests, window_seconds
        )
        
        if is_limited:
            retry_after = max(1, reset_time - int(time.time()))
            
            logger.warning(
                f"Advanced rate limit exceeded for {request.client.host} on {request.url.path}: {current_count}/{max_requests} requests",
                extra={
                    'request_id': getattr(request.state, 'request_id', 'unknown'),
                    'client_ip': request.client.host if request.client else 'unknown',
                    'path': str(request.url.path),
                    'current_count': current_count,
                    'max_requests': max_requests,
                    'window_seconds': window_seconds,
                    'retry_after': retry_after,
                    'event_type': 'advanced_rate_limit_exceeded'
                }
            )
            
            raise create_rate_limit_error(retry_after)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(max(0, max_requests - current_count))
        response.headers["X-RateLimit-Reset"] = str(reset_time)
        response.headers["X-RateLimit-Window"] = str(window_seconds)
        
        return response

