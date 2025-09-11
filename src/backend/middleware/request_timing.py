"""
Request timing middleware for performance monitoring
"""

import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import logging

from config.logging_config import get_logger, log_api_request, log_api_response

logger = get_logger("request_timing")


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track request timing and performance metrics
    """
    
    def __init__(self, app, skip_paths: list = None):
        super().__init__(app)
        self.skip_paths = skip_paths or ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and track timing
        """
        # Skip timing for certain paths
        if request.url.path in self.skip_paths:
            return await call_next(request)
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Record start time
        start_time = time.time()
        
        # Log request
        log_api_request(
            logger=logger,
            method=request.method,
            path=str(request.url.path),
            request_id=request_id,
            user_id=getattr(request.state, 'user_id', None)
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            process_time = time.time() - start_time
            
            # Add timing headers
            response.headers["X-Process-Time"] = str(round(process_time, 4))
            response.headers["X-Request-ID"] = request_id
            
            # Log response
            log_api_response(
                logger=logger,
                method=request.method,
                path=str(request.url.path),
                status_code=response.status_code,
                response_time=process_time,
                request_id=request_id
            )
            
            return response
            
        except Exception as e:
            # Calculate error response time
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path} | Time: {process_time:.3f}s | Error: {str(e)}",
                extra={
                    'request_id': request_id,
                    'method': request.method,
                    'path': str(request.url.path),
                    'response_time': process_time,
                    'error': str(e),
                    'event_type': 'api_error'
                }
            )
            
            raise