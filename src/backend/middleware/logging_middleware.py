"""
Comprehensive logging middleware for request/response tracking
"""

import time
import json
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import logging

from config.logging_config import get_logger, log_api_request, log_api_response

logger = get_logger("logging_middleware")


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive logging middleware for API requests and responses
    """
    
    def __init__(
        self, 
        app, 
        log_requests: bool = True,
        log_responses: bool = True,
        log_request_body: bool = False,
        log_response_body: bool = False,
        skip_paths: list = None,
        max_body_size: int = 1024
    ):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.skip_paths = skip_paths or ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
        self.max_body_size = max_body_size
        
        logger.info("Logging middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with comprehensive logging
        """
        # Skip logging for certain paths
        if request.url.path in self.skip_paths:
            return await call_next(request)
        
        # Generate request ID if not present
        if not hasattr(request.state, 'request_id'):
            request.state.request_id = str(uuid.uuid4())
        
        request_id = request.state.request_id
        start_time = time.time()
        
        # Log request details
        if self.log_requests:
            await self._log_request(request, request_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response details
            if self.log_responses:
                await self._log_response(request, response, process_time, request_id)
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            await self._log_error(request, e, process_time, request_id)
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """
        Log detailed request information
        """
        try:
            # Basic request info
            request_info = {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
                "request_id": request_id
            }
            
            # Log request body if enabled
            if self.log_request_body:
                try:
                    body = await request.body()
                    if body:
                        # Truncate large bodies
                        if len(body) > self.max_body_size:
                            body = body[:self.max_body_size] + b"... [truncated]"
                        
                        # Try to parse as JSON
                        try:
                            request_info["body"] = json.loads(body.decode())
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            request_info["body"] = body.decode('utf-8', errors='replace')
                except Exception as e:
                    request_info["body_error"] = str(e)
            
            logger.info(
                f"Request: {request.method} {request.url.path}",
                extra={
                    'request_id': request_id,
                    'event_type': 'request_detail',
                    'request_info': request_info
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to log request: {e}")
    
    async def _log_response(self, request: Request, response: Response, process_time: float, request_id: str):
        """
        Log detailed response information
        """
        try:
            # Basic response info
            response_info = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "process_time": round(process_time, 4),
                "request_id": request_id
            }
            
            # Log response body if enabled
            if self.log_response_body and hasattr(response, 'body'):
                try:
                    body = response.body
                    if body:
                        # Truncate large bodies
                        if len(body) > self.max_body_size:
                            body = body[:self.max_body_size] + b"... [truncated]"
                        
                        # Try to parse as JSON
                        try:
                            response_info["body"] = json.loads(body.decode())
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            response_info["body"] = body.decode('utf-8', errors='replace')
                except Exception as e:
                    response_info["body_error"] = str(e)
            
            # Determine log level based on status code
            if response.status_code >= 500:
                log_level = logging.ERROR
            elif response.status_code >= 400:
                log_level = logging.WARNING
            else:
                log_level = logging.INFO
            
            logger.log(
                log_level,
                f"Response: {request.method} {request.url.path} | {response.status_code} | {process_time:.3f}s",
                extra={
                    'request_id': request_id,
                    'event_type': 'response_detail',
                    'response_info': response_info
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to log response: {e}")
    
    async def _log_error(self, request: Request, error: Exception, process_time: float, request_id: str):
        """
        Log error information
        """
        try:
            error_info = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "process_time": round(process_time, 4),
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown"
            }
            
            logger.error(
                f"Request Error: {request.method} {request.url.path} | {type(error).__name__}: {str(error)} | {process_time:.3f}s",
                extra={
                    'request_id': request_id,
                    'event_type': 'request_error',
                    'error_info': error_info
                },
                exc_info=True
            )
            
        except Exception as e:
            logger.error(f"Failed to log error: {e}")


class SecurityLoggingMiddleware(BaseHTTPMiddleware):
    """
    Security-focused logging middleware for suspicious activities
    """
    
    def __init__(self, app, suspicious_patterns: list = None):
        super().__init__(app)
        self.suspicious_patterns = suspicious_patterns or [
            "admin", "login", "password", "token", "auth",
            "sql", "script", "eval", "exec", "cmd",
            "..", "//", "\\", "~", "$", "`"
        ]
        
        logger.info("Security logging middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with security logging
        """
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        # Check for suspicious patterns
        await self._check_suspicious_activity(request, request_id)
        
        # Process request
        response = await call_next(request)
        
        # Log security-relevant responses
        await self._log_security_response(request, response, request_id)
        
        return response
    
    async def _check_suspicious_activity(self, request: Request, request_id: str):
        """
        Check for suspicious patterns in request
        """
        try:
            suspicious_indicators = []
            
            # Check URL path
            path = request.url.path.lower()
            for pattern in self.suspicious_patterns:
                if pattern in path:
                    suspicious_indicators.append(f"path contains '{pattern}'")
            
            # Check query parameters
            for param, value in request.query_params.items():
                param_lower = param.lower()
                value_lower = str(value).lower()
                
                for pattern in self.suspicious_patterns:
                    if pattern in param_lower or pattern in value_lower:
                        suspicious_indicators.append(f"query param '{param}' contains '{pattern}'")
            
            # Check headers
            for header, value in request.headers.items():
                header_lower = header.lower()
                value_lower = str(value).lower()
                
                for pattern in self.suspicious_patterns:
                    if pattern in header_lower or pattern in value_lower:
                        suspicious_indicators.append(f"header '{header}' contains '{pattern}'")
            
            # Log if suspicious activity detected
            if suspicious_indicators:
                logger.warning(
                    f"Suspicious activity detected: {request.method} {request.url.path}",
                    extra={
                        'request_id': request_id,
                        'client_ip': request.client.host if request.client else "unknown",
                        'user_agent': request.headers.get("user-agent", "unknown"),
                        'suspicious_indicators': suspicious_indicators,
                        'event_type': 'suspicious_activity'
                    }
                )
        
        except Exception as e:
            logger.error(f"Failed to check suspicious activity: {e}")
    
    async def _log_security_response(self, request: Request, response: Response, request_id: str):
        """
        Log security-relevant response information
        """
        try:
            # Log authentication failures
            if response.status_code == 401:
                logger.warning(
                    f"Authentication failure: {request.method} {request.url.path}",
                    extra={
                        'request_id': request_id,
                        'client_ip': request.client.host if request.client else "unknown",
                        'status_code': response.status_code,
                        'event_type': 'auth_failure'
                    }
                )
            
            # Log authorization failures
            elif response.status_code == 403:
                logger.warning(
                    f"Authorization failure: {request.method} {request.url.path}",
                    extra={
                        'request_id': request_id,
                        'client_ip': request.client.host if request.client else "unknown",
                        'status_code': response.status_code,
                        'event_type': 'authz_failure'
                    }
                )
            
            # Log rate limiting
            elif response.status_code == 429:
                logger.warning(
                    f"Rate limit exceeded: {request.method} {request.url.path}",
                    extra={
                        'request_id': request_id,
                        'client_ip': request.client.host if request.client else "unknown",
                        'status_code': response.status_code,
                        'event_type': 'rate_limit_exceeded'
                    }
                )
            
            # Log server errors
            elif response.status_code >= 500:
                logger.error(
                    f"Server error: {request.method} {request.url.path} | {response.status_code}",
                    extra={
                        'request_id': request_id,
                        'client_ip': request.client.host if request.client else "unknown",
                        'status_code': response.status_code,
                        'event_type': 'server_error'
                    }
                )
        
        except Exception as e:
            logger.error(f"Failed to log security response: {e}")


